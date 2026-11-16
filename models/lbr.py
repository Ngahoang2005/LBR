"""LBR (Last-Block Replay): bo nho lop cu la Gauss cua x'_CLS, dau vao cua nhanh MLP/Adapter khoi cuoi ViT.

Khoi cuoi (AdaptFormer song song): x' = x + Attn(LN1(x)), f = LN(x' + MLP(LN2(x')) + Adapter(x'))[CLS]. Moi phep sau Attn
tac dong rieng tung token nen f tinh dung tu x'_CLS (head_from_xprime).
- Lop moi: mu'_c tu anh train, Sigma'_c vay hinh dang (borrowed_shapes): co covariance mau ve tien nghiem
  sum_k a_ck Sigma'_k, a = softmax(tau cos(mu'_c, mu'_k)) tren lop cu va lop moi khac, n0 mau gia. Tuy chon: cos tinh
  sau khi tru tam chung (center), tau chon de so nguon hieu dung (perplexity) bang ban khong tru tam; jitter
  RIDGE * tr(Sigma)/d thay cho RIDGE co dinh.
- Lop cu: mu' dich theo SDC tinh tren x' cua anh task moi (models/RSIAT_adapter.py); bang thong co dinh hoac
  thich nghi (sdc_bandwidth).
- CA: x' gia ~ N(mu'_c, s Sigma'_c) day qua khoi cuoi hien tai -> f gia, train lai head (train_ca_push).
"""
import torch
import torch.nn.functional as F

RIDGE = 1e-3


def xprime(vit, images):
    """x'_CLS cua khoi cuoi (N x d): forward qua khoi 1..L-1 va nhanh attention cua khoi cuoi."""
    B = images.shape[0]
    x = vit.patch_embed(images)
    x = torch.cat((vit.cls_token.expand(B, -1, -1), x), dim=1)
    x = vit.pos_drop(x + vit.pos_embed)
    for blk in vit.blocks[:-1]:
        x = blk(x)
    blk = vit.blocks[-1]
    x = x + blk.drop_path(blk.attn(blk.norm1(x)))
    return x[:, 0]


def head_from_xprime(vit, xp):
    """f = LN(x' + MLP(LN2(x')) + Adapter(x')) cho x'_CLS (N x d): dung nhu Block.forward (song song) + norm cuoi."""
    blk = vit.blocks[-1]
    x = xp[:, None, :]
    adapt = blk.adaptmlp(x, add_residual=False)
    h = blk.mlp_drop(blk.act(blk.fc1(blk.norm2(x))))
    h = blk.drop_path(blk.mlp_drop(blk.fc2(h)))
    return vit.norm(x + h + adapt)[:, 0]


# ---------------------------------------------------------------------- vay hinh dang
def sample_cov(X):
    X = X.double()
    if X.size(0) < 2:
        return torch.zeros(X.size(1), X.size(1), dtype=torch.float64, device=X.device)
    return torch.cov(X.t())


def shrink(cov_hat, n, prior, n0, jitter_rel=False):
    """[n0 prior + (n - 1) cov_hat] / (n0 + n - 1) + j I; n0 = 0 hoac khong co nguon: cov_hat + j I.
    j = RIDGE (co dinh) hoac RIDGE * tr(.)/d (jitter_rel, theo thang cua Sigma)."""
    if n0 <= 0.0 or prior is None:
        C = cov_hat
    else:
        w = max(float(n) - 1.0, 0.0)
        C = (n0 * prior + w * cov_hat) / (n0 + w)
    j = RIDGE * float(torch.diagonal(C).mean()) if jitter_rel else RIDGE
    return C + torch.eye(cov_hat.size(-1), device=cov_hat.device) * j


def borrow_cos(mu_q, mu_src, exclude):
    """cos(mu_q[c], mu_src[k]) (C x K, float64); exclude[c] >= 0: nguon cua chinh lop c dat -inf."""
    cos = F.normalize(mu_q.double(), dim=1) @ F.normalize(mu_src.double(), dim=1).t()
    exclude = exclude.to(cos.device)
    rows = torch.nonzero(exclude >= 0).squeeze(1)
    cos[rows, exclude[rows]] = float("-inf")
    return cos


def perplexity(A):
    """So nguon hieu dung exp(entropy) cua moi hang A (C x K)."""
    return torch.exp(-(A * torch.log(A.clamp_min(1e-300))).sum(1))


def match_tau(cos, target, lo=1e-2, hi=1e6, iters=80):
    """tau sao cho perplexity tb cua softmax(tau cos) bang target (giam don dieu theo tau), chia doi theo log."""
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        if float(perplexity(torch.softmax(mid * cos, 1)).mean()) > target:
            lo = mid
        else:
            hi = mid
    return (lo * hi) ** 0.5


@torch.no_grad()
def prior_covs(A, src_cov, device, chunk=64):
    """sum_k A[:, k] src_cov[k] (C x d x d, float64, tren device). src_cov: K x d x d (CPU)."""
    C, K = A.shape
    d = src_cov.size(-1)
    A = A.to(device, torch.float64)
    out = torch.zeros(C, d * d, device=device, dtype=torch.float64)
    for i in range(0, K, chunk):
        out += A[:, i:i + chunk] @ src_cov[i:i + chunk].to(device, torch.float64).reshape(-1, d * d)
    return out.view(C, d, d)


@torch.no_grad()
def borrowed_shapes(feats, mu_new, mu_old, S_old, n0, tau, device, center=None, tau_match=False, jitter_rel=False):
    """Sigma' (C x d x d, float32, CPU) cua C lop moi va info (tau, perplexity tb). feats: list x' (n_c x d); nguon vay:
    S_old cua lop cu va covariance mau cua cac lop moi khac (lop co >= 2 anh). center (d,): cos tinh tren mean da
    tru center; tau_match: tau chon de perplexity tb bang ban khong tru tam voi tau da cho."""
    n = [X.size(0) for X in feats]
    cov_hat = [sample_cov(X) for X in feats]
    P, info = None, {}
    if n0 > 0.0:
        C, K = len(feats), mu_old.size(0)
        valid = [c for c in range(C) if n[c] >= 2]
        src_mu = torch.cat([mu_old.double(), mu_new.double()[valid]]).to(device)
        src_cov = torch.cat([S_old.float()] + ([torch.stack([cov_hat[c] for c in valid]).float()] if valid else []))
        self_index = torch.full((C,), -1, dtype=torch.long)
        for j, c in enumerate(valid):
            self_index[c] = K + j
        if src_cov.size(0) >= 2:
            q = mu_new.to(device).double()
            cos = borrow_cos(q, src_mu, self_index)
            ppl_raw = float(perplexity(torch.softmax(tau * cos, 1)).mean())
            if center is not None:
                ctr = center.to(device, torch.float64)
                cos = borrow_cos(q - ctr, src_mu - ctr, self_index)
                if tau_match:
                    tau = match_tau(cos, ppl_raw)
            A = torch.softmax(tau * cos, 1)
            info = {"tau": tau, "ppl": float(perplexity(A).mean()), "ppl_raw": ppl_raw}
            P = prior_covs(A, src_cov, device).cpu()
    S = torch.empty(len(feats), cov_hat[0].size(0), cov_hat[0].size(1), dtype=torch.float32)
    for c in range(len(feats)):
        S[c] = shrink(cov_hat[c], float(n[c]), None if P is None else P[c], n0, jitter_rel).float()
    return S, info


# ---------------------------------------------------------------------- SDC o z
def sdc_bandwidth(Y_old, mu_old):
    """Bang thong thich nghi: median theo lop cu cua khoang cach tu mu_c toi anh task moi gan nhat (qua model cu)."""
    D = torch.cdist(torch.as_tensor(mu_old, dtype=torch.float64), torch.as_tensor(Y_old, dtype=torch.float64))
    return float(D.min(1).values.median())


def kernel_ess(Y_old, mu_old, sigma):
    """So anh hieu dung (sum W)^2 / sum W^2 cua trong so SDC moi lop cu (W co san 1e-5 nhu displacement)."""
    D2 = torch.cdist(torch.as_tensor(mu_old, dtype=torch.float64), torch.as_tensor(Y_old, dtype=torch.float64)) ** 2
    W = torch.exp(-D2 / (2 * sigma ** 2)) + 1e-5
    return W.sum(1) ** 2 / (W ** 2).sum(1)


# ---------------------------------------------------------------------- CA o tang x'
def train_ca_push(vit, W0, MU, L, kap, a, gen, chunk=16384):
    """Moi epoch sinh B x' gia moi lop tu N(mu'_c, Sigma'_c) (RNG rieng gen), day qua khoi cuoi hien tai -> f gia, roi
    train lai head cosine (moi epoch C buoc, batch B, SGD + cosine). kap (C,) hoac None: dich f gia cua lop c them
    (kap_c - 1) * mean mau cua lop do. W0: C x d; MU: C x d; L: C x d x d (Cholesky cua Sigma')."""
    C, d = MU.shape
    B = int(a.get("samples", 256))
    W = W0.clone().requires_grad_(True)
    opt = torch.optim.SGD([W], lr=a["init_lr"], momentum=0.9, weight_decay=a["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a["ca_epochs"])
    Y_all = torch.arange(C, device=W.device).repeat_interleave(B)
    was = vit.training
    vit.eval()
    cs = max(1, chunk // B)
    for _ in range(a["ca_epochs"]):
        with torch.no_grad():
            Fx = torch.empty(C, B, d, device=W.device)
            for i in range(0, C, cs):
                j = min(C, i + cs)
                Z = torch.randn(j - i, B, d, device=W.device, generator=gen)
                X = (MU[i:j, None, :] + torch.einsum("cij,cnj->cni", L[i:j], Z)).reshape(-1, d)
                Fx[i:j] = head_from_xprime(vit, X).float().reshape(j - i, B, d)
            del Z, X
            if kap is not None:
                Fx = Fx + (kap.to(Fx) - 1.0)[:, None, None] * Fx.mean(1, keepdim=True)
            p = torch.randperm(C * B, device=W.device, generator=gen)
            X = F.normalize(Fx.reshape(C * B, d)[p], dim=-1)
            Y = Y_all[p]
            del Fx
        for i in range(C):
            x, y = X[i * B:(i + 1) * B], Y[i * B:(i + 1) * B]
            logits = a["scale"] * F.linear(x, F.normalize(W, dim=-1))
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
    vit.train(was)
    return W.detach()
