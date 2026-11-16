"""Danh gia offline cac cach dung U o head tren dump cua mot run V3 (can diag_replay va diag_v3_full).

Backbone, STS (buoc E, van chuyen) va hinh dang vay S giu nguyen nhu run; chi doi covariance cho head (CA, LDA, AGM)
va he so co mean kappa_c cua CA. CA duoc train lai offline (ban sao base._stage2_compact_classifier) voi cung nhieu
ngau nhien cho moi phuong an (so sanh theo cap). Phuong an <cov>|<kappa>:
  cov  : shape  (1 + 1/n) S          | volnorm (1 + 1/n) (S + U) giu det S | pred (1 + 1/n) S + U
  kappa: v2     0.9 + delta_c (V2)   | U  (1 + tr U_c / tr S_c)^(-1/2)    | 1  khong co mean
Dung: python exps/v3/head_variants.py <dump_dir> [cov|kappa ...]   (mac dinh: 5 phuong an duoi day)
"""
import glob
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from exps.agm.final_eval import AGM_M, LAMBDA_RATIO, lda_logits, z  # noqa: E402
from models.agm import AGMHead  # noqa: E402
from models.v3stats import head_covs  # noqa: E402

dev = "cuda"
DEFAULT = ["shape|v2", "volnorm|v2", "pred|v2", "shape|U", "shape|1"]


def kappa(mode, r, t):
    T, a = r["total_classes"], r["ca_args"]
    tid = torch.arange(T) // a["task_size"]
    if mode == "v2":
        return 0.9 + (tid + 1).double() / (t + 1) * 0.1
    if mode == "1":
        return torch.ones(T, dtype=torch.float64)
    assert mode == "U", mode
    return (1.0 + r["class_trU"] / r["class_trS"]).rsqrt()


def train_ca(pre_fc, means, covs, kap, t, a, seed):
    """Ban sao base._stage2_compact_classifier (nhu exps/sts/replay.py train_ca), mean nhan kap (T)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    W = torch.cat([pre_fc[f"heads.{i}.0.weight"] for i in range(t + 1)]).to(dev).clone().requires_grad_(True)
    T, d = W.shape
    I = torch.eye(d, dtype=torch.float64)
    Ls = torch.stack([torch.linalg.cholesky(c.double() + 1e-6 * I) for c in covs]).float().to(dev)
    opt = torch.optim.SGD([W], lr=a["init_lr"], momentum=0.9, weight_decay=a["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a["ca_epochs"])
    mu = means.float().to(dev) * kap.float().to(dev)[:, None]
    for _ in range(a["ca_epochs"]):
        X = (mu[:, None, :] + torch.einsum("cij,cnj->cni", Ls, torch.randn(T, 256, d, device=dev, generator=g))).reshape(-1, d)
        Y = torch.arange(T, device=dev).repeat_interleave(256)
        p = torch.randperm(len(Y), device=dev, generator=g)
        X, Y = X[p], Y[p]
        for i in range(T):
            x, y = X[i * 256:(i + 1) * 256], Y[i * 256:(i + 1) * 256]
            loss = F.cross_entropy(a["scale"] * F.linear(F.normalize(x, dim=1), F.normalize(W, dim=1)), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
    return W.detach()


def evaluate(r, t, cov_mode, kap_mode):
    means = r["class_means"].double()
    covs = head_covs(r["class_S_full"], r["class_U_full"], r["class_n"], cov_mode, True, dev)
    X = r["test_feat"].double()
    if t == 0:
        ca = r["head_logits"].double()
    else:
        W = train_ca(r["pre_ca_fc"], means, covs, kappa(kap_mode, r, t), t, r["ca_args"], seed=t)
        ca = F.linear(F.normalize(X.float().to(dev), dim=1), F.normalize(W, dim=1)).double().cpu()
    lda = lda_logits(means, covs.double(), X, r.get("lda_weight")).double()
    T = means.size(0)
    agm = AGMHead(X.size(1), M=AGM_M, device=dev, lin_gamma=1.0, bias=10.0)
    agm.fit(means, covs.double(), np.full(T, int(np.round(r["n_train_per_class"].numpy().mean()))))
    agm.solve(LAMBDA_RATIO * torch.trace(agm.G).item() / agm.M)
    la = agm.logits(r["test_feat"]).double().cpu()
    del agm
    torch.cuda.empty_cache()
    y, K = r["test_label"], r["known_classes"]
    out = {}
    for k, lg in (("CA", ca), ("LDA", lda), ("AGM", la), ("ENS", z(ca) + z(lda) + z(la))):
        p = lg.argmax(1)
        out[k] = 100.0 * (p == y).double().mean().item()
        if k == "ENS" and K > 0:
            out["ENS_old"] = 100.0 * (p[y < K] == y[y < K]).double().mean().item()
            out["ENS_new"] = 100.0 * (p[y >= K] == y[y >= K]).double().mean().item()
    return out


if __name__ == "__main__":
    d = sys.argv[1]
    variants = sys.argv[2:] or DEFAULT
    T_n = len(glob.glob(f"{d}/task_*.pt"))
    res = {v: [] for v in variants}
    for t in range(T_n):
        r = torch.load(f"{d}/task_{t}.pt", map_location="cpu", weights_only=False)
        for v in variants:
            res[v].append(evaluate(r, t, *v.split("|")))
        print(f"task {t}: " + " | ".join(f"{v} {res[v][-1]['ENS']:.2f}" for v in variants), flush=True)
    keys = ["CA", "LDA", "AGM", "ENS"]
    print("\n| phuong an | " + " | ".join(keys) + " | ENS cu / moi (TB task >= 1) |")
    print("|---|" + "---|" * (len(keys) + 1))
    for v in variants:
        R = res[v]
        cells = [f"{np.mean([x[k] for x in R]):.2f} / {R[-1][k]:.2f}" for k in keys]
        old = np.mean([x["ENS_old"] for x in R[1:]])
        new = np.mean([x["ENS_new"] for x in R[1:]])
        print(f"| {v} | " + " | ".join(cells) + f" | {old:.2f} / {new:.2f} |")
    print("(moi o: trung binh / task cuoi)")
