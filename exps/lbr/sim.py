"""Mo phong offline LBR-FT (trunk dong bang sau task 0, chi khoi cuoi + head hoc) tren cache x' (exps/lbr/sim_cache.py).

Voi trunk dong bang, x'_CLS cua moi anh co dinh, nen moi task >= 1 chi phu thuoc khoi cuoi + head, dung nhu pipeline
online (models/RSIAT_adapter.py voi lbr_freeze_trunk): train khoi cuoi 15 epoch (CosFace lop moi + RAE + CE replay
LBR tren x' gia cua lop cu), Gauss x' lop moi (models/hrc.fixed_shapes), thong ke f: lop moi tu anh train, lop cu day
Gauss x' qua khoi cuoi hien tai (push), CA (models/hrc.train_ca_batch), danh gia tren test.
Moi lan bien doi anh train cua epoch e la view e cua cache (nhu online: dropout adapter trunk bat; mang cu cua RAE eval).

Them cac thanh phan de tach vai tro (khong co online):
  fake "real"      : replay x' THAT cua anh train lop cu (can tren cua co che replay; dung du lieu cu, chi de chan doan)
  stats "oracle"   : thong ke f lop cu tu anh train that (can tren cua thong ke cho CA)  | "stale": khong cap nhat
  extra_w > 0      : them adapter song song o khoi cuoi (be rong extra_w, khoi tao 0 o up_proj), hoc tu task 1
  mlp_lr > 0       : train ca fc1 / fc2 cua khoi cuoi (lr = init_lr * mlp_lr)
  new_stats "push" : thong ke f lop moi cung day tu Gauss x' (cung mo hinh hinh dang voi lop cu) thay vi covariance mau
Chan doan moi task (truoc CA, head cu): do chinh xac tren x' gia / x' that cua lop cu (khoang cach do trung thuc).

Dung: python exps/lbr/sim.py <ten_luoi> [--seeds 0 1 2]   (luoi trong GRIDS; ket qua exps/lbr/sim/<ten_luoi>.json)
"""
import argparse
import copy
import itertools
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, ".")
from models.hrc import fixed_shapes, train_ca_batch  # noqa: E402
from network.vision_transformer_adapter import Adapter  # noqa: E402
from utils.inc_net import SimpleVitNet  # noqa: E402
from utils.loss import AngularPenaltySMLoss  # noqa: E402
from utils.toolkit import AutoencoderSigmoid  # noqa: E402

dev = "cuda"
CFG = "exps/lbr/imageneta_lbr_ft10_s1993.json"
CKPT = "exps/lbr/ckpt/imageneta_s1993/task_0.pt"
CACHE = "exps/lbr/sim/cache_imageneta_s1993.pt"
# = cau hinh online lbr_ft10 (models/RSIAT_adapter.py, lbr_freeze_trunk)
FT10 = dict(lam=1.0, n_fake="balanced", s_fake=1.0, lr_mult=10.0, beta=3.0, real="full", new_head="train", fake="gauss",
            epochs=15, batch=48, extra_w=0, extra_scale=0.1, extra_lr_mult=10.0, mlp_lr=0.0, stats="push", new_stats="real",
            ca_B=256, ca_s=1.0, ca_ep=10, ca_lr=1.0, kappa0=0.9, n0=20.0, tau=256.0, push_mc=2000)


def grid(base, **axes):
    keys = list(axes)
    return [dict(base, **dict(zip(keys, vals))) for vals in itertools.product(*[axes[k] for k in keys])]


GRIDS = {
    "check": [FT10],
    # tach vai tro tung thanh phan quanh FT10 (bo / thay mot thanh phan)
    "ablate": [FT10, dict(FT10, lam=0.0), dict(FT10, beta=0.0), dict(FT10, lam=0.0, beta=0.0),
               dict(FT10, real="none"), dict(FT10, new_head="frozen"), dict(FT10, n_fake=48),
               dict(FT10, lr_mult=1.0), dict(FT10, lr_mult=3.0), dict(FT10, s_fake=2.0),
               dict(FT10, stats="stale"), dict(FT10, stats="oracle"), dict(FT10, fake="real"),
               dict(FT10, fake="real", stats="oracle"), dict(FT10, new_stats="push"),
               dict(FT10, new_stats="push", lam=0.0)],
}


def cfg_key(c):
    return ",".join(f"{k}={c[k]}" for k in sorted(c))


class Data:
    def __init__(self):
        args = json.load(open(CFG))
        args["seed"] = args["seed"][0]
        self.args = args
        C = torch.load(CACHE, map_location="cpu", weights_only=False)
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        assert C["class_order"] == ck["class_order"]
        net = SimpleVitNet(args, True)
        net.update_fc(args["init_cls"])
        net.load_state_dict(ck["net"])
        self.blk0 = copy.deepcopy(net.convnet.blocks[-1]).to(dev)
        self.norm0 = copy.deepcopy(net.convnet.norm).to(dev)
        self.W0 = net.fc.heads[0][0].weight.detach().clone().to(dev)
        del net
        self.ytr, self.yte = C["train_label"].to(dev), C["test_label"].to(dev)
        self.xtr, self.xte = C["train_test"].to(dev), C["test"].to(dev)
        self.vt, self.ve = C["views_train"].to(dev), C["views_eval"].to(dev)
        self.inc = C["increments"]
        self.bounds = np.cumsum([0] + self.inc)
        self.cls_idx = [torch.nonzero(self.ytr == c).flatten() for c in range(self.bounds[-1])]


def feat(m, xp):
    """f = LN(x' + MLP(LN2 x') + Adapter(x') [+ extra(x')]) cho x'_CLS (nhu models/lbr.head_from_xprime)."""
    blk, x = m["blk"], xp.float()[:, None, :]
    h = blk.mlp_drop(blk.act(blk.fc1(blk.norm2(x))))
    out = x + blk.drop_path(blk.mlp_drop(blk.fc2(h))) + blk.adaptmlp(x, add_residual=False)
    if m["extra"] is not None:
        out = out + m["extra"](x, add_residual=False)
    return m["norm"](out)[:, 0]


def set_mode(m, train):
    for v in m.values():
        if v is not None:
            v.train(train)


@torch.no_grad()
def f_stats(m, X):
    f = feat(m, X).double()
    return f.mean(0), torch.cov(f.T) + 1e-3 * torch.eye(f.size(1), device=f.device, dtype=f.dtype)


@torch.no_grad()
def push(m, mu, L, n, gen):
    """Mean / covariance cua f = khoi cuoi(x'), x' ~ N(mu, L L^T) (nhu models/lbr.push_stats)."""
    out_m, out_c = [], []
    for c in range(mu.size(0)):
        z = torch.randn(n, mu.size(1), device=dev, generator=gen)
        mm, cc = f_stats(m, mu[c] + z @ L[c].T)
        out_m.append(mm)
        out_c.append(cc)
    return torch.stack(out_m), torch.stack(out_c)


def sample_fake(mu, L, n):
    """n x' gia, nhan lop deu ngau nhien (nhu models/lbr.OldSampler), lay theo lop de khong gom L[y] (n x d x d)."""
    K, d = mu.shape
    cnt = torch.bincount(torch.randint(K, (n,), device=dev), minlength=K)
    kmax = int(cnt.max())
    Z = torch.randn(K, kmax, d, device=dev)
    X = mu[:, None, :] + torch.einsum("kij,knj->kni", L, Z)
    keep = torch.arange(kmax, device=dev)[None, :] < cnt[:, None]
    return X[keep], torch.arange(K, device=dev).repeat_interleave(cnt)


def chol(S):
    L, info = torch.linalg.cholesky_ex(S.double())
    assert int(info.abs().max()) == 0
    return L.float()


def evaluate(m, W, D, T, K):
    set_mode(m, False)
    keep = D.yte < T
    with torch.no_grad():
        f = feat(m, D.xte[keep])
        pred = (F.normalize(f, dim=1) @ F.normalize(W, dim=1).T).argmax(1)
    y = D.yte[keep]
    ok = (pred == y).float()
    old = y < K
    return (100 * ok.mean().item(), 100 * ok[old].mean().item() if K > 0 else 0.0, 100 * ok[~old].mean().item())


def run(cfg, D, seed):
    a = D.args
    torch.manual_seed(seed)
    m = {"blk": copy.deepcopy(D.blk0), "norm": D.norm0, "extra": None}
    for p in itertools.chain(m["blk"].parameters(), m["norm"].parameters()):
        p.requires_grad_(False)
    if cfg["extra_w"] > 0:
        m["extra"] = Adapter(None, d_model=768, bottleneck=int(cfg["extra_w"]), dropout=0.1, init_option="lora",
                             adapter_scalar=str(cfg["extra_scale"]), adapter_layernorm_option="none").to(dev)
    cosface = AngularPenaltySMLoss(loss_type="cosface", eps=1e-7, s=a["scale"], m=a["margin"])
    nt, B = len(D.inc), D.bounds
    # task 0: head da train online; Gauss x' va thong ke f cua lop task 0
    W = D.W0.clone()
    feats0 = [D.xtr[D.cls_idx[c]].float().cpu() for c in range(B[1])]
    mu_x = torch.stack([f.double().mean(0) for f in feats0])
    S_x = fixed_shapes(feats0, mu_x, torch.zeros(0, 768, dtype=torch.float64), torch.zeros(0, 768, 768),
                       cfg["n0"], cfg["tau"], dev)
    set_mode(m, False)
    st = [f_stats(m, D.xtr[D.cls_idx[c]]) for c in range(B[1])]
    mu_f, cov_f = torch.stack([s[0] for s in st]), torch.stack([s[1] for s in st])
    res = {"acc": [evaluate(m, W, D, B[1], 0)], "diag": []}
    ae = None
    for t in range(1, nt):
        K, T = int(B[t]), int(B[t + 1])
        old = {k: (copy.deepcopy(v).eval() if v is not None else None) for k, v in m.items()}
        if ae is None and cfg["beta"] > 0:
            ae = AutoencoderSigmoid(input_dims=768, code_dims=a["ae_code_dims"]).to(dev)
        W_new = torch.empty(T - K, 768, device=dev)
        nn.init.trunc_normal_(W_new, std=0.02)
        W_new.requires_grad_(True)
        lr, wd = a["init_lr"], a["weight_decay"]
        ad = list(m["blk"].adaptmlp.parameters())
        for p in ad:
            p.requires_grad_(True)
        groups = [{"params": ad, "lr": lr * cfg["lr_mult"], "weight_decay": wd},
                  {"params": [W_new], "lr": lr, "weight_decay": wd}]
        if m["extra"] is not None:
            groups.append({"params": list(m["extra"].parameters()), "lr": lr * cfg["extra_lr_mult"], "weight_decay": wd})
        if cfg["mlp_lr"] > 0:
            mp = list(m["blk"].fc1.parameters()) + list(m["blk"].fc2.parameters())
            for p in mp:
                p.requires_grad_(True)
            groups.append({"params": mp, "lr": lr * cfg["mlp_lr"], "weight_decay": wd})
        if ae is not None:
            groups.append({"params": list(ae.parameters()), "lr": a["ae_init_lr"], "weight_decay": a["ae_weight_decay"]})
        opt = torch.optim.SGD(groups, momentum=0.9)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=a["min_lr"])
        idx = torch.cat(D.cls_idx[K:T])
        W_old = W[:K].detach()
        L_fake = chol(S_x[:K].to(dev) * cfg["s_fake"])
        mu_fake = mu_x[:K].float().to(dev)
        old_idx = torch.cat(D.cls_idx[:K])
        V = D.vt.size(0)
        for e in range(cfg["epochs"]):
            set_mode(m, True)
            perm = idx[torch.randperm(idx.numel(), device=dev)]
            for b in perm.split(cfg["batch"]):
                y = D.ytr[b]
                f = feat(m, D.vt[e % V][b])
                loss = cosface(F.linear(F.normalize(f, dim=1), F.normalize(W_new, dim=1)), y - K)
                if ae is not None:
                    with torch.no_grad():
                        f_old = feat(old, D.ve[e % V][b])
                    loss = loss + cfg["beta"] * F.mse_loss(f, ae(f_old))
                if cfg["lam"] > 0:
                    n_fake = (max(1, int(round(b.numel() * K / (T - K)))) if cfg["n_fake"] == "balanced"
                              else int(cfg["n_fake"]))
                    if cfg["fake"] == "real":       # chan doan: x' that cua anh train lop cu (view ngau nhien)
                        j = old_idx[torch.randint(old_idx.numel(), (n_fake,), device=dev)]
                        xp, yo = D.vt[int(torch.randint(V, (1,)))][j], D.ytr[j]
                    else:
                        xp, yo = sample_fake(mu_fake, L_fake, n_fake)
                    Wn = W_new.detach() if cfg["new_head"] == "frozen" else W_new
                    fs, ys = [feat(m, xp)], [yo]
                    if cfg["real"] == "full":
                        fs.append(f)
                        ys.append(y)
                    logits = a["scale"] * F.linear(F.normalize(torch.cat(fs), dim=1),
                                                   F.normalize(torch.cat([W_old, Wn]), dim=1))
                    loss = loss + cfg["lam"] * F.cross_entropy(logits, torch.cat(ys))
                opt.zero_grad()
                loss.backward()
                opt.step()
            sch.step()
        set_mode(m, False)
        W_pre = torch.cat([W_old, W_new.detach()])
        # chan doan: head cu tren x' gia / x' that cua lop cu (truoc CA)
        with torch.no_grad():
            xg, yg = sample_fake(mu_fake, chol(S_x[:K].to(dev)), 20 * K)
            acc_g = (feat(m, xg) @ F.normalize(W_pre, dim=1).T).argmax(1).eq(yg).float().mean().item() * 100
            acc_r = (feat(m, D.xtr[old_idx]) @ F.normalize(W_pre, dim=1).T).argmax(1).eq(D.ytr[old_idx]).float()
            acc_r = acc_r.mean().item() * 100
        # Gauss x' lop moi; thong ke f
        feats = [D.xtr[D.cls_idx[c]].float().cpu() for c in range(K, T)]
        mu_n = torch.stack([f.double().mean(0) for f in feats])
        S_n = fixed_shapes(feats, mu_n, mu_x[:K], S_x[:K], cfg["n0"], cfg["tau"], dev)
        gen = torch.Generator(device=dev).manual_seed(1000 * seed + t)
        if cfg["stats"] == "push":
            mo, co = push(m, mu_x[:K].float().to(dev), chol(S_x[:K].to(dev)), cfg["push_mc"], gen)
        elif cfg["stats"] == "oracle":
            st = [f_stats(m, D.xtr[D.cls_idx[c]]) for c in range(K)]
            mo, co = torch.stack([s[0] for s in st]), torch.stack([s[1] for s in st])
        else:
            mo, co = mu_f[:K], cov_f[:K]
        if cfg["new_stats"] == "push":
            mn, cn = push(m, mu_n.float().to(dev), chol(S_n.to(dev)), cfg["push_mc"], gen)
        else:
            st = [f_stats(m, D.xtr[D.cls_idx[c]]) for c in range(K, T)]
            mn, cn = torch.stack([s[0] for s in st]), torch.stack([s[1] for s in st])
        mu_f, cov_f = torch.cat([mo, mn]), torch.cat([co, cn])
        mu_x, S_x = torch.cat([mu_x, mu_n]), torch.cat([S_x, S_n])
        # CA
        tid = torch.arange(T, device=dev) // (T - K)
        kap = (cfg["kappa0"] + (1 - cfg["kappa0"]) * (tid + 1).double() / (t + 1))
        ca = {"scale": a["scale"], "init_lr": lr * cfg["ca_lr"], "weight_decay": wd, "ca_epochs": int(cfg["ca_ep"]),
              "samples": int(cfg["ca_B"])}
        Ls = chol(cov_f)[None]
        W = train_ca_batch(W_pre[None], (mu_f * kap[:, None]).float()[None], Ls, torch.zeros(1, dtype=torch.long,
                           device=dev), torch.tensor([math.sqrt(cfg["ca_s"])], device=dev),
                           torch.arange(T, device=dev).repeat_interleave(int(cfg["ca_B"])), ca, gen)[0]
        del Ls
        res["acc"].append(evaluate(m, W, D, T, K))
        res["diag"].append({"acc_fake": acc_g, "acc_real_old_train": acc_r,
                            "pre_ca": evaluate(m, W_pre, D, T, K)})
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    D = Data()
    os.makedirs("exps/lbr/sim", exist_ok=True)
    out = f"exps/lbr/sim/{a.grid}.json"
    res = json.load(open(out)) if os.path.exists(out) else {}
    for c in GRIDS[a.grid]:
        for s in a.seeds:
            k = f"{s}|{cfg_key(c)}"
            if k in res:
                continue
            t0 = time.time()
            res[k] = run(c, D, s)
            json.dump(res, open(out, "w"))
            acc = [x[0] for x in res[k]["acc"]]
            print(f"seed {s} {time.time() - t0:.0f}s avg {np.mean(acc):.2f} last {acc[-1]:.2f} | "
                  + " ".join(f"{x:.2f}" for x in acc), flush=True)
    base = None
    print(f"\n== {a.grid} (trung binh {len(a.seeds)} seed)")
    for c in GRIDS[a.grid]:
        A = np.array([[x[0] for x in res[f"{s}|{cfg_key(c)}"]["acc"]] for s in a.seeds])
        O = np.array([[x[1] for x in res[f"{s}|{cfg_key(c)}"]["acc"]] for s in a.seeds])
        N = np.array([[x[2] for x in res[f"{s}|{cfg_key(c)}"]["acc"]] for s in a.seeds])
        G = np.array([[d["acc_fake"] - d["acc_real_old_train"] for d in res[f"{s}|{cfg_key(c)}"]["diag"]]
                      for s in a.seeds])
        if base is None:
            base = A
        diff = {k: v for k, v in c.items() if FT10.get(k) != v}
        print(f"{json.dumps(diff):46s} Avg {A.mean():.2f} Last {A[:, -1].mean():.2f} (dAvg {(A - base).mean():+.2f}"
              f" sd {A.mean(1).std():.2f}) | cu/moi cuoi {O[:, -1].mean():.2f}/{N[:, -1].mean():.2f} | "
              f"gia-that {G.mean():+.1f} | " + " ".join(f"{x:.2f}" for x in A.mean(0)))


if __name__ == "__main__":
    main()
