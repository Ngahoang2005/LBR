"""LBR v2: hai phep do offline tren dump cua run quan sat (lbr_lambda 0, lbr_ca xprime: backbone trung V2), quyet dinh
co dang chay LBR v2 hay khong.

1. Ti so lac quan o tang x' va o f. Moi lop: phan tan quanh mean lop (mean tu anh train) cua anh test chia cho cua anh
   train, cung mo hinh; chia them (n+1)/(n-1) (ti so ky vong cua Gauss n mau khi khong lac quan), nen 1 = khong lac quan.
   Bao cao rieng lop moi cua task (vua train, lac quan nhat) va lop cu. f_norm: f da chuan hoa (hinh hoc ma CA dung).
2. CA tren dump, cung khoi tao (head truoc CA cua lop moi + chuoi head CA cua chinh phuong an), cung seed CA:
   f_gauss  : replay Gauss o f (V2: mean / cov STS o f, mean nhan kappa)
   xp_gauss : Gauss hoa o f phan phoi day xuoi tu Gauss x' (mean / cov Monte Carlo 2000 mau; = lbr_ca push)
   xp_push  : mau x' gia day xuoi qua h_theta, train truc tiep tren f gia (LBR v2, khong Gauss hoa)
   xp_gauss - f_gauss: loi ich cua bo nho o x'; xp_push - xp_gauss: loi ich cua phan phoi khong Gauss do mang tao ra.

Dung: python exps/lbr/xp_offline.py <dump_dir> <config.json> [--seeds 0 1 2] [--skip_ca]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from models.hrc import train_ca_batch  # noqa: E402
from models.lbr import push_stats, train_ca_push  # noqa: E402
from utils.inc_net import get_convnet  # noqa: E402

dev = os.environ.get("TUNE_DEV", "cuda")
METHODS = ("f_gauss", "xp_gauss", "xp_push")


def load(d, t):
    return torch.load(f"{d}/task_{t}.pt", map_location="cpu", weights_only=False)


@torch.no_grad()
def optimism(r):
    K, T = r["known_classes"], r["total_classes"]
    ytr, yte = r["train_label_all"], r["test_label"]
    ftr, fte = r["train_feat_all"].float(), r["test_feat"].float()
    out = {}
    for name, tr, te in (("x'", r["train_xp_all"], r["test_xp"]), ("f", ftr, fte),
                         ("f_norm", F.normalize(ftr, dim=1), F.normalize(fte, dim=1))):
        rat = torch.full((T,), float("nan"), dtype=torch.float64)
        for c in range(T):
            a, b = tr[ytr == c].double(), te[yte == c].double()
            n = a.size(0)
            if n < 3 or b.size(0) == 0:
                continue
            mu = a.mean(0)
            rat[c] = ((b - mu) ** 2).sum(1).mean() / ((a - mu) ** 2).sum(1).mean() / ((n + 1) / (n - 1))
        new, old = rat[K:T], rat[:K]
        out[name] = (float(new[~new.isnan()].mean()), float(old[~old.isnan()].mean()) if K > 0 else float("nan"))
    return out


def chol(S):
    L, info = torch.linalg.cholesky_ex(S.double().to(dev))
    assert int(info.abs().max()) == 0
    return L.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("config")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--skip_ca", action="store_true")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    nt = len(glob.glob(f"{a.dump}/task_*.pt"))
    print(f"== {a.dump}: {nt} task")

    print("\n[1] ti so lac quan (test / train, chia (n+1)/(n-1); 1 = khong lac quan): lop moi | lop cu")
    for t in range(nt):
        o = optimism(load(a.dump, t))
        print(f"  task {t}: " + "   ".join(f"{k} {v[0]:.3f} | {v[1]:.3f}" for k, v in o.items()), flush=True)
    if a.skip_ca or nt < 2:
        return

    vit = get_convnet(json.load(open(a.config))).to(dev).eval()
    res = np.zeros((len(a.seeds), len(METHODS), nt, 3))
    for si, seed in enumerate(a.seeds):
        W_prev = None
        for t in range(1, nt):
            r = load(a.dump, t)
            K, T = r["known_classes"], r["total_classes"]
            vit.blocks[-1].load_state_dict(r["last_block"])
            vit.norm.load_state_dict(r["vit_norm"])
            if W_prev is None:
                w0 = r["pre_ca_fc"]["heads.0.0.weight"].float().to(dev)
                W_prev = {m: w0.clone() for m in METHODS}
            Wn = r["pre_ca_fc"][f"heads.{t}.0.weight"].float().to(dev)
            tid = (torch.arange(T) // (T - K)).float()
            kap = 0.9 + 0.1 * (tid + 1) / (t + 1)
            ca = dict(r["ca_args"], samples=256)
            Y = torch.arange(T, device=dev).repeat_interleave(256)
            one, zero = torch.ones(1, device=dev), torch.zeros(1, dtype=torch.long, device=dev)
            x = F.normalize(r["test_feat"].float().to(dev), dim=1)
            y = r["test_label"].to(dev)
            L_xp = chol(r["xp_covs"])
            MU_xp = r["xp_means"].float().to(dev)
            for mi, m in enumerate(METHODS):
                W0 = torch.cat([W_prev[m][:K], Wn])
                gen = torch.Generator(device=dev).manual_seed(1000 * seed + t)
                if m == "f_gauss":
                    MU = (r["class_means"].double() * kap.double()[:, None]).float().to(dev)
                    L = chol(r["class_covs_full"])
                    W = train_ca_batch(W0[None], MU[None], L[None], zero, one, Y, ca, gen)[0]
                elif m == "xp_gauss":
                    mu_f, cov_f = push_stats(vit, r["xp_means"], r["xp_covs"], 2000, dev, seed=7919 * seed + t)
                    MU = (mu_f * kap.double()[:, None]).float().to(dev)
                    L = chol(cov_f + 1e-3 * torch.eye(cov_f.size(-1)))
                    W = train_ca_batch(W0[None], MU[None], L[None], zero, one, Y, ca, gen)[0]
                else:
                    W = train_ca_push(vit, W0, MU_xp, L_xp, kap.to(dev), ca, gen)
                p = (x @ F.normalize(W, dim=1).T).argmax(1)
                ok, old = (p == y).float(), y < K
                res[si, mi, t] = [100 * ok.mean().item(), 100 * ok[old].mean().item(), 100 * ok[~old].mean().item()]
                W_prev[m] = W
                if m != "xp_push":
                    del L
            del L_xp
            torch.cuda.empty_cache()
            print(f"  seed {seed} task {t}: " + "  ".join(f"{m} {res[si, mi, t, 0]:.2f}" for mi, m in enumerate(METHODS)),
                  flush=True)
    m = res.mean(0)
    print(f"\n[2] CA tren dump, trung binh {len(a.seeds)} seed (task 1..{nt - 1}; tong / cu / moi)")
    for mi, name in enumerate(METHODS):
        print(f"  {name:9s} " + " ".join(f"{v:.2f}" for v in m[mi, 1:, 0]) + f"  | avg {m[mi, 1:, 0].mean():.2f}"
              f"  old {m[mi, 1:, 1].mean():.2f}  new {m[mi, 1:, 2].mean():.2f}")
    for mi, name in enumerate(METHODS[1:], 1):
        d = res[:, mi, 1:, 0] - res[:, 0, 1:, 0]
        print(f"  {name} - f_gauss: " + " ".join(f"{v:+.2f}" for v in d.mean(0)) + f"  | avg {d.mean():+.2f}"
              f"  (moi seed " + " ".join(f"{v:+.2f}" for v in d.mean(1)) + ")")


if __name__ == "__main__":
    main()
