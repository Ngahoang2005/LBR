"""Kiem tra cheo offline mot theta co dinh cua HRC tren dump V2 bat ky (seed / dataset khac), so theo cap voi V2-CA.

theta = (rho, tau, s[, s_moi, kappa]): rho < 0 nghia la n0 tuyet doi = -rho; s_moi: s rieng cho lop cua task hien tai
(mac dinh = s); kappa: "rsiat" (mac dinh) hoac "one". n0_c = rho (n_c - 1) (tien nghiem nang bang rho lan du lieu), vay theo softmax(tau cos), replay
N(kappa mu, s S) voi s chung cho moi lop. Cach dung lai thong ke nhu exps/hrc/g0_offline.py (van chuyen tuyen tinh
cua STS). Khoi tao head cho CA: pre_ca_fc neu dump co, neu khong thi mean lop (chuan hoa) cho moi head - xap xi, dung
chung cho V2 va theta nen chenh lech theo cap van dung huong.

Dung: python exps/hrc/xval.py <dump_dir> [--init auto|mean] [--seeds 0 1 2] [--cands ten=rho,tau,s ...]
"""
import argparse
import glob
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from models.hrc import train_ca_batch  # noqa: E402

dev = "cuda"
RIDGE = 1e-3


def load(d, t):
    return torch.load(f"{d}/task_{t}.pt", map_location="cpu", weights_only=False)


def mixing(meta, rho, tau):
    Tn = meta["T"][-1]
    M = torch.zeros(Tn, Tn, dtype=torch.float64)
    r = torch.zeros(Tn, dtype=torch.float64)
    for t in range(len(meta["T"])):
        K, T = meta["K"][t], meta["T"][t]
        mu = F.normalize(meta["mu_learn"][t].double(), dim=1)
        for c in range(K, T):
            nc = float(meta["n"][c])
            n0 = -rho if rho < 0 else rho * (nc - 1.0)
            if n0 <= 0.0 or T < 2:
                M[c, c], r[c] = 1.0, RIDGE
                continue
            lg = tau * (mu[c] @ mu[:T].t())
            lg[c] = float("-inf")
            w = torch.softmax(lg, 0)
            a, b = n0 / (n0 + nc - 1.0), (nc - 1.0) / (n0 + nc - 1.0)
            row = a * (w[:K] @ M[:K])
            row[K:T] += a * w[K:T]
            row[c] += b
            M[c], r[c] = row, a * float(w[:K] @ r[:K]) + RIDGE
    return M, r


@torch.no_grad()
def chol_shapes(M, r, cov, s):
    T, d = cov.shape[0], cov.shape[1]
    C = cov.to(dev) - RIDGE * torch.eye(d, device=dev)
    S = (M[:T, :T].float().to(dev) @ C.reshape(T, -1)).reshape(T, d, d)
    del C
    S += r[:T].float().to(dev)[:, None, None] * torch.eye(d, device=dev)
    L, info = torch.linalg.cholesky_ex(S.double())
    assert int(info.abs().max()) == 0
    return L.float() * math.sqrt(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--init", default="auto")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--cands", nargs="+", default=["hrc=1,1024,2"])
    ap.add_argument("--ca", default="")               # init_lr,weight_decay,scale,ca_epochs neu dump khong co ca_args
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    nt = len(glob.glob(f"{a.dump}/task_*.pt"))
    meta = {"K": [], "T": [], "mu_learn": []}
    tasks = []
    for t in range(nt):
        r = load(a.dump, t)
        meta["K"].append(r["known_classes"])
        meta["T"].append(r["total_classes"])
        meta["mu_learn"].append(r["class_means"].clone())
        meta["n"] = r["n_train_per_class"].clone()
        tasks.append({"x": r["test_feat"].float().to(dev), "y": r["test_label"].to(dev), "fc": r.get("pre_ca_fc"),
                      "ca": r.get("ca_args"), "h0": r["head_logits"], "y_cpu": r["test_label"]})
        del r
    use_fc = a.init == "auto" and tasks[1]["fc"] is not None
    ca = tasks[1]["ca"]
    if ca is None:
        lr, wd, sc, ep = (float(v) for v in a.ca.split(","))
        ca = {"init_lr": lr, "weight_decay": wd, "scale": sc, "ca_epochs": int(ep)}
    names = ["V2"] + [c.split("=")[0] for c in a.cands]
    def th_parse(x):
        v = x.split(",")
        return (float(v[0]), float(v[1]), float(v[2]), float(v[3]) if len(v) > 3 else float(v[2]),
                v[4] if len(v) > 4 else "rsiat")
    ths = [(0.0, 0.0, 1.0, 1.0, "rsiat")] + [th_parse(c.split("=")[1]) for c in a.cands]
    mix = [mixing(meta, th[0], th[1]) for th in ths]
    G = len(ths)
    acc = np.zeros((len(a.seeds), G, nt))
    for si, seed in enumerate(a.seeds):
        W_prev = None
        for t in range(nt):
            K, T = meta["K"][t], meta["T"][t]
            if t == 0:
                acc[si, :, 0] = (tasks[0]["h0"].float().argmax(1) == tasks[0]["y_cpu"]).float().mean().item() * 100
                mu0 = meta["mu_learn"][0].float()
                W_prev = (tasks[1]["fc"]["heads.0.0.weight"] if use_fc else F.normalize(mu0, dim=1)).to(dev)
                W_prev = W_prev[None].expand(G, -1, -1).clone()
                continue
            new = (tasks[t]["fc"][f"heads.{t}.0.weight"] if use_fc else
                   F.normalize(meta["mu_learn"][t][K:T].float(), dim=1)).to(dev)
            W0 = torch.cat([W_prev[:, :K], new[None].expand(G, -1, -1)], 1)
            cov = load(a.dump, t)["class_covs_full"]
            Ls = torch.stack([chol_shapes(M, rr, cov, 1.0) for (M, rr), th in zip(mix, ths)])
            for g, th in enumerate(ths):                      # s cu (lop < K) va s moi (lop cua task t)
                Ls[g, :K] *= math.sqrt(th[2])
                Ls[g, K:] *= math.sqrt(th[3])
            del cov
            ts = T - K
            tid = torch.arange(T) // ts if meta["K"][1] == meta["T"][0] else torch.tensor(
                [0 if c < meta["T"][0] else 1 + (c - meta["T"][0]) // ts for c in range(T)])
            kap = (0.9 + (tid + 1).float() / (t + 1) * 0.1).to(dev)
            MU = torch.stack([meta["mu_learn"][t].float().to(dev) * (kap[:, None] if th[4] == "rsiat" else 1.0)
                              for th in ths])
            gen = torch.Generator(device=dev).manual_seed(1000 * seed + t)
            W = train_ca_batch(W0, MU, Ls, torch.arange(G, device=dev), torch.ones(G, device=dev),
                               torch.arange(T, device=dev).repeat_interleave(256), ca, gen)
            pred = torch.einsum("nd,gcd->gnc", F.normalize(tasks[t]["x"], dim=1), F.normalize(W, dim=-1)).argmax(-1)
            acc[si, :, t] = ((pred == tasks[t]["y"][None]).float().mean(1) * 100).cpu().numpy()
            W_prev = W
            del Ls, W
            torch.cuda.empty_cache()
        print(f"seed {seed}: " + " | ".join(f"{n} {acc[si, i].mean():.2f}/{acc[si, i, -1]:.2f}" for i, n in enumerate(names)),
              flush=True)
    m = acc.mean(0)
    print(f"== {a.dump} (init {'pre_ca_fc' if use_fc else 'mean'}, {len(a.seeds)} seed nhieu CA)")
    for i, n in enumerate(names):
        d = acc[:, i] - acc[:, 0]
        print(f"{n:8s} Avg {m[i].mean():.2f} Last {m[i][-1]:.2f} | dAvg {d.mean():+.2f} dLast {d[:, -1].mean():+.2f} | "
              "moi task " + " ".join(f"{x:+.2f}" for x in d.mean(0)))


if __name__ == "__main__":
    main()
