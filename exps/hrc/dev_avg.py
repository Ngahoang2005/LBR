"""Phan tich (seed 1993, offline): CA lay trung binh K head (moi head mot lan lay mau Gauss khac), so voi 1 head.
Cung chuoi CA voi exps/hrc/xval.py (pre_ca_fc). Head trung binh = trung binh cac vector trong so da chuan hoa."""
import math, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "."); sys.path.insert(0, "exps/hrc")
from xval import RIDGE, dev, load, mixing, chol_shapes  # noqa
from models.hrc import train_ca_batch  # noqa
D = "exps/sts/dump/imageneta_RSIAT_raests_end"
torch.backends.cuda.matmul.allow_tf32 = False
meta, tasks = {"K": [], "T": [], "mu_learn": []}, []
for t in range(10):
    r = load(D, t); meta["K"].append(r["known_classes"]); meta["T"].append(r["total_classes"])
    meta["mu_learn"].append(r["class_means"].clone()); meta["n"] = r["n_train_per_class"].clone()
    tasks.append({"x": r["test_feat"].float().to(dev), "y": r["test_label"].to(dev), "fc": r.get("pre_ca_fc"), "ca": r.get("ca_args"), "h0": r["head_logits"], "yc": r["test_label"]})
cfgs = [("V2", 0.0, 0.0, 1.0), ("HRC", -20.0, 256.0, 2.0)]
mix = [mixing(meta, c[1], c[2]) for c in cfgs]
Ks = (1, 4)
res = {}
for K_avg in Ks:
    for rep in range(3):
        W_prev = None; accs = np.zeros((len(cfgs), 10))
        for t in range(10):
            K, T = meta["K"][t], meta["T"][t]
            if t == 0:
                accs[:, 0] = (tasks[0]["h0"].float().argmax(1) == tasks[0]["yc"]).float().mean().item() * 100
                W_prev = tasks[1]["fc"]["heads.0.0.weight"].to(dev)[None].expand(len(cfgs), -1, -1).clone(); continue
            new = tasks[t]["fc"][f"heads.{t}.0.weight"].to(dev)
            W0 = torch.cat([W_prev[:, :K], new[None].expand(len(cfgs), -1, -1)], 1)
            cov = load(D, t)["class_covs_full"]
            Ls = torch.stack([chol_shapes(M, rr, cov, c[3]) for (M, rr), c in zip(mix, cfgs)])
            tid = torch.arange(T) // (T - K); kap = (0.9 + (tid + 1).float() / (t + 1) * 0.1).to(dev)
            MU = (meta["mu_learn"][t].float().to(dev) * kap[:, None])[None].expand(len(cfgs), -1, -1).contiguous()
            Wsum = 0
            for k in range(K_avg):
                gen = torch.Generator(device=dev).manual_seed(100000 * rep + 1000 * k + t)
                W = train_ca_batch(W0, MU, Ls, torch.arange(len(cfgs), device=dev), torch.ones(len(cfgs), device=dev),
                                   torch.arange(T, device=dev).repeat_interleave(256), tasks[t]["ca"], gen)
                Wsum = Wsum + F.normalize(W, dim=-1)
            W = Wsum / K_avg
            pred = torch.einsum("nd,gcd->gnc", F.normalize(tasks[t]["x"], dim=1), F.normalize(W, dim=-1)).argmax(-1)
            accs[:, t] = ((pred == tasks[t]["y"][None]).float().mean(1) * 100).cpu().numpy()
            W_prev = W; del Ls; torch.cuda.empty_cache()
        res[(K_avg, rep)] = accs
        print(f"K={K_avg} lan {rep}: " + " | ".join(f"{c[0]} {accs[i].mean():.2f}/{accs[i, -1]:.2f}" for i, c in enumerate(cfgs)), flush=True)
for K_avg in Ks:
    A = np.stack([res[(K_avg, r)] for r in range(3)])
    print(f"K={K_avg}: " + " | ".join(f"{c[0]} Avg {A[:, i].mean():.2f}+-{A[:, i].mean(1).std():.2f} Last {A[:, i, -1].mean():.2f}+-{A[:, i, -1].std():.2f}" for i, c in enumerate(cfgs)))
