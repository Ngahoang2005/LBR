"""Phan tich (dung tap test, chi de quyet dinh huong phat trien): cong khoang chenh tong quat hoa D vao replay cua CA.

D_t = [trung binh cac lop moi cua task t] Cov(x_test - mu_c) - C_c  (phan duong), tuc phan covariance ma dac trung
ngoai mau trai rong hon dac trung trong mau. Neu ban oracle nay khong hon HRC thi bo huong (ban trung thuc se uoc
luong D tren nua O cua adapter phu). Cung chuoi CA voi exps/hrc/xval.py (seed 1993, pre_ca_fc).
"""
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
sys.path.insert(0, "exps/hrc")
from xval import RIDGE, dev, load, mixing  # noqa: E402
from models.hrc import train_ca_batch  # noqa: E402

D = "exps/sts/dump/imageneta_RSIAT_raests_end"


def psd(A):
    w, V = torch.linalg.eigh(A)
    return (V * w.clamp_min(0)) @ V.T


def main(seeds=(0, 1, 2)):
    global EST0
    torch.backends.cuda.matmul.allow_tf32 = False
    nt = 10
    meta, tasks = {"K": [], "T": [], "mu_learn": []}, []
    for t in range(nt):
        r = load(D, t)
        K, T = r["known_classes"], r["total_classes"]
        meta["K"].append(K)
        meta["T"].append(T)
        meta["mu_learn"].append(r["class_means"].clone())
        meta["n"] = r["n_train_per_class"].clone()
        X, y = r["test_feat"].double(), r["test_label"]
        # chia test: nua uoc luong D (vi tri chan trong moi lop) / nua cham diem (le) - khong ro ri
        est = torch.zeros(len(y), dtype=torch.bool)
        for c in range(T):
            idx = torch.nonzero(y == c).squeeze(1)
            est[idx[0::2]] = True
        tasks_est = est
        X, y = X[est], y[est]
        Rs = [torch.cov((X[y == c] - r["class_means"][c].double()).T, correction=0) +
              torch.outer(X[y == c].mean(0) - r["class_means"][c].double(), X[y == c].mean(0) - r["class_means"][c].double())
              for c in range(K, T)]
        C = r["class_covs_full"][K:T].double() - RIDGE * torch.eye(768, dtype=torch.float64)
        Dt = psd(torch.stack(Rs).mean(0) - C.mean(0))
        tasks.append({"x": r["test_feat"][~tasks_est].float().to(dev), "y": r["test_label"][~tasks_est].to(dev), "fc": r.get("pre_ca_fc"),
                      "ca": r.get("ca_args"), "h0": r["head_logits"], "D": Dt.float()})
        print(f"task {t}: tr D = {torch.trace(Dt):.2f}, tr C tb = {torch.diagonal(C, dim1=1, dim2=2).sum(1).mean():.2f}",
              flush=True)
        if t == 0:
            EST0 = tasks_est.to(dev)
        del r
    cfgs = [("V2", 0.0, 0.0, 1.0, 0.0, "all"), ("HRC", -20.0, 256.0, 2.0, 0.0, "all"),
            ("V2+D", 0.0, 0.0, 1.0, 1.0, "all"), ("V2+D moi", 0.0, 0.0, 1.0, 1.0, "new"),
            ("HRC1+D", -20.0, 256.0, 1.0, 1.0, "all"), ("HRC2+D", -20.0, 256.0, 2.0, 1.0, "all"),
            ("V2+2D", 0.0, 0.0, 1.0, 2.0, "all")]
    mix = [mixing(meta, c[1], c[2]) if c[1] != 0 else mixing(meta, 0.0, 0.0) for c in cfgs]
    G = len(cfgs)
    acc = np.zeros((len(seeds), G, nt))
    for si, seed in enumerate(seeds):
        W_prev = None
        for t in range(nt):
            K, T = meta["K"][t], meta["T"][t]
            if t == 0:
                acc[si, :, 0] = (tasks[0]["h0"].float().argmax(1).to(dev)[~EST0] == tasks[0]["y"]).float().mean().item() * 100
                W_prev = tasks[1]["fc"]["heads.0.0.weight"].to(dev)[None].expand(G, -1, -1).clone()
                continue
            new = tasks[t]["fc"][f"heads.{t}.0.weight"].to(dev)
            W0 = torch.cat([W_prev[:, :K], new[None].expand(G, -1, -1)], 1)
            cov = load(D, t)["class_covs_full"]
            Ls = []
            for (M, rr), c in zip(mix, cfgs):
                C = cov.to(dev) - RIDGE * torch.eye(768, device=dev)
                S = (M[:T, :T].float().to(dev) @ C.reshape(T, -1)).reshape(T, 768, 768)
                del C
                S += rr[:T].float().to(dev)[:, None, None] * torch.eye(768, device=dev)
                S *= c[3]
                if c[4] > 0:
                    rows = slice(0, T) if c[5] == "all" else slice(K, T)
                    S[rows] += c[4] * tasks[t]["D"].to(dev)
                Ls.append(torch.linalg.cholesky(S.double()).float())
                del S
            Ls = torch.stack(Ls)
            tid = torch.arange(T) // (T - K)
            kap = (0.9 + (tid + 1).float() / (t + 1) * 0.1).to(dev)
            MU = (meta["mu_learn"][t].float().to(dev) * kap[:, None])[None].expand(G, -1, -1).contiguous()
            gen = torch.Generator(device=dev).manual_seed(1000 * seed + t)
            W = train_ca_batch(W0, MU, Ls, torch.arange(G, device=dev), torch.ones(G, device=dev),
                               torch.arange(T, device=dev).repeat_interleave(256), tasks[t]["ca"], gen)
            pred = torch.einsum("nd,gcd->gnc", F.normalize(tasks[t]["x"], dim=1), F.normalize(W, dim=-1)).argmax(-1)
            acc[si, :, t] = ((pred == tasks[t]["y"][None]).float().mean(1) * 100).cpu().numpy()
            W_prev = W
            del Ls
            torch.cuda.empty_cache()
        print(f"seed {seed}: " + " | ".join(f"{c[0]} {acc[si, i].mean():.2f}/{acc[si, i, -1]:.2f}" for i, c in enumerate(cfgs)),
              flush=True)
    for i, c in enumerate(cfgs):
        d = acc[:, i] - acc[:, 0]
        print(f"{c[0]:9s} Avg {acc[:, i].mean():.2f} Last {acc[:, i, -1].mean():.2f} | dAvg {d.mean():+.2f} "
              f"dLast {d[:, -1].mean():+.2f} | " + " ".join(f"{x:+.2f}" for x in d.mean(0)))


if __name__ == "__main__":
    main()
