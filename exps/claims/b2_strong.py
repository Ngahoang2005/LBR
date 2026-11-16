"""B2 ban manh: Gauss tai z + day xuoi mo hinh hoa phan phoi lop tot hon Gauss tai f khong? (khong co drift)

Model sau task --task (mac dinh task cuoi), moi lop da hoc; thong ke tinh bang chinh model nay nen khong co drift.
Bien the (giong exps/claims/b2_fidelity.py): a_real (feature that), b_f (N tai f, cov mau + 1e-3 I nhu RSIAT),
b2_f_borrow (tai f, Sigma vay hinh dang), c_z_borrow (tai z, Sigma vay, day xuoi; s = 1), d_z_noshrink (tai z, cov
mau + 1e-3 I, day xuoi); hau to _s2: mau tu N(mu, 2 Sigma) (c_z_borrow_s2 = cai dat CA cua LBR; b_f_s2, b2_f_borrow_s2
la doi chung nong rong tai f). Vay hinh dang: moi lop vay tu cac lop con lai.
- Accuracy: thong ke fit tren toan bo anh train, CA tren moi lop (khoi tao head = prototype cua bien the, cung quy trinh
  voi models/lbr.train_ca_push), test tren anh test cua moi lop; tb +- std qua cac seed CA.
- Do khop phan phoi: chia doi anh train moi lop (A / B, co dinh), fit tren A, mau gia (s nhu bien the) so voi f that
  tren mat cau: voi A (trong mau) va voi phan giu ngoai B + anh test. MMD^2 khong chech (bang thong = median khoang cach
  trong lop cua f that), sliced-W2, goc giua mean; tb theo lop. a_real: f cua A so voi phan giu ngoai = san nhieu.

Chay: python exps/claims/b2_strong.py --config <json> --ckpt_dir <dir> --tag <ten> [--task 9]
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b1_drift import load_vit  # noqa: E402
from b2_fidelity import mmd2, sliced_w2  # noqa: E402
from b3_memory import ca_accuracy, chol, features, real_sampler, sampler, split  # noqa: E402
from data.data_manager import DataManager  # noqa: E402
from models.lbr import RIDGE, borrowed_shapes, sample_cov  # noqa: E402


def gaussians(Z, Fs, n0, tau):
    """mean va Cholesky (GPU) cua tung bien the Gauss tu feature theo lop (list z, list f)."""
    d = Z[0].size(1)
    eye = RIDGE * torch.eye(d, device="cuda", dtype=torch.float64)
    empty_mu, empty_S = torch.zeros(0, d, dtype=torch.float64), torch.zeros(0, d, d)

    def borrow(X):
        mu = torch.stack([x.mean(0) for x in X])
        S, _ = borrowed_shapes([x.float().cpu() for x in X], mu.cpu(), empty_mu, empty_S, n0, tau, "cuda")
        return mu.float(), chol(S)

    def plain(X):
        return torch.stack([x.mean(0) for x in X]).float(), chol([sample_cov(x) + eye for x in X])

    out = {"b_f": plain(Fs), "b2_f_borrow": borrow(Fs), "c_z_borrow": borrow(Z), "d_z_noshrink": plain(Z)}
    for k in list(out):
        out[k + "_s2"] = out[k]
    return out


def make_sampler(name, g, vit, Fs):
    if name == "a_real":
        return real_sampler(Fs)
    M, L = g[name]
    s = 2.0 if name.endswith("_s2") else 1.0
    return sampler(M, L * s ** 0.5, vit if "_z_" in name else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--fid_samples", type=int, default=2048)
    o = ap.parse_args()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"b2s_{o.tag}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s => %(message)s",
                        handlers=[logging.FileHandler(out + ".log", mode="w"), logging.StreamHandler(sys.stdout)])
    args = json.load(open(o.config))
    args["seed"] = args["seed"][0] if isinstance(args["seed"], list) else args["seed"]
    dm = DataManager(args["dataset"], args["shuffle"], args["seed"], args["init_cls"], args["increment"])
    t = dm.nb_tasks - 1 if o.task is None else o.task
    path = os.path.join(o.ckpt_dir, f"task_{t}.pt")
    order = torch.load(path, map_location="cpu", weights_only=False).get("class_order")
    assert order is None or list(order) == list(dm._class_order), "thu tu lop khac checkpoint"
    vit = load_vit(args, path)
    C = sum(dm.get_task_size(s) for s in range(t + 1))
    n0, tau = float(args.get("lbr_n0", 20.0)), float(args.get("lbr_tau", 256.0))
    a = {"scale": float(args["scale"]), "init_lr": float(args["init_lr"]), "weight_decay": float(args["weight_decay"]),
         "ca_epochs": int(args["ca_epochs"]), "samples": int(args.get("ca_samples", 1024))}

    Ztr, Ftr, ytr = features(vit, dm, 0, C, "train")
    _, Fte, yte = features(vit, dm, 0, C, "test")
    Z, Fs, Ft = split(Ztr, ytr, 0, C), split(Ftr, ytr, 0, C), split(Fte, yte, 0, C)
    gen = torch.Generator().manual_seed(0)
    perm = [torch.randperm(len(f), generator=gen).cuda() for f in Fs]
    half = [(len(f) + 1) // 2 for f in Fs]
    ZA = [z[p[:h]] for z, p, h in zip(Z, perm, half)]
    FA = [f[p[:h]] for f, p, h in zip(Fs, perm, half)]
    FH = [torch.cat([f[p[h:]], te]) for f, p, h, te in zip(Fs, perm, half, Ft)]
    logging.info("B2 manh %s: model task %d, %d lop; anh train/lop %d-%d (fit A = nua), giu ngoai (B + test)/lop %d-%d, "
                 "test %d anh; n0 %g, tau %g", o.tag, t, C, min(map(len, Fs)), max(map(len, Fs)), min(map(len, FH)),
                 max(map(len, FH)), len(yte), n0, tau)

    names = ["a_real", "b_f", "b2_f_borrow", "c_z_borrow", "d_z_noshrink", "b_f_s2", "b2_f_borrow_s2", "c_z_borrow_s2"]
    res = {n: {} for n in names}
    g_all = gaussians(Z, Fs, n0, tau)
    for n in names:
        res[n]["acc"] = [ca_accuracy(make_sampler(n, g_all, vit, Fs), a, Fte.float(), yte, seed=s) for s in range(o.seeds)]
        logging.info("%-14s acc %s", n, " ".join(f"{x:.2f}" for x in res[n]["acc"]))
    del g_all
    torch.cuda.empty_cache()

    real = {"fit": [F.normalize(f.float(), dim=-1) for f in FA], "ho": [F.normalize(f.float(), dim=-1) for f in FH]}
    bws = [float(torch.pdist(torch.cat([x, y])).median()) for x, y in zip(real["fit"], real["ho"])]
    dirs = F.normalize(torch.randn(128, Z[0].size(1), device="cuda",
                                   generator=torch.Generator(device="cuda").manual_seed(12345)), dim=-1)
    angle = lambda x, y: float(torch.rad2deg(torch.arccos(F.cosine_similarity(x.mean(0), y.mean(0), dim=0).clamp(-1, 1))))
    g_A = gaussians(ZA, FA, n0, tau)
    for n in names:
        if n == "a_real":
            fake = real["fit"]
        else:
            fake = []
            for i in range(0, C, 40):          # theo cum lop de gioi han bo nho
                M, L = (g_A[n][0][i:i + 40], g_A[n][1][i:i + 40] * (2.0 if n.endswith("_s2") else 1.0) ** 0.5)
                fake += list(F.normalize(sampler(M, L, vit if "_z_" in n else None)(
                    torch.Generator(device="cuda").manual_seed(999 + i), o.fid_samples), dim=-1))
        for sp in ("fit", "ho"):
            if n == "a_real" and sp == "fit":
                continue
            R = real[sp]
            res[n][f"mmd_{sp}"] = float(np.nanmean([mmd2(fake[c], R[c], bws[c]) for c in range(C)]))
            res[n][f"sw2_{sp}"] = float(np.mean([sliced_w2(fake[c], R[c], dirs) for c in range(C)]))
            res[n][f"angle_{sp}"] = float(np.mean([angle(fake[c], R[c]) for c in range(C)]))
        del fake
        torch.cuda.empty_cache()

    fmt = lambda v: "        -" if v is None else f"{v:9.4f}"
    logging.info("%-14s %15s | %9s %9s %9s | %9s %9s %9s", "bien the", "acc (%)", "MMD2 fit", "SW2 fit", "goc fit",
                 "MMD2 ho", "SW2 ho", "goc ho")
    for n in names:
        r, acc = res[n], np.array(res[n]["acc"])
        logging.info("%-14s %8.2f +- %.2f | %s %s %s | %s %s %s", n, acc.mean(), acc.std(), fmt(r.get("mmd_fit")),
                     fmt(r.get("sw2_fit")), fmt(r.get("angle_fit")), fmt(r.get("mmd_ho")), fmt(r.get("sw2_ho")),
                     fmt(r.get("angle_ho")))
    logging.info("fit = nua A (anh da fit), ho = giu ngoai (nua B + test); a_real ho = f cua A so voi giu ngoai (san nhieu)")
    json.dump(res, open(out + ".json", "w"), indent=2)


if __name__ == "__main__":
    main()
