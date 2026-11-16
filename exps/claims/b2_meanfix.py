"""Do lech mean sau push-forward va cach sua (setup B2 ban manh: model task cuoi, moi lop, khong drift).

Bien the Gauss tai z (Sigma vay hinh dang, n0/tau nhu LBR), moi bien the voi s in {1, 2} (mau tu N(mu, s Sigma)):
  z        nhu LBR
  z_std    moi buoc uoc luong (mean, Sigma vay, jitter) trong toa do chuan hoa tung chieu (mean/std cua z moi anh
           train), mau anh xa nguoc ve z roi day xuoi (bien doi affine kha nghich, day xuoi van chinh xac)
  z_corr   chan doan: cong vector hieu chinh theo lop c = mu_f - mean h(z~) (tinh luc luu, cung s) vao f gia
  z_mm     moment matching: chon mu_z (Gauss-Newton qua Jacobian trung binh cua h) sao cho mean h(z~) = mu_f
Doi chung tai f: f (N tai f, cov mau + 1e-3 I), f_borrow (Sigma vay). Do: goc giua mean gia va mean that (anh fit,
tren mat cau), MMD^2 / SW2 voi phan giu ngoai, accuracy sau CA (nhu exps/claims/b2_strong.py).

Chay: python exps/claims/b2_meanfix.py --config <json> --ckpt_dir <dir> --tag <ten> [--task 9]
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jacrev, vmap

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b1_drift import load_vit  # noqa: E402
from b2_fidelity import mmd2, sliced_w2  # noqa: E402
from b3_memory import ca_accuracy, chol, features, split  # noqa: E402
from data.data_manager import DataManager  # noqa: E402
from models.lbr import RIDGE, borrowed_shapes, head_from_xprime, sample_cov  # noqa: E402


def fit(X, n0, tau, borrow=True):
    """mean (C x d, float64 GPU) va Sigma (C x d x d, float64 GPU) theo lop; vay hinh dang nhu LBR hoac cov mau."""
    d = X[0].size(1)
    mu = torch.stack([x.mean(0) for x in X])
    if borrow:
        S, _ = borrowed_shapes([x.float().cpu() for x in X], mu.cpu(), torch.zeros(0, d, dtype=torch.float64),
                               torch.zeros(0, d, d), n0, tau, "cuda")
        return mu, S.double().cuda()
    eye = RIDGE * torch.eye(d, device="cuda", dtype=torch.float64)
    return mu, torch.stack([sample_cov(x) + eye for x in X])


def push_mean(vit, M, L, eps):
    """mean h(M_c + L_c eps_i) (C x d) voi eps co dinh (n x d)."""
    out = []
    for c in range(M.size(0)):
        out.append(head_from_xprime(vit, (M[c] + eps @ L[c].t()).float()).double().mean(0))
    return torch.stack(out)


def moment_match(vit, M, L, mu_f, eps, eps_j, iters=4):
    """Gauss-Newton: mu <- mu + (J^T J + lam I)^-1 J^T (mu_f - mean h(mu + L eps)), J = mean Jacobian cua h."""
    h1 = lambda z: head_from_xprime(vit, z[None])[0]
    M = M.clone()
    hist = []
    for it in range(iters + 1):
        r = mu_f - push_mean(vit, M, L, eps)
        hist.append(float(r.norm(dim=1).mean() / mu_f.norm(dim=1).mean()))
        if it == iters:
            break
        for c in range(M.size(0)):
            J = vmap(jacrev(h1))((M[c] + eps_j @ L[c].t()).float()).double().mean(0)
            A = J.t() @ J
            lam = 1e-3 * float(torch.diagonal(A).mean())
            M[c] = M[c] + torch.linalg.solve(A + lam * torch.eye(A.size(0), device="cuda", dtype=torch.float64),
                                             J.t() @ r[c])
    return M, hist


def build(name, s, Z, Fs, vit, n0, tau, std, gens):
    """Bien the -> (sampler(gen, B), log). Mau o z day xuoi; z_corr cong vector hieu chinh; z_mm doi mean z."""
    mu_f = torch.stack([f.mean(0) for f in Fs])
    if name.startswith("f"):
        M, S = fit(Fs, n0, tau, borrow=name == "f_borrow")
        L = chol(S) * s ** 0.5
        return _sampler(M.float(), L, None, None), {}
    if name == "z_std":
        m, sd = std
        Mz, Sz = fit([(z - m) / sd for z in Z], n0, tau)
        M, S = m + sd * Mz, sd[:, None] * Sz * sd[None, :]
    else:
        M, S = fit(Z, n0, tau)
    L = chol(S).double() * s ** 0.5
    eps = torch.randn(1024, M.size(1), device="cuda", dtype=torch.float64, generator=gens)
    log = {"lech_tuong_doi_mean": float((mu_f - push_mean(vit, M, L, eps)).norm(dim=1).mean() / mu_f.norm(dim=1).mean())}
    corr = None
    if name == "z_corr":
        corr = (mu_f - push_mean(vit, M, L, eps)).float()
    elif name == "z_mm":
        eps_j = torch.randn(64, M.size(1), device="cuda", dtype=torch.float64, generator=gens)
        M, hist = moment_match(vit, M, L, mu_f, eps, eps_j)
        log["gauss_newton_lech_tuong_doi"] = hist
    return _sampler(M.float(), L.float(), vit, corr), log


def _sampler(M, L, vit, corr, chunk=16384):
    def sample(gen, B):
        C, d = M.shape
        out = torch.empty(C, B, d, device="cuda")
        cs = max(1, chunk // B)
        for i in range(0, C, cs):
            j = min(C, i + cs)
            X = M[i:j, None, :] + torch.einsum("cij,cnj->cni", L[i:j],
                                               torch.randn(j - i, B, d, device="cuda", generator=gen))
            if vit is not None:
                X = head_from_xprime(vit, X.reshape(-1, d)).float().reshape(j - i, B, d)
            if corr is not None:
                X = X + corr[i:j, None, :]
            out[i:j] = X
        return out
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--fid_samples", type=int, default=2048)
    o = ap.parse_args()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"b2m_{o.tag}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s => %(message)s",
                        handlers=[logging.FileHandler(out + ".log", mode="w"), logging.StreamHandler(sys.stdout)])
    args = json.load(open(o.config))
    args["seed"] = args["seed"][0] if isinstance(args["seed"], list) else args["seed"]
    dm = DataManager(args["dataset"], args["shuffle"], args["seed"], args["init_cls"], args["increment"])
    t = dm.nb_tasks - 1 if o.task is None else o.task
    vit = load_vit(args, os.path.join(o.ckpt_dir, f"task_{t}.pt"))
    for p in vit.parameters():
        p.requires_grad_(False)
    C = sum(dm.get_task_size(s) for s in range(t + 1))
    n0, tau = float(args.get("lbr_n0", 20.0)), float(args.get("lbr_tau", 256.0))
    a = {"scale": float(args["scale"]), "init_lr": float(args["init_lr"]), "weight_decay": float(args["weight_decay"]),
         "ca_epochs": int(args["ca_epochs"]), "samples": int(args.get("ca_samples", 1024))}

    Ztr, Ftr, ytr = features(vit, dm, 0, C, "train")
    _, Fte, yte = features(vit, dm, 0, C, "test")
    std = (Ztr.mean(0), Ztr.std(0))
    Z, Fs, Ft = split(Ztr, ytr, 0, C), split(Ftr, ytr, 0, C), split(Fte, yte, 0, C)
    gen = torch.Generator().manual_seed(0)
    perm = [torch.randperm(len(f), generator=gen).cuda() for f in Fs]
    half = [(len(f) + 1) // 2 for f in Fs]
    ZA = [z[p[:h]] for z, p, h in zip(Z, perm, half)]
    FA = [f[p[:h]] for f, p, h in zip(Fs, perm, half)]
    FH = [torch.cat([f[p[h:]], te]) for f, p, h, te in zip(Fs, perm, half, Ft)]
    real_fit = [F.normalize(f.float(), dim=-1) for f in FA]
    real_ho = [F.normalize(f.float(), dim=-1) for f in FH]
    bws = [float(torch.pdist(torch.cat([x, y])).median()) for x, y in zip(real_fit, real_ho)]
    dirs = F.normalize(torch.randn(128, Z[0].size(1), device="cuda",
                                   generator=torch.Generator(device="cuda").manual_seed(12345)), dim=-1)
    angle = lambda x, y: float(torch.rad2deg(torch.arccos(F.cosine_similarity(x.mean(0), y.mean(0), dim=0).clamp(-1, 1))))
    logging.info("B2 mean %s: model task %d, %d lop, test %d anh", o.tag, t, C, len(yte))

    res = {}
    for name in ["f", "f_borrow", "z", "z_std", "z_corr", "z_mm"]:
        for s in (1.0, 2.0):
            key = f"{name}_s{int(s)}"
            gens = torch.Generator(device="cuda").manual_seed(4242)
            smp, log = build(name, s, Z, Fs, vit, n0, tau, std, gens)       # fit tren toan bo anh train -> accuracy
            r = {"acc": [ca_accuracy(smp, a, Fte.float(), yte, seed=k) for k in range(o.seeds)], **log}
            del smp
            gens = torch.Generator(device="cuda").manual_seed(4242)
            smpA, _ = build(name, s, ZA, FA, vit, n0, tau, std, gens)       # fit tren nua A -> do khop
            fake = []
            for i in range(0, C, 40):
                part = smpA(torch.Generator(device="cuda").manual_seed(999 + i), o.fid_samples)
                fake += list(F.normalize(part, dim=-1))
            r["angle_fit"] = float(np.mean([angle(fake[c], real_fit[c]) for c in range(C)]))
            r["mmd_ho"] = float(np.nanmean([mmd2(fake[c], real_ho[c], bws[c]) for c in range(C)]))
            r["sw2_ho"] = float(np.mean([sliced_w2(fake[c], real_ho[c], dirs) for c in range(C)]))
            del fake, smpA
            torch.cuda.empty_cache()
            res[key] = r
            logging.info("%-10s acc %.2f +- %.2f, goc fit %.2f, MMD2 ho %.4f, SW2 ho %.4f %s", key, np.mean(r["acc"]),
                         np.std(r["acc"]), r["angle_fit"], r["mmd_ho"], r["sw2_ho"],
                         {k: v for k, v in log.items()})
    logging.info("%-10s %15s %9s %9s %9s", "bien the", "acc (%)", "goc fit", "MMD2 ho", "SW2 ho")
    for k, r in res.items():
        logging.info("%-10s %8.2f +- %.2f %9.2f %9.4f %9.4f", k, np.mean(r["acc"]), np.std(r["acc"]), r["angle_fit"],
                     r["mmd_ho"], r["sw2_ho"])
    json.dump(res, open(out + ".json", "w"), indent=2)


if __name__ == "__main__":
    main()
