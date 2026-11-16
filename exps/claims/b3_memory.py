"""B3: bo nho LBR sai lech it hon o cac task sau? So voi oracle (feature that cua lop cu, chi de phan tich).

Tren cung mot quy dao backbone (checkpoint sau moi task, <ckpt_dir>/task_<t>.pt), dung song song cac bo nho lop cu:
  f_sdc     mean f dich bang SDC tai f (sigma 4), Sigma_f = cov mau + 1e-3 I (nhu RSIAT)       <- baseline
  z_sdc     mean z dich bang SDC tai z (sigma 4), Sigma_z vay hinh dang, day qua head hien tai  <- LBR
  z_oracle  mean z that (model hien tai), Sigma_z da luu, day xuoi                               <- oracle z
  tham chieu: f_oracle (mean f that, Sigma_f da luu), f_frozen / z_frozen (khong bu drift)
SDC dung anh train task moi (transform test) qua model truoc/sau, nhu models/RSIAT_adapter.py; lop moi: thong ke tren
anh train qua model hien tai, Sigma_z vay tu mean z da theo doi (nhu LBR). Moi task t >= 1, tren lop cu, so phan phoi
feature gia (mau tu bo nho, chuan hoa len mat cau) voi feature that cua anh train qua model t:
  goc va sai so giua mean (tren mat cau), sliced-W2, MMD^2 (bang thong = median khoang cach trong lop that), so voi
  anh train (cung anh da fit bo nho, co loi cho Sigma mau) va anh test (giu ngoai);
  accuracy sau CA chi dung bo nho do (lop moi lay thong ke moi cung vi tri; khoi tao head = prototype cua bo nho;
  s = 1, khong kappa; 'real' = CA tren feature that, can tren) tren anh test cua moi lop da hoc.
Them sai so theo doi mean tho: |mu_sdc - mu_that| / RMS |.| tai f va tai z.

Chay: python exps/claims/b3_memory.py --config <json> --ckpt_dir <dir> --tag <ten>
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b1_drift import load_vit  # noqa: E402
from b2_fidelity import mmd2, sliced_w2, train_ca  # noqa: E402
from data.data_manager import DataManager  # noqa: E402
from models.base import BaseLearner  # noqa: E402
from models.lbr import RIDGE, borrowed_shapes, head_from_xprime, sample_cov, xprime  # noqa: E402

MEMS = ["f_sdc", "z_sdc", "z_oracle", "f_oracle", "f_frozen", "z_frozen"]


@torch.no_grad()
def features(vit, dm, lo, hi, source):
    """z, f (float64, GPU) va nhan cua anh lop [lo, hi) (transform test, thu tu co dinh)."""
    ds = dm.get_dataset(np.arange(lo, hi), source=source, mode="test")
    Z, Y = [], []
    for _, x, y in DataLoader(ds, batch_size=64, shuffle=False, num_workers=4):
        Z.append(xprime(vit, x.cuda()))
        Y.append(y)
    Z = torch.cat(Z)
    return Z.double(), head_from_xprime(vit, Z).double(), torch.cat(Y).cuda()


def split(X, Y, lo, hi):
    return [X[Y == c] for c in range(lo, hi)]


def chol(S):
    return torch.stack([torch.linalg.cholesky(s.double().cuda()).float() for s in S])


def sampler(M, L, vit=None, chunk=16384):
    """Mau (C x B x d) tu N(M_c, L_c L_c^T); vit: mau o z, day qua head."""
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
            out[i:j] = X
        return out
    return sample


def real_sampler(Fs):
    def sample(gen, B):
        return torch.stack([f[torch.randint(f.size(0), (B,), device="cuda", generator=gen)].float() for f in Fs])
    return sample


def ca_accuracy(sample, a, Fte, yte, seed, preds=None, key=None):
    """accuracy (%) sau CA; neu preds (dict) duoc truyen, luu du doan tung anh test vao preds[key]."""
    with torch.no_grad():
        W0 = F.normalize(sample(torch.Generator(device="cuda").manual_seed(10_000 + seed), 256), dim=-1).mean(1)
    W = train_ca(W0, sample, a, torch.Generator(device="cuda").manual_seed(seed))
    pred = F.linear(F.normalize(Fte, dim=-1), F.normalize(W, dim=-1)).argmax(1)
    if preds is not None:
        preds[key] = pred.cpu().numpy()
    return 100.0 * float((pred == yte).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--eval_samples", type=int, default=2048)
    ap.add_argument("--max_tasks", type=int, default=None)
    o = ap.parse_args()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"b3_{o.tag}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s => %(message)s",
                        handlers=[logging.FileHandler(out + ".log", mode="w"), logging.StreamHandler(sys.stdout)])
    args = json.load(open(o.config))
    args["seed"] = args["seed"][0] if isinstance(args["seed"], list) else args["seed"]
    dm = DataManager(args["dataset"], args["shuffle"], args["seed"], args["init_cls"], args["increment"])
    T = min(dm.nb_tasks, o.max_tasks or dm.nb_tasks)
    ck = lambda t: os.path.join(o.ckpt_dir, f"task_{t}.pt")
    order = torch.load(ck(0), map_location="cpu", weights_only=False).get("class_order")
    assert order is None or list(order) == list(dm._class_order), "thu tu lop khac checkpoint"
    n0, tau = float(args.get("lbr_n0", 20.0)), float(args.get("lbr_tau", 256.0))
    a = {"scale": float(args["scale"]), "init_lr": float(args["init_lr"]), "weight_decay": float(args["weight_decay"]),
         "ca_epochs": int(args["ca_epochs"]), "samples": int(args.get("ca_samples", 1024))}
    sdc = lambda Y1, Y2, M: torch.as_tensor(BaseLearner.displacement(None, Y1.cpu().numpy(), Y2.cpu().numpy(),
                                                                     M.cpu().numpy(), 4.0)).cuda()
    g = torch.Generator(device="cuda").manual_seed(12345)
    dirs = None

    mem = {m: None for m in MEMS}           # mean cua lop da hoc (K x d, float64, GPU), theo bo nho
    Lf = Lz = Sz = None                     # Cholesky Sigma_f / Sigma_z (float32 GPU), Sigma_z (CPU, de vay)
    res = {}
    v_prev = None
    for t in range(T):
        v = load_vit(args, ck(t))
        K = sum(dm.get_task_size(s) for s in range(t))
        Kt = K + dm.get_task_size(t)
        Zn, Fn, Yn = features(v, dm, K, Kt, "train")
        if t > 0:                           # SDC tren anh task moi qua model t-1 -> t
            Zp, Fp, _ = features(v_prev, dm, K, Kt, "train")
            mem["f_sdc"] = mem["f_sdc"] + sdc(Fp, Fn, mem["f_sdc"])
            mem["z_sdc"] = mem["z_sdc"] + sdc(Zp, Zn, mem["z_sdc"])
            Zo, Fo, Yo = features(v, dm, 0, K, "train")
            Zo_c, Fo_c = split(Zo, Yo, 0, K), split(Fo, Yo, 0, K)
            mem["z_oracle"] = torch.stack([z.mean(0) for z in Zo_c])
            mem["f_oracle"] = torch.stack([f.mean(0) for f in Fo_c])
            track = {"f_rel": float((mem["f_sdc"] - mem["f_oracle"]).norm(dim=1).mean() / Fo.norm(dim=1).pow(2).mean().sqrt()),
                     "z_rel": float((mem["z_sdc"] - mem["z_oracle"]).norm(dim=1).mean() / Zo.norm(dim=1).pow(2).mean().sqrt()),
                     "f_frozen_rel": float((mem["f_frozen"] - mem["f_oracle"]).norm(dim=1).mean() / Fo.norm(dim=1).pow(2).mean().sqrt()),
                     "z_frozen_rel": float((mem["z_frozen"] - mem["z_oracle"]).norm(dim=1).mean() / Zo.norm(dim=1).pow(2).mean().sqrt())}
        # lop moi: thong ke qua model t; Sigma_z vay tu mean z da theo doi (nhu LBR)
        Zn_c, Fn_c = split(Zn, Yn, K, Kt), split(Fn, Yn, K, Kt)
        mz_new = torch.stack([z.mean(0) for z in Zn_c])
        mf_new = torch.stack([f.mean(0) for f in Fn_c])
        d = mz_new.size(1)
        mu_old = mem["z_sdc"].cpu() if t > 0 else torch.zeros(0, d, dtype=torch.float64)
        S_new, _ = borrowed_shapes([z.float().cpu() for z in Zn_c], mz_new.cpu(), mu_old,
                                   Sz if t > 0 else torch.zeros(0, d, d), n0, tau, "cuda")
        Sf_new = [sample_cov(f) + RIDGE * torch.eye(d, device="cuda", dtype=torch.float64) for f in Fn_c]
        Sz = S_new if t == 0 else torch.cat([Sz, S_new])
        Lz = chol(S_new) if t == 0 else torch.cat([Lz, chol(S_new)])
        Lf = chol(Sf_new) if t == 0 else torch.cat([Lf, chol(Sf_new)])
        if dirs is None:
            dirs = F.normalize(torch.randn(128, d, device="cuda", generator=g), dim=-1)
        for m in MEMS:
            new = mz_new if m.startswith("z") else mf_new
            mem[m] = new if t == 0 else torch.cat([mem[m], new])

        if t > 0:
            Zte, Fte, yte = features(v, dm, 0, Kt, "test")
            r = {"track": track}
            last = {"y_true": yte.cpu().numpy()} if t == T - 1 else None   # du doan task cuoi -> McNemar
            real = {"tr": [F.normalize(f.float(), dim=-1) for f in Fo_c],
                    "te": [F.normalize(f.float(), dim=-1) for f in split(Fte, yte, 0, K)]}
            bws = [float(torch.pdist(f).median()) if len(f) > 2 else 1.0 for f in real["tr"]]
            for m in MEMS:
                is_z = m.startswith("z")
                smp = sampler(mem[m].float(), Lz if is_z else Lf, v if is_z else None)
                fake = F.normalize(sampler(mem[m][:K].float(), (Lz if is_z else Lf)[:K], v if is_z else None)(
                    torch.Generator(device="cuda").manual_seed(999), o.eval_samples), dim=-1)
                r[m] = {"acc": ca_accuracy(smp, a, Fte.float(), yte, seed=t, preds=last, key=m)}
                for sp, R in real.items():
                    ok = [c for c in range(K) if len(R[c]) > 0]
                    r[m][f"mean_err_{sp}"] = float(np.mean([float((fake[c].mean(0) - R[c].mean(0)).norm()) for c in ok]))
                    r[m][f"angle_{sp}"] = float(np.mean([float(torch.rad2deg(torch.arccos(F.cosine_similarity(
                        fake[c].mean(0), R[c].mean(0), dim=0).clamp(-1, 1)))) for c in ok]))
                    r[m][f"sw2_{sp}"] = float(np.mean([sliced_w2(fake[c], R[c], dirs) for c in ok]))
                    r[m][f"mmd_{sp}"] = float(np.nanmean([mmd2(fake[c], R[c], bws[c]) for c in ok]))
                del fake
            r["real"] = {"acc": ca_accuracy(real_sampler(Fo_c + Fn_c), a, Fte.float(), yte, seed=t, preds=last, key="real")}
            res[t] = r
            logging.info("task %d (%d lop cu): goc mean (test, do) %s; acc %s", t, K,
                         " ".join(f"{m} {r[m]['angle_te']:.2f}" for m in MEMS[:3]),
                         " ".join(f"{m} {r[m]['acc']:.2f}" for m in MEMS[:3] + ["real"]))
        v_prev = v
        torch.cuda.empty_cache()

    def table(title, key, mems):
        logging.info("%s", title)
        logging.info("%-5s " + " ".join(f"{m:>9s}" for m in mems), "task")
        for t, r in res.items():
            logging.info("%-5d " + " ".join(f"{r[m][key]:9.4f}" for m in mems), t)
        logging.info("%-5s " + " ".join(f"{np.mean([r[m][key] for r in res.values()]):9.4f}" for m in mems), "tb")

    for sp, name in (("tr", "anh train (cung anh da fit bo nho)"), ("te", "anh test (giu ngoai)")):
        table(f"Goc (do) giua mean gia va mean that, {name} - thap hon tot hon", f"angle_{sp}", MEMS)
        table(f"Sai so mean tren mat cau, {name}", f"mean_err_{sp}", MEMS)
        table(f"Sliced-W2, {name}", f"sw2_{sp}", MEMS)
        table(f"MMD^2 (bang thong trong lop), {name}", f"mmd_{sp}", MEMS)
    table("Accuracy (%) sau CA chi dung bo nho", "acc", MEMS + ["real"])
    logging.info("Sai so theo doi mean tho |mu - mu_that| / RMS: tai f (SDC), tai z (SDC), khong bu f, khong bu z")
    for t, r in res.items():
        tr = r["track"]
        logging.info("%-5d %9.4f %9.4f %9.4f %9.4f", t, tr["f_rel"], tr["z_rel"], tr["f_frozen_rel"], tr["z_frozen_rel"])
    if last is not None:
        np.savez(out + "_pred.npz", **last)
    json.dump(res, open(out + ".json", "w"), indent=2)


if __name__ == "__main__":
    main()
