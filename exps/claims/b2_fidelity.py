"""B2 (go/no-go): Gauss tai z + day xuoi co mo hinh hoa phan phoi lop tot hon Gauss tai f khong?

Model sau task 0 (chua co drift). Moi lop task 0: x' (z) va f = h(z) cua anh train (transform test), f cua anh test.
Train lai head (CA, cung quy trinh voi models/lbr.train_ca_push: moi epoch B mau/lop, C buoc batch B, SGD + cosine,
khoi tao tu head sau task 0) bang:
  a_real          feature that cua anh train (lay lai co hoan lai)
  b_f             N(mu_f, Sigma_f + 1e-3 I) tai f (nhu RSIAT)
  b2_f_borrow     tai f, Sigma vay hinh dang (cung n0, tau voi LBR)
  c_z_borrow      N(mu_z, Sigma_z vay hinh dang) tai z, day qua h (LBR, s = 1)
  c_z_borrow_s2   nhu c, s = 2 (cai dat CA cua LBR)
  d_z_noshrink    nhu c nhung khong shrinkage (Sigma_z mau + 1e-3 I)
Do: accuracy tren anh test cua lop task 0 (tb +- std qua cac seed CA); khoang cach giua feature gia (s = 1, 1024
mau/lop) va feature that tren mat cau (f chuan hoa): MMD^2 (RBF, bang thong = median khoang cach cap cua f that) va
sliced-W2, tb theo lop, so voi anh train (do khop) va anh test (giu ngoai); dong a_real vs test la san nhieu.

Chay: python exps/claims/b2_fidelity.py --config <json> --ckpt <task_0.pt> --tag <ten>
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from data.data_manager import DataManager  # noqa: E402
from models.lbr import RIDGE, borrowed_shapes, head_from_xprime, sample_cov, xprime  # noqa: E402
from utils.inc_net import SimpleVitNet  # noqa: E402


@torch.no_grad()
def class_features(vit, dm, c, source):
    ds = dm.get_dataset(np.arange(c, c + 1), source=source, mode="test")
    Z = torch.cat([xprime(vit, x.cuda()) for _, x, _ in DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)])
    return Z.float(), head_from_xprime(vit, Z).float()


def cholesky(S):
    return torch.linalg.cholesky(S.double().cuda()).float()


def gauss_sampler(MU, L, push_vit=None, s=1.0):
    """Mau (C x B x d) tu N(MU_c, s L_c L_c^T); push_vit: day qua head (mau o z)."""
    def sample(gen, B):
        C, d = MU.shape
        Z = torch.randn(C, B, d, device="cuda", generator=gen)
        X = MU[:, None, :] + (s ** 0.5) * torch.einsum("cij,cnj->cni", L, Z)
        if push_vit is not None:
            X = head_from_xprime(push_vit, X.reshape(-1, d)).float().reshape(C, B, d)
        return X
    return sample


def real_sampler(Ftr):
    def sample(gen, B):
        return torch.stack([F_[torch.randint(F_.size(0), (B,), device="cuda", generator=gen)] for F_ in Ftr])
    return sample


def train_ca(W0, sampler, a, gen):
    C, d = W0.shape
    B = a["samples"]
    W = W0.clone().requires_grad_(True)
    opt = torch.optim.SGD([W], lr=a["init_lr"], momentum=0.9, weight_decay=a["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a["ca_epochs"])
    Y_all = torch.arange(C, device="cuda").repeat_interleave(B)
    for _ in range(a["ca_epochs"]):
        with torch.no_grad():
            Fx = sampler(gen, B)
            p = torch.randperm(C * B, device="cuda", generator=gen)
            X = F.normalize(Fx.reshape(C * B, d)[p], dim=-1)
            Y = Y_all[p]
        for i in range(C):
            loss = F.cross_entropy(a["scale"] * F.linear(X[i * B:(i + 1) * B], F.normalize(W, dim=-1)),
                                   Y[i * B:(i + 1) * B])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
    return W.detach()


def accuracy(W, Fte, yte):
    return 100.0 * float((F.linear(F.normalize(Fte, dim=-1), F.normalize(W, dim=-1)).argmax(1) == yte).float().mean())


def mmd2(X, Y, bw):
    """MMD^2 khong chech, kernel RBF exp(-|x - y|^2 / (2 bw^2)); nan neu mot tap co < 2 diem."""
    if min(X.size(0), Y.size(0)) < 2:
        return float("nan")
    k = lambda A, B_: torch.exp(-torch.cdist(A, B_).pow(2) / (2 * bw ** 2))
    m, n = X.size(0), Y.size(0)
    Kxx, Kyy = k(X, X), k(Y, Y)
    return float((Kxx.sum() - Kxx.diagonal().sum()) / (m * (m - 1)) + (Kyy.sum() - Kyy.diagonal().sum()) / (n * (n - 1))
                 - 2 * k(X, Y).mean())


def sliced_w2(X, Y, dirs, q=100):
    lv = (torch.arange(q, device=X.device, dtype=X.dtype) + 0.5) / q
    qx = torch.quantile(X @ dirs.t(), lv, dim=0)
    qy = torch.quantile(Y @ dirs.t(), lv, dim=0)
    return float((qx - qy).pow(2).mean().sqrt())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--fid_samples", type=int, default=1024)
    o = ap.parse_args()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"b2_{o.tag}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s => %(message)s",
                        handlers=[logging.FileHandler(out + ".log", mode="w"), logging.StreamHandler(sys.stdout)])
    args = json.load(open(o.config))
    args["seed"] = args["seed"][0] if isinstance(args["seed"], list) else args["seed"]
    torch.manual_seed(1)
    dm = DataManager(args["dataset"], args["shuffle"], args["seed"], args["init_cls"], args["increment"])
    state = torch.load(o.ckpt, map_location="cpu", weights_only=False)
    if "class_order" in state:
        assert list(state["class_order"]) == list(dm._class_order), "thu tu lop khac checkpoint"
    C = dm.get_task_size(0)
    net = SimpleVitNet(args, True)
    net.update_fc(C)
    net.load_state_dict(state["net"])
    net = net.cuda().eval()
    vit = net.convnet
    W0 = torch.cat([h[0].weight for h in net.fc.heads]).detach().float().cuda()

    Ztr, Ftr, Fte, yte = [], [], [], []
    for c in range(C):
        z, f = class_features(vit, dm, c, "train")
        Ztr.append(z)
        Ftr.append(f)
        Fte.append(class_features(vit, dm, c, "test")[1])
        yte.append(torch.full((Fte[-1].size(0),), c, device="cuda"))
    Fte_all, yte = torch.cat(Fte), torch.cat(yte)
    d = Ztr[0].size(1)
    n0, tau = float(args.get("lbr_n0", 20.0)), float(args.get("lbr_tau", 256.0))
    logging.info("B2 %s: %d lop task 0, anh train/lop %d-%d, test/lop %d-%d; n0 %g, tau %g", o.tag, C,
                 min(len(f) for f in Ftr), max(len(f) for f in Ftr), min(len(f) for f in Fte),
                 max(len(f) for f in Fte), n0, tau)

    def borrowed(feats):
        mu = torch.stack([f.double().mean(0) for f in feats]).cpu()
        S, _ = borrowed_shapes([f.cpu() for f in feats], mu, torch.zeros(0, d, dtype=torch.float64),
                               torch.zeros(0, d, d), n0, tau, "cuda")
        return mu.float().cuda(), cholesky(S)

    def plain(feats):
        mu = torch.stack([f.double().mean(0) for f in feats]).float().cuda()
        return mu, torch.stack([cholesky(sample_cov(f) + RIDGE * torch.eye(d, device=f.device, dtype=torch.float64))
                                for f in feats])

    mf, Lf = plain(Ftr)
    mfb, Lfb = borrowed(Ftr)
    mz, Lz = borrowed(Ztr)
    mzp, Lzp = plain(Ztr)
    variants = {
        "a_real": real_sampler(Ftr),
        "b_f": gauss_sampler(mf, Lf),
        "b2_f_borrow": gauss_sampler(mfb, Lfb),
        "c_z_borrow": gauss_sampler(mz, Lz, vit),
        "c_z_borrow_s2": gauss_sampler(mz, Lz, vit, s=2.0),
        "d_z_noshrink": gauss_sampler(mzp, Lzp, vit),
    }
    a = {"scale": float(args["scale"]), "init_lr": float(args["init_lr"]), "weight_decay": float(args["weight_decay"]),
         "ca_epochs": int(args["ca_epochs"]), "samples": int(args.get("ca_samples", 1024))}

    Ftr_n = [F.normalize(f, dim=-1) for f in Ftr]
    Fte_n = [F.normalize(f, dim=-1) for f in Fte]
    pooled = torch.cat(Ftr_n)
    bw = float(torch.cdist(pooled, pooled)[~torch.eye(len(pooled), dtype=torch.bool, device="cuda")].median())
    g = torch.Generator(device="cuda").manual_seed(12345)
    dirs = F.normalize(torch.randn(256, d, device="cuda", generator=g), dim=-1)

    res = {"no_ca": {"acc": [accuracy(W0, Fte_all, yte)]}}
    for name, sampler in variants.items():
        accs = [accuracy(train_ca(W0, sampler, a, torch.Generator(device="cuda").manual_seed(s)), Fte_all, yte)
                for s in range(o.seeds)]
        r = {"acc": accs}
        if name == "a_real":     # san nhieu: anh train that vs anh test that
            r["mmd_test"] = float(np.nanmean([mmd2(x, y, bw) for x, y in zip(Ftr_n, Fte_n)]))
            r["sw2_test"] = float(np.mean([sliced_w2(x, y, dirs) for x, y in zip(Ftr_n, Fte_n)]))
        else:
            fake = F.normalize(sampler(torch.Generator(device="cuda").manual_seed(999), o.fid_samples), dim=-1)
            r["mmd_train"] = float(np.nanmean([mmd2(fake[c], Ftr_n[c], bw) for c in range(C)]))
            r["mmd_test"] = float(np.nanmean([mmd2(fake[c], Fte_n[c], bw) for c in range(C)]))
            r["sw2_train"] = float(np.mean([sliced_w2(fake[c], Ftr_n[c], dirs) for c in range(C)]))
            r["sw2_test"] = float(np.mean([sliced_w2(fake[c], Fte_n[c], dirs) for c in range(C)]))
        res[name] = r
        logging.info("%-14s acc %s", name, accs)

    logging.info("bang thong MMD %.4f; %d huong sliced-W2; %d seed CA; %d mau/lop cho do khoang cach", bw, len(dirs),
                 o.seeds, o.fid_samples)
    fmt = lambda v: "-" if v is None else f"{v:.4f}"
    logging.info("%-14s %14s %10s %10s %10s %10s", "bien the", "acc", "MMD2 train", "MMD2 test", "SW2 train", "SW2 test")
    for name, r in res.items():
        acc = np.array(r["acc"])
        logging.info("%-14s %7.2f +- %.2f %10s %10s %10s %10s", name, acc.mean(), acc.std(), fmt(r.get("mmd_train")),
                     fmt(r.get("mmd_test")), fmt(r.get("sw2_train")), fmt(r.get("sw2_test")))
    json.dump(res, open(out + ".json", "w"), indent=2)


if __name__ == "__main__":
    main()
