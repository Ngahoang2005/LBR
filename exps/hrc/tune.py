"""Tune offline cho HRC mot head (ImageNet-A seed 1993, dump V2, cung backbone): theta (n0, tau, s) cua replay va sieu
tham so CA (kappa0, so mau / lop / epoch, so epoch, lr, thang logit). Cung thu tuc CA voi base._ca_private
(models/hrc.train_ca_batch, generator 1000 * seed + t), nen ket qua seed 0 ~ ket qua online voi ca_seed 0.

Thong ke dung lai tu dump V2 theo exps/hrc/xval.py (van chuyen tuyen tinh cua STS; n0 tuyet doi). Head cu khoi tao tu
ket qua CA task truoc cua chinh cau hinh, head moi tu pre_ca_fc cua V2.

Dung: python exps/hrc/tune.py <ten_luoi> [--seeds 0 1 2] [--dump <dump_dir>]   (luoi dinh nghia trong GRIDS)
Ket qua: exps/hrc/tune/<ten_luoi>.json (dump khac mac dinh: exps/hrc/tune/<ten_dump>_<ten_luoi>.json) va bang Avg / Last (trung binh seed) + chenh lech so voi V2 (cung seed).
"""
import argparse
import itertools
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
sys.path.insert(0, "exps/hrc")
from models.hrc import train_ca_batch  # noqa: E402
from xval import RIDGE, load, mixing  # noqa: E402

dev = os.environ.get("TUNE_DEV", "cuda")
DUMP = "exps/sts/dump/imageneta_RSIAT_raests_end"
V2 = dict(n0=0.0, tau=0.0, s=1.0, k0=0.9, B=256, ep=10, lr=1.0, scale=30.0)
HRC = dict(V2, n0=20.0, tau=256.0, s=2.0)


def grid(base, **axes):
    keys = list(axes)
    return [dict(base, **dict(zip(keys, vals))) for vals in itertools.product(*[axes[k] for k in keys])]


GRIDS = {
    # sieu tham so CA cho V2 va HRC
    "ca": grid(V2, B=[256, 1024], scale=[20.0, 30.0, 40.0]) + grid(HRC, B=[256, 1024], scale=[20.0, 30.0, 40.0]),
    "kappa": grid(HRC, k0=[0.8, 0.85, 0.9, 0.95, 1.0]) + grid(V2, k0=[0.8, 0.9, 1.0]),
    "train": grid(HRC, ep=[10, 20], lr=[0.5, 1.0, 2.0]) + grid(V2, ep=[10, 20], lr=[1.0, 2.0]),
    "base": [V2],
    # doi chung cho LBR v2: vay hinh dang o f (n0 20, tau 256) khong noi rong (s 1) va co noi rong (s 2)
    "shape": [V2, dict(V2, n0=20.0, tau=256.0), HRC],
    "big": grid(HRC, B=[2048, 4096]) + grid(HRC, B=[1024], s=[1.5, 2.5, 3.0]) + grid(HRC, B=[1024], n0=[10.0, 30.0]),
}


def cfg_key(c):
    return ",".join(f"{k}={c[k]:g}" for k in ("n0", "tau", "s", "k0", "B", "ep", "lr", "scale"))


def load_all(DUMP=DUMP):
    meta, tasks = {"K": [], "T": [], "mu_learn": []}, []
    nt = len([f for f in os.listdir(DUMP) if f.startswith("task_")])
    for t in range(nt):
        r = load(DUMP, t)
        meta["K"].append(r["known_classes"])
        meta["T"].append(r["total_classes"])
        meta["mu_learn"].append(r["class_means"].clone())
        meta["n"] = r["n_train_per_class"].clone()
        tasks.append({"x": F.normalize(r["test_feat"].float().to(dev), dim=1), "y": r["test_label"].to(dev),
                      "fc": r.get("pre_ca_fc"), "ca": r.get("ca_args"), "h0": r["head_logits"],
                      "yc": r["test_label"]})
        del r
    return meta, tasks


@torch.no_grad()
def shape_chol(M, rho, cov):
    T, d = cov.shape[0], cov.shape[1]
    C = cov.to(dev) - RIDGE * torch.eye(d, device=dev)
    S = (M[:T, :T].float().to(dev) @ C.reshape(T, -1)).reshape(T, d, d)
    del C
    S += rho[:T].float().to(dev)[:, None, None] * torch.eye(d, device=dev)
    L, info = torch.linalg.cholesky_ex(S.double())
    assert int(info.abs().max()) == 0
    return L.float()


def run_group(cfgs, meta, tasks, seed, mixes, DUMP=DUMP):
    """Moi cau hinh trong nhom cung (B, ep, lr, scale): chay chung (cung mau chuan, cung hoan vi)."""
    G, nt = len(cfgs), len(meta["T"])
    shapes = sorted({(c["n0"], c["tau"]) for c in cfgs})
    L_idx = torch.tensor([shapes.index((c["n0"], c["tau"])) for c in cfgs], device=dev)
    sq = torch.tensor([math.sqrt(c["s"]) for c in cfgs], device=dev)
    c0 = cfgs[0]
    acc = np.zeros((G, nt))
    W_prev = None
    for t in range(nt):
        K, T = meta["K"][t], meta["T"][t]
        if t == 0:
            acc[:, 0] = (tasks[0]["h0"].float().argmax(1) == tasks[0]["yc"]).float().mean().item() * 100
            W_prev = tasks[1]["fc"]["heads.0.0.weight"].to(dev)[None].expand(G, -1, -1).clone()
            continue
        cov = load(DUMP, t)["class_covs_full"]
        Ls = torch.stack([shape_chol(*mixes[sh], cov) for sh in shapes])
        del cov
        tid = (torch.arange(T) // (T - K)).double()
        MU = torch.stack([(meta["mu_learn"][t].double() * (c["k0"] + (1 - c["k0"]) * (tid + 1) / (t + 1))[:, None])
                          .float() for c in cfgs]).to(dev)
        W0 = torch.cat([W_prev[:, :K], tasks[t]["fc"][f"heads.{t}.0.weight"].to(dev)[None].expand(G, -1, -1)], 1)
        a = tasks[t]["ca"]
        ca = {"scale": c0["scale"], "init_lr": a["init_lr"] * c0["lr"], "weight_decay": a["weight_decay"],
              "ca_epochs": int(c0["ep"]), "samples": int(c0["B"])}
        gen = torch.Generator(device=dev).manual_seed(1000 * seed + t)
        W = train_ca_batch(W0, MU, Ls, L_idx, sq, torch.arange(T, device=dev).repeat_interleave(int(c0["B"])), ca, gen)
        pred = torch.einsum("nd,gcd->gnc", tasks[t]["x"], F.normalize(W, dim=-1)).argmax(-1)
        acc[:, t] = ((pred == tasks[t]["y"][None]).float().mean(1) * 100).cpu().numpy()
        W_prev = W
        del Ls, W, MU, W0
        torch.cuda.empty_cache()
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--gmax", type=int, default=8)
    ap.add_argument("--dump", default=DUMP)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    cfgs = GRIDS[a.grid]
    if V2 not in cfgs:
        cfgs = [V2] + cfgs
    os.makedirs("exps/hrc/tune", exist_ok=True)
    tag = "" if a.dump == DUMP else os.path.basename(a.dump.rstrip("/")) + "_"
    out_path = f"exps/hrc/tune/{tag}{a.grid}.json"
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    meta, tasks = load_all(a.dump)
    mixes = {}
    for c in cfgs:
        if (c["n0"], c["tau"]) not in mixes:
            mixes[(c["n0"], c["tau"])] = mixing(meta, -c["n0"] if c["n0"] > 0 else 0.0, c["tau"])
    groups = {}
    for c in cfgs:
        groups.setdefault((c["B"], c["ep"], c["lr"], c["scale"]), []).append(c)
    for seed in a.seeds:
        for gk, gc in groups.items():
            todo = [c for c in gc if f"{seed}|{cfg_key(c)}" not in res]
            gmax = max(1, a.gmax * 256 // int(gk[0]))
            for i in range(0, len(todo), gmax):
                part = todo[i:i + gmax]
                t0 = time.time()
                acc = run_group(part, meta, tasks, seed, mixes, a.dump)
                for c, r in zip(part, acc):
                    res[f"{seed}|{cfg_key(c)}"] = r.tolist()
                json.dump(res, open(out_path, "w"))
                print(f"seed {seed} nhom {gk}: {len(part)} cau hinh, {time.time() - t0:.0f}s", flush=True)
    base = np.array([res[f"{s}|{cfg_key(V2)}"] for s in a.seeds])
    print(f"\n== {a.grid}: trung binh {len(a.seeds)} seed nhieu CA (V2 = {base.mean():.2f} / {base[:, -1].mean():.2f})")
    print(f"{'cau hinh':66s}  Avg    Last   dAvg   dLast  dAvg moi seed")
    rows = []
    for c in cfgs:
        A = np.array([res[f"{s}|{cfg_key(c)}"] for s in a.seeds])
        d = A - base
        rows.append((d.mean(), cfg_key(c), A.mean(), A[:, -1].mean(), d[:, -1].mean(), d.mean(1)))
    for r in sorted(rows, key=lambda r: -r[0]):
        print(f"{r[1]:66s} {r[2]:.2f}  {r[3]:.2f}  {r[0]:+.2f}  {r[4]:+.2f}  " + " ".join(f"{x:+.2f}" for x in r[5]))


if __name__ == "__main__":
    main()
