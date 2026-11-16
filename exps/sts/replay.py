"""Chay lai STS offline tren dump cua mot run co ghi ("sts_record" va "diag_replay"), cung quy dao backbone.

STS khong anh huong train backbone, nen moi cach bam thong ke lop cu co the danh gia tren cung mot run. Moi cach
co chuoi thong ke rieng qua cac task:
  start  : khong bu (thong ke lop cu giu nguyen)
  sts    : STS voi cau hinh tren dong lenh (mac dinh = mac dinh online), dung chung code voi models/sts.py
  oracle : thong ke that cua lop cu (feature train qua model hien tai), chi lam can tren
  online : thong ke va head CA trong dump (ket qua cua chinh run)
Moi task: CA (ban sao base._stage2_compact_classifier, train lai tren thong ke cua cach do; "online" dung head trong
dump), LDA, AGM (--agm), z-score ensemble; sai so mean lop cu so voi thong ke that.

Dung: python exps/sts/replay.py <dump_dir> [--agm] [--name ten] [sts_khoa=gia_tri ...]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "exps", "agm"))
from final_eval import AGM_M, LAMBDA_RATIO, acc, lda_logits, z  # noqa: E402
from models.agm import AGMHead  # noqa: E402
from models.sts import half_split, initial_estep, pool_estep, select_adversarial, sts_config, update  # noqa: E402

dev = "cuda"


def class_stats(feat, label, classes):
    """Mean va covariance (+1e-3 I) moi lop, nhu base._compute_class_mean."""
    feat = feat.double()
    I = torch.eye(feat.size(1), dtype=torch.float64)
    means = torch.stack([feat[label == c].mean(0) for c in classes])
    covs = torch.stack([torch.cov(feat[label == c].t()) + 1e-3 * I for c in classes]).float()
    return means, covs


def train_ca(pre_fc, means, covs, t, a, seed=0):
    """Ban sao base._stage2_compact_classifier: head cosine, SGD, LR cosine, 256 mau Gaussian/lop/epoch."""
    g = torch.Generator(device=dev).manual_seed(seed)
    W = torch.cat([pre_fc[f"heads.{i}.0.weight"] for i in range(t + 1)]).to(dev).clone().requires_grad_(True)
    T, d = W.shape
    I = torch.eye(d, dtype=torch.float64)
    Ls = torch.stack([torch.linalg.cholesky(c.double() + 1e-6 * I) for c in covs]).float().to(dev)
    opt = torch.optim.SGD([W], lr=a["init_lr"], momentum=0.9, weight_decay=a["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a["ca_epochs"])
    tid = torch.arange(T, device=dev) // a["task_size"]
    mu = means.float().to(dev) * (0.9 + (tid + 1).float() / (t + 1) * 0.1)[:, None]
    for _ in range(a["ca_epochs"]):
        X = (mu[:, None, :] + torch.einsum("cij,cnj->cni", Ls, torch.randn(T, 256, d, device=dev, generator=g))).reshape(-1, d)
        Y = torch.arange(T, device=dev).repeat_interleave(256)
        p = torch.randperm(len(Y), device=dev, generator=g)
        X, Y = X[p], Y[p]
        for i in range(T):
            x, y = X[i * 256:(i + 1) * 256], Y[i * 256:(i + 1) * 256]
            loss = F.cross_entropy(a["scale"] * F.linear(F.normalize(x, dim=1), F.normalize(W, dim=1)), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
    return W.detach()


def replay_sts(rec, mu, cov, cfg):
    """STS tren vi tri da ghi (cung ham voi online). mu: K x d float64 (sua tai cho), cov: K x d x d float32 CPU."""
    F_real, F_adv = rec["F_real"].to(dev).double(), rec["F_adv"].to(dev).double()
    adv_c, E, n_real = rec["adv_c"], F_real.size(0) - 1, F_real.size(1)
    _, bg, low = initial_estep(F_real[0], mu, cov, cfg)
    used = select_adversarial(F_adv[0], adv_c, mu, cov, bg, low, cfg).to(dev)
    pool = [torch.cat([F_real[e], F_adv[e][used]]) for e in range(E + 1)]
    split = half_split(rec["y_real"])
    r = pool_estep(pool[0], n_real, mu, cov, cfg)
    prev = 0
    for e in (range(1, E + 1) if cfg["freq"] == "epoch" else [E]):
        r, _ = update(pool[prev], pool[e], r, mu, cov, n_real, split, cfg, cfg["eta"])
        prev = e
    return {"mass": r.sum(0).cpu(), "n_adv": int(used.sum())}


def heads(r, t, means, covs, with_agm, ca_logits=None):
    """Logit CA, LDA (va AGM) tren feature test cho thong ke (means, covs) cua mot cach."""
    X = r["test_feat"].double()
    if ca_logits is None:
        ca_logits = (r["head_logits"].float() if t == 0 else
                     F.linear(F.normalize(X.float().to(dev), dim=1),
                              F.normalize(train_ca(r["pre_ca_fc"], means, covs, t, r["ca_args"]), dim=1)).cpu())
    out = {"CA": ca_logits.double(), "LDA": lda_logits(means, covs.double(), X).double()}
    if with_agm:
        T = means.size(0)
        nb = np.full(T, int(np.round(r["n_train_per_class"].numpy().mean())))
        agm = AGMHead(X.size(1), M=AGM_M, device=dev, lin_gamma=1.0, bias=10.0)
        agm.fit(means, covs.double(), nb)
        agm.solve(LAMBDA_RATIO * torch.trace(agm.G).item() / agm.M)
        out["AGM"] = agm.logits(r["test_feat"]).cpu()
        del agm
        torch.cuda.empty_cache()
    return out


def scores(lg, y, K):
    s = {k: acc(v, y) for k, v in lg.items()}
    s["zCA_LDA"] = acc(z(lg["CA"]) + z(lg["LDA"]), y)
    main = z(lg["CA"]) + z(lg["LDA"])
    if "AGM" in lg:
        main = main + z(lg["AGM"])
        s["zCA_LDA_AGM"] = acc(main, y)
    if K > 0:                                           # lop cu / lop moi cua ensemble chinh
        s["old"], s["new"] = acc(main[y < K], y[y < K]), acc(main[y >= K], y[y >= K])
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--agm", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--methods", default="start,sts,oracle,online")
    ap.add_argument("overrides", nargs="*", help="sts_khoa=gia_tri (json)")
    a = ap.parse_intermixed_args()
    args = {}
    for kv in a.overrides:
        k, v = kv.split("=", 1)
        args[k] = json.loads(v)
    cfg = sts_config(args)
    methods = a.methods.split(",")
    NT = len(glob.glob(f"{a.dump}/task_*.pt"))
    stats, rows = {}, {m: [] for m in methods}
    for t in range(NT):
        r = torch.load(f"{a.dump}/task_{t}.pt", map_location="cpu", weights_only=False)
        K, T, y = r["known_classes"], r["total_classes"], r["test_label"]
        if t == 0:
            for m in methods:
                stats[m] = (r["class_means"].double(), r["class_covs_full"].float())
        else:
            ftr, ytr = r["train_feat_all"], r["train_label_all"]
            new_m, new_c = class_stats(ftr, ytr, range(K, T))
            true_m, true_c = class_stats(ftr, ytr, range(K))
            info = {}
            for m in methods:
                if m == "online":
                    stats[m] = (r["class_means"].double(), r["class_covs_full"].float())
                    continue
                om, oc = stats[m][0][:K].clone(), stats[m][1][:K].clone()
                if m == "sts":
                    mu = om.to(dev)
                    info = replay_sts(r["sts_record"], mu, oc, cfg)
                    om = mu.cpu()
                    dev_online = ((om - r["class_means"][:K].double()).norm(dim=1) /
                                  r["class_means"][:K].double().norm(dim=1)).max().item()
                    info["lech_online"] = dev_online
                elif m == "oracle":
                    om, oc = true_m, true_c
                stats[m] = (torch.cat([om, new_m]), torch.cat([oc, new_c]))
        for m in methods:
            means, covs = stats[m]
            lg = heads(r, t, means, covs, a.agm, ca_logits=r["head_logits"].float() if m == "online" else None)
            s = scores(lg, y, K)
            if t > 0:
                s["mean_err"] = (means[:K] - true_m).norm(dim=1).mean().item()
                s["cov_err"] = ((covs[:K].double() - true_c.double()).flatten(1).norm(dim=1) /
                                true_c.double().flatten(1).norm(dim=1)).mean().item()
            rows[m].append(s)
        line = " | ".join(f"{m} " + " ".join(f"{k} {v:.2f}" for k, v in rows[m][-1].items()) for m in methods)
        extra = (f" | sts N_c tv {info['mass'].median().item():.2f}, doi khang {info['n_adv']}, "
                 f"lech so voi online {info['lech_online']:.1e}") if t > 0 and "sts" in methods else ""
        print(f"T{t} {line}{extra}", flush=True)

    print("== Avg / Last (old/new: TB tu task 1)")
    for m in methods:
        keys = [k for k in rows[m][0].keys()]
        later = rows[m][1:]
        print(f"{m:7s} " + " | ".join(f"{k} {np.mean([x[k] for x in rows[m]]):.2f}/{rows[m][-1][k]:.2f}" for k in keys) +
              ("".join(f" | {k} {np.mean([x[k] for x in later]):.3f}/{later[-1][k]:.3f}"
                       for k in ("old", "new", "mean_err", "cov_err")) if later else ""))
    if a.name:
        out = os.path.join(ROOT, "exps", "sts", "replay")
        os.makedirs(out, exist_ok=True)
        json.dump({"cfg": {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()}, "rows": rows},
                  open(os.path.join(out, f"{a.name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
