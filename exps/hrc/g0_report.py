"""Tong hop cong G0(b) tu exps/hrc/g0.json (exps/hrc/g0_offline.py). Moi so: CA, ImageNet-A seed 1993, trung binh
cac seed nhieu CA (so theo cap voi V2-CA = n0 0, s 1, kappa RSIAT cung seed).
Dung: python exps/hrc/g0_report.py [exps/hrc/g0_b.json]
"""
import json
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "exps/hrc/g0_b.json"
B = json.load(open(path))["b"]
seeds = sorted(B)
V2 = ("n0=0,tau=0", "s=1,k=rsiat")


def arr(combo, cfg, key="acc"):
    return np.array([B[s][combo][cfg][key] for s in seeds])          # seeds x tasks


base = arr(*V2)
rows = []
for combo in B[seeds[0]]:
    for cfg in B[seeds[0]][combo]:
        a = arr(combo, cfg)
        d = a - base
        rows.append({"combo": combo, "cfg": cfg, "avg": a.mean(), "last": a[:, -1].mean(), "davg": d.mean(),
                     "dlast": d[:, -1].mean(), "davg_seed": d.mean(1), "dtask": d.mean(0), "acc": a.mean(0)})

print(f"V2-CA (offline, {len(seeds)} seed nhieu CA): Avg {base.mean():.2f}, Last {base[:, -1].mean():.2f}; "
      f"Avg moi seed " + " ".join(f"{x:.2f}" for x in base.mean(1)))
for kap in ("rsiat", "one"):
    R = sorted([r for r in rows if r["cfg"].endswith(kap)], key=lambda r: -r["davg"])
    print(f"\n== kappa = {kap}: 10 theta tot nhat theo Avg trung binh seed (chon bang test)")
    print("theta                         Avg    Last   dAvg   dLast  dAvg moi seed")
    for r in R[:10]:
        print(f"{r['combo'] + ',' + r['cfg'].split(',')[0]:28s} {r['avg']:.2f}  {r['last']:.2f}  {r['davg']:+.2f}  "
              f"{r['dlast']:+.2f}  " + " ".join(f"{x:+.2f}" for x in r["davg_seed"]))
    best = R[0]
    print("tran (kappa %s): %s %s  dAvg %+.2f  dLast %+.2f" % (kap, best["combo"], best["cfg"], best["davg"],
                                                                  best["dlast"]))
    print("  moi task (dAcc):", " ".join(f"{x:+.2f}" for x in best["dtask"]))
    # tran lac quan: chon rieng tung seed
    per_seed = [max(r["davg_seed"][i] for r in rows if r["cfg"].endswith(kap)) for i in range(len(seeds))]
    print("  tran chon rieng moi seed (lac quan hon):", " ".join(f"{x:+.2f}" for x in per_seed))

# bang Avg theo (n0, s) voi tau tot nhat, kappa RSIAT
print("\n== dAvg so V2 (kappa RSIAT), tau tot nhat cho moi (n0, s):")
s_vals = sorted({float(r["cfg"].split(",")[0][2:]) for r in rows})
n0_vals = sorted({float(r["combo"].split(",")[0][3:]) for r in rows})
print("n0 \\ s   " + " ".join(f"{s:>6g}" for s in s_vals))
for n0 in n0_vals:
    cells = []
    for s in s_vals:
        c = [r["davg"] for r in rows if r["cfg"] == f"s={s:g},k=rsiat" and float(r["combo"].split(",")[0][3:]) == n0]
        cells.append(f"{max(c):+6.2f}" if c else "     -")
    print(f"{n0:<8g} " + " ".join(cells))
