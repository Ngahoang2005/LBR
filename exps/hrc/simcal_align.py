"""Phan tich (dung ket qua test da co, chi de danh gia tieu chi): tieu chi SimCal co xep hang theta giong do chinh xac
that khong. Do chinh xac that (CA, seed 1993) cua moi theta lay tu sweep offline exps/hrc/g0_b.json (dump V2, cung
backbone, 3 seed nhieu CA). Tieu chi SimCal lay tu bang trong dump task 1 cua run SimCal.

Dung: python exps/hrc/simcal_align.py <dump_dir_simcal>
"""
import json
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

d = sys.argv[1] if len(sys.argv) > 1 else "exps/hrc/dump/imageneta_simcal2_s1993"
sel = torch.load(f"{d}/task_1.pt", map_location="cpu", weights_only=False)["hrc_select"]
tab = {tuple(k.split(",")): v for k, v in sel["table"].items()}
B = json.load(open("exps/hrc/g0_b.json"))["b"]
seeds = sorted(B)
base = np.array([B[s]["n0=0,tau=0"]["s=1,k=rsiat"]["acc"] for s in seeds])
rows = []
for (n0, tau, s, kap), v in tab.items():
    combo = f"n0={float(n0):g},tau={float(tau):g}" if float(n0) > 0 else "n0=0,tau=0"
    cfg = f"s={float(s):g},k={kap}"
    if combo in B[seeds[0]] and cfg in B[seeds[0]][combo]:
        a = np.array([B[x][combo][cfg]["acc"] for x in seeds])
        rows.append((float(n0), float(tau), float(s), kap, v, (a - base).mean(), (a - base)[:, -1].mean()))
print(f"{len(rows)} theta chung giua bang SimCal va sweep that")
names = ["refinement (wref)", "loi co trong so (werr)", "log-loss co trong so (wll)", "loi nhom cu", "loi nhom moi"]
for i, n in enumerate(names):
    crit = np.array([r[4][i] for r in rows])
    ra, rl = spearmanr(-crit, [r[5] for r in rows])[0], spearmanr(-crit, [r[6] for r in rows])[0]
    best = rows[int(np.argmin(crit))]
    print(f"{n:28s} Spearman voi dAvg that {ra:+.2f}, voi dLast {rl:+.2f} | theta tot nhat theo tieu chi "
          f"{best[:4]}: dAvg that {best[5]:+.2f}, dLast {best[6]:+.2f}")
top = sorted(rows, key=lambda r: -r[5])[:5]
print("5 theta tot nhat theo test (tham chieu):", [(r[:4], round(r[5], 2)) for r in top])
