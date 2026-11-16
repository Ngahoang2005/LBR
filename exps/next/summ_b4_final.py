"""B4 factorial: quy dao x bo nho (B3, acc TB task 1-9), Avg/Last, ti so head/trunk (mean lop, TB moi buoc)."""
import glob
import json
import re

import numpy as np

D, O = "logs/adapter/imageneta/0/20/", "../RSIAT_orig/logs/adapter/imageneta/0/20/"
ROWS = [("RSIAT+", "rsiatd"), ("chi L_trunk", "T2"), ("chi L_MD", "T5"), ("LBR", "T4"), ("LBR+L_orth push", "T4p")]


def tags(r, al, s):
    """(tag B3/B1, glob log train)."""
    if r == "rsiatd":
        t = f"b6_rsiatd_a{al}" + ("" if s == 1993 else f"_s{s}")
        if al == 64 and s != 1993:
            return t, O + f"rsiat_s2shrink_s{s}_{s}_*adapter.log"
        return t, O + f"{t}_{s}_*adapter.log"
    t = f"b4_{r}_a{al}_s{s}"
    if r == "T4" and al == 64:
        return t, D + f"lbr_split_ae_md1_s{s}_{s}_*adapter.log"
    if r == "T4" and al == 256 and s == 1993:
        return t, D + f"b6_lbr_a256_s1993_*adapter.log"
    return t, D + f"{t}_{s}_*adapter.log"


def avg_last(pat):
    L = glob.glob(pat)
    if not L:
        return None
    c = [l for l in open(L[0]) if "CNN top1 curve" in l]
    v = [float(x) for x in re.findall(r"[\d.]+", c[-1].split("curve:")[1])] if c else []
    return (np.mean(v), v[-1]) if len(v) == 10 else None


def b3(t):
    try:
        d = json.load(open(f"exps/claims/b3_{t}.json"))
    except FileNotFoundError:
        return None
    return [np.mean([d[k][m]["acc"] for k in d]) for m in ("f_sdc", "z_sdc", "z_oracle")]


def ratio(t):
    try:
        return float(re.search(r"step: .*?mean-A ([\d.]+)", open(f"exps/claims/b1_{t}.log").read()).group(1))
    except (FileNotFoundError, AttributeError):
        return None


f2 = lambda x: "    -" if x is None else f"{x:5.2f}"
for al in (64, 256):
    print(f"\n=== A_L {al}")
    print(f"{'quy dao':16s} seed |   f_sdc  z_sdc  oracle   z-f | Avg    Last  | h/t")
    agg = {}
    for name, r in ROWS:
        for s in (1993, 1994, 1995):
            t, pat = tags(r, al, s)
            al_ = avg_last(pat)
            m = b3(t)
            h = ratio(t)
            if al_ is None and m is None:
                continue
            agg.setdefault(name, []).append((m, al_, h))
            mm = m or [None] * 3
            print(f"{name:16s} {s} |   {f2(mm[0])}  {f2(mm[1])}  {f2(mm[2])}  "
                  f"{'    -' if m is None else f'{m[1]-m[0]:+.2f}'} | {f2(al_ and al_[0])}  {f2(al_ and al_[1])} | {f2(h)}")
    print("-- trung binh cac seed co so")
    for name, rows in agg.items():
        M = [x[0] for x in rows if x[0]]
        A = [x[1] for x in rows if x[1]]
        H = [x[2] for x in rows if x[2]]
        mm = np.mean(M, 0) if M else [None] * 3
        print(f"{name:16s} n={len(M)}/{len(A)} | {f2(mm[0])}  {f2(mm[1])}  {f2(mm[2])}  "
              f"{'    -' if not M else f'{mm[1]-mm[0]:+.2f}'} | {f2(np.mean([a[0] for a in A]))}  "
              f"{f2(np.mean([a[1] for a in A]))} | {f2(np.mean(H) if H else None)}")
