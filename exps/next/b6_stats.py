"""B6 (A_L 64/128/256), LBR vs RSIAT+, seed 1993 va 1994: Avg/Last mean +- std, McNemar va paired bootstrap tren Last.

(a) LBR - RSIAT+ co gian toi ~+1 khi A_L lon: bootstrap hieu cua hieu (A_L 256/128 so voi 64).
(b) RSIAT+ kem di khi A_L lon: McNemar/bootstrap RSIAT+ A_L 64 vs 128/256.
Chay tu thu muc RSIAT: python exps/next/b6_stats.py
"""
import re

import numpy as np
from scipy.stats import binomtest

D, O = "logs/adapter/imageneta/0/20/", "../RSIAT_orig/logs/adapter/imageneta/0/20/"
S = (1993, 1994)
SUF = "pretrained_vit_b16_224_in21k_adapter"
# A_L 64 = cau hinh mac dinh (lbr_split_ae_md1, rsiat_s2shrink); LBR A_L 256 seed 1994 = b4_T4_a256_s1994 (cung cau hinh)
RUN = {
    ("LBR", 64): {1993: D + f"lbr_split_ae_md1_s1993_1993_{SUF}.log", 1994: D + f"lbr_split_ae_md1_s1994_1994_{SUF}"},
    ("LBR", 128): {s: D + f"b6_lbr_a128_s{s}_{s}_{SUF}" for s in S},
    ("LBR", 256): {1993: D + f"b6_lbr_a256_s1993_1993_{SUF}", 1994: D + f"b4_T4_a256_s1994_1994_{SUF}"},
    ("RSIAT+", 64): {s: O + f"rsiat_s2shrink_s{s}_{s}_{SUF}" for s in S},
    ("RSIAT+", 128): {1993: O + f"b6_rsiatd_a128_1993_{SUF}", 1994: O + f"b6_rsiatd_a128_s1994_1994_{SUF}"},
    ("RSIAT+", 256): {1993: O + f"b6_rsiatd_a256_1993_{SUF}", 1994: O + f"b6_rsiatd_a256_s1994_1994_{SUF}"},
}
B = 10000
rng = np.random.default_rng(0)


def curve(stem):
    log = stem if stem.endswith(".log") else stem + ".log"
    line = [l for l in open(log) if "CNN top1 curve" in l][-1]
    return [float(x) for x in re.findall(r"[\d.]+", line.split("curve:")[1])]


def correct(stem):
    z = np.load(stem + "_pred.npz")
    return z["y_pred"] == z["y_true"], z["y_true"]


R = {k: {s: {"curve": curve(p), "corr": correct(p)} for s, p in d.items()} for k, d in RUN.items()}
for s in S:
    ys = [R[k][s]["corr"][1] for k in R]
    assert all((y == ys[0]).all() for y in ys), s
C = {k: {s: R[k][s]["corr"][0] for s in S} for k in R}

print("Avg / Last theo seed va mean +- std (std mau, 2 seed)")
for k in RUN:
    av = [np.mean(R[k][s]["curve"]) for s in S]
    la = [R[k][s]["curve"][-1] for s in S]
    print(f"  {k[0]:7s} A_L {k[1]:3d}: Avg {av[0]:.2f}, {av[1]:.2f} -> {np.mean(av):.2f} +- {np.std(av, ddof=1):.2f} | "
          f"Last {la[0]:.2f}, {la[1]:.2f} -> {np.mean(la):.2f} +- {np.std(la, ddof=1):.2f}")


def boot(diff_fn):
    """paired bootstrap trong tung seed (resample anh test), gop trung binh 2 seed; tra ve (diem, CI95, p hai phia)."""
    n = len(C[("LBR", 64)][1993])
    pt = np.mean([diff_fn(s, np.arange(n)) for s in S])
    bs = np.array([np.mean([diff_fn(s, rng.integers(0, n, n)) for s in S]) for _ in range(B)])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return pt, np.percentile(bs, [2.5, 97.5]), min(p, 1.0)


def mcnemar(a, b):
    n10 = sum(int((C[a][s] & ~C[b][s]).sum()) for s in S)
    n01 = sum(int((~C[a][s] & C[b][s]).sum()) for s in S)
    return n10, n01, binomtest(n10, n10 + n01).pvalue


def report(title, a, b):
    pt, ci, p = boot(lambda s, i: 100 * (C[a][s][i].mean() - C[b][s][i].mean()))
    n10, n01, pm = mcnemar(a, b)
    print(f"  {title}: Last {pt:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}], bootstrap p = {p:.3f}; "
          f"McNemar gop {n10}/{n01}, p = {pm:.3f}")


print("\nLBR vs RSIAT+ (Last, gop 2 seed)")
for al in (64, 128, 256):
    report(f"A_L {al:3d}", ("LBR", al), ("RSIAT+", al))

print("\n(a) khoang cach LBR - RSIAT+ o A_L lon tru khoang cach o A_L 64 (hieu cua hieu)")
gap = lambda al: lambda s, i: 100 * (C[("LBR", al)][s][i].mean() - C[("RSIAT+", al)][s][i].mean())
for al in (128, 256):
    pt, ci, p = boot(lambda s, i: gap(al)(s, i) - gap(64)(s, i))
    print(f"  A_L {al} - A_L 64: {pt:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}], p = {p:.3f}")

print("\n(b) RSIAT+ A_L lon so voi A_L 64")
for al in (128, 256):
    report(f"RSIAT+ {al} - 64", ("RSIAT+", al), ("RSIAT+", 64))
print("   doi chung LBR:")
for al in (128, 256):
    report(f"LBR {al} - 64", ("LBR", al), ("LBR", 64))
