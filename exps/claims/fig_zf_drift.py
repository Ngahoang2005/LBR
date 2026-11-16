"""z - f (B3) theo do lon drift (B1), moi quy dao co ca B3 va B1; them cot "z khong SDC".

Drift cua quy dao: |Delta_head| + |Delta_trunk| tuyet doi (theo mean lop, chia RMS |f|, tach A), cong don task 0 -> t.
- Panel trai: moi quy dao 1 diem = (TB_t drift 0->t, TB_t (z_sdc - f_sdc)).
- Panel phai: moi (quy dao, task t) 1 diem.
In bang: f_sdc, z_sdc, z khong SDC (z_frozen), f khong SDC (f_frozen), oracle; TB task 1-9.
Chay tu thu muc RSIAT: python exps/claims/fig_zf_drift.py -> exps/claims/fig_zf_drift.{pdf,png}
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
GROUPS = [  # (nhom, regex tren tag, mau, marker)
        ("SSIAT-like", r"^ssiat$", "k", "s"),
    ("RSIAT+", r"^b6_rsiatd_a\d+", "C0", "o"),
    ("RSIAT+ gamma_L / LR", r"^b6_rsiatd_(gl|lrx)", "C9", "o"),
    ("RSIAT+ MLP_L", r"^b6_rsiatd_mlp", "C0", "*"),
    ("chi L_trunk", r"^b4_T2_a", "C2", "^"),
    ("chi L_MD", r"^b4_T5_", "C1", "v"),
    ("LBR", r"^(b4_T4_|b6_lbr_a\d+)", "C3", "D"),
    ("LBR + L_orth", r"^b4_T4p_", "C5", "D"),
    ("LBR gamma_L / LR", r"^b6_lbr_(gl|lrx)", "C6", "D"),
    ("LBR MLP_L", r"^b6_lbr_mlp", "C3", "*"),
    ("B4 khac (T1/T3, +-L_orth f)", r"^b4_T[13]|^b4_T[24]o_", "C8", "x"),
]
SKIP = {"rsiat", "lbr", "b6_lbr_a256"}   # trung quy dao backbone voi b6_rsiatd_a64, b4_T4_a64_s1993, b4_T4_a256_s1993


def group(tag):
    for g in GROUPS:
        if re.search(g[1], tag):
            return g
    return None


def load():
    out = []
    for p in sorted(glob.glob(os.path.join(D, "b3_*.json"))):
        tag = os.path.basename(p)[3:].replace(".v1.json", "").replace(".json", "")
        if p.endswith(".json") and not p.endswith(".v1.json") and os.path.exists(p.replace(".json", ".v1.json")):
            continue
        g, b1 = group(tag), os.path.join(D, f"b1_{tag}.json")
        if tag in SKIP or g is None or not os.path.exists(b1):
            continue
        r3, cum = json.load(open(p)), json.load(open(b1))["cum"]
        ts = sorted(int(t) for t in r3)
        acc = {m: np.array([r3[str(t)][m]["acc"] for t in ts]) for m in
               ("f_sdc", "z_sdc", "z_frozen", "f_frozen", "z_oracle", "f_oracle")}
        drift = np.array([cum[str(t)]["mean_A_head"] + cum[str(t)]["mean_A_trunk"] for t in ts])
        head = np.array([cum[str(t)]["mean_A_head"] for t in ts])
        out.append(dict(tag=tag, g=g, acc=acc, drift=drift, head=head))
    return out


R = load()
print(f"{'quy dao':28s} {'nhom':20s} {'drift':>6s} | {'f_sdc':>6s} {'z_sdc':>6s} {'z_noSDC':>7s} {'f_noSDC':>7s} "
      f"{'oracle':>6s} | {'z-f':>6s} {'zno-f':>6s} {'SDC@z':>6s}")
for r in sorted(R, key=lambda r: r["drift"].mean()):
    a = {m: v.mean() for m, v in r["acc"].items()}
    print(f"{r['tag']:28s} {r['g'][0]:20s} {r['drift'].mean():6.3f} | {a['f_sdc']:6.2f} {a['z_sdc']:6.2f} "
          f"{a['z_frozen']:7.2f} {a['f_frozen']:7.2f} {a['z_oracle']:6.2f} | {a['z_sdc'] - a['f_sdc']:+6.2f} "
          f"{a['z_frozen'] - a['f_sdc']:+6.2f} {a['z_sdc'] - a['z_frozen']:+6.2f}")

x = np.array([r["drift"].mean() for r in R])
y = np.array([(r["acc"]["z_sdc"] - r["acc"]["f_sdc"]).mean() for r in R])
xs = np.concatenate([r["drift"] for r in R])
ys = np.concatenate([r["acc"]["z_sdc"] - r["acc"]["f_sdc"] for r in R])
rho, p = spearmanr(x, y)
nm = np.array(["mlp" not in r["tag"] for r in R])
rho_n, p_n = spearmanr(x[nm], y[nm])     # bo MLP_L de xem xu huong o vung binh thuong
rho2, p2 = spearmanr(xs, ys)
print(f"\nSpearman (moi quy dao 1 diem, n={len(x)}): rho {rho:.2f}, p {p:.3g}; bo MLP_L (n={nm.sum()}): "
      f"rho {rho_n:.2f}, p {p_n:.3g}")
print(f"Spearman (quy dao x task, n={len(xs)}): rho {rho2:.2f}, p {p2:.3g}")
print(f"z - f < -0.3: {(y < -0.3).sum()}/{len(y)} quy dao; theo task: {(ys < -0.5).sum()}/{len(ys)} diem < -0.5")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
seen = set()
for r, xi, yi in zip(R, x, y):
    g = r["g"]
    ax[0].scatter(xi, yi, c=g[2], marker=g[3], s=70 if g[3] == "*" else 40, edgecolor="k", lw=0.4,
                  label=None if g[0] in seen else g[0])
    ax[1].scatter(r["drift"], r["acc"]["z_sdc"] - r["acc"]["f_sdc"], c=g[2], marker=g[3], s=14, alpha=0.7)
    seen.add(g[0])
for a, t in zip(ax, (f"moi quy dao (TB task 1-9); Spearman rho {rho:.2f} (p {p:.2g})",
                     f"moi (quy dao, task); Spearman rho {rho2:.2f} (p {p2:.2g})")):
    a.axhline(0, color="k", lw=0.8)
    a.set_xscale("log")
    a.set_xlabel("drift |D_head| + |D_trunk| (mean lop, / RMS |f|, cong don tu task 0)")
    a.set_title(t, fontsize=10)
    a.grid(alpha=0.3)
ax[0].set_ylabel("z - f: acc CA (bo nho z + SDC) - (bo nho f + SDC)")
ax[0].legend(fontsize=7, ncol=2)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(D, f"fig_zf_drift.{ext}"), dpi=150)
