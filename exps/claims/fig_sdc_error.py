"""Figure sai so SDC so voi oracle mean (tu b3_<tag>.json), 3 quy dao LBR / RSIAT / SSIAT.

Hang 1: sai so theo doi mean tho trong khong gian cua no, |mu_sdc - mu_that| / RMS: tai f, tai z, khong bu.
Hang 2: sai so mean tren mat cau sau push-forward (do o f, anh train da fit bo nho): f_sdc, z_sdc, z_oracle, f_oracle.
Chay: python exps/claims/fig_sdc_error.py -> exps/claims/fig_sdc_error.{pdf,png}
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
RUNS = [("lbr", "LBR"), ("rsiat", "RSIAT"), ("ssiat", "SSIAT")]
TRACK = [("f_rel", "f + SDC", "C0", "-"), ("z_rel", "z + SDC", "C3", "-"),
         ("f_frozen_rel", "f, khong bu", "C0", ":"), ("z_frozen_rel", "z, khong bu", "C3", ":")]
PUSH = [("f_sdc", "f + SDC", "C0", "-"), ("z_sdc", "z + SDC (push-forward)", "C3", "-"),
        ("f_oracle", "f oracle", "C0", "--"), ("z_oracle", "z oracle (push-forward)", "C3", "--")]

fig, ax = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True)
for j, (tag, name) in enumerate(RUNS):
    r = json.load(open(os.path.join(D, f"b3_{tag}.json")))
    ts = sorted(int(t) for t in r)
    for k, lab, c, ls in TRACK:
        ax[0, j].plot(ts, [r[str(t)]["track"][k] for t in ts], color=c, ls=ls, marker="o", ms=3, label=lab)
    for k, lab, c, ls in PUSH:
        ax[1, j].plot(ts, [r[str(t)][k]["mean_err_tr"] for t in ts], color=c, ls=ls, marker="o", ms=3, label=lab)
    ax[0, j].set_title(f"Quy dao {name}")
    ax[1, j].set_xlabel("task")
ax[0, 0].set_ylabel("|mu_sdc - mu_that| / RMS\n(khong gian rieng)")
ax[1, 0].set_ylabel("|mean gia - mean that| o f\n(mat cau, sau push-forward)")
ax[0, 0].legend(fontsize=8)
ax[1, 0].legend(fontsize=8)
for a in ax.flat:
    a.grid(alpha=0.3)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(D, f"fig_sdc_error.{ext}"), dpi=150)
