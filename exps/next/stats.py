"""McNemar (chinh xac, nhi thuc) va paired bootstrap tren du doan tap test o task cuoi, theo tung seed va gop 3 seed."""
import numpy as np
from scipy.stats import binomtest
D, O = "logs/adapter/imageneta/0/20/", "../RSIAT_orig/logs/adapter/imageneta/0/20/"
P = {"RSIAT": {1993: D + "claims_rsiat_1993_pretrained_vit_b16_224_in21k_adapter.log_pred.npz",
               1994: O + "rsiat_s1994_1994_pretrained_vit_b16_224_in21k_adapter_pred.npz",
               1995: O + "rsiat_s1995_1995_pretrained_vit_b16_224_in21k_adapter_pred.npz"},
     "RSIAT+s2+shrink": {s: O + f"rsiat_s2shrink_s{s}_{s}_pretrained_vit_b16_224_in21k_adapter_pred.npz" for s in (1993, 1994, 1995)},
     "LBR": {1993: D + "lbr_split_ae_md1_s1993_1993_pretrained_vit_b16_224_in21k_adapter.log_pred.npz",
             **{s: D + f"lbr_split_ae_md1_s{s}_{s}_pretrained_vit_b16_224_in21k_adapter_pred.npz" for s in (1994, 1995)}}}
C = {m: {s: np.load(p) for s, p in d.items()} for m, d in P.items()}
for s in (1993, 1994, 1995):
    ys = [C[m][s]["y_true"] for m in C]
    assert all((y == ys[0]).all() for y in ys), s
corr = {m: {s: C[m][s]["y_pred"] == C[m][s]["y_true"] for s in C[m]} for m in C}
print("Last acc:", {m: [round(100 * corr[m][s].mean(), 2) for s in (1993, 1994, 1995)] for m in corr})
rng = np.random.default_rng(0)
for a, b in (("RSIAT+s2+shrink", "RSIAT"), ("LBR", "RSIAT"), ("LBR", "RSIAT+s2+shrink")):
    print(f"\n{a} vs {b}")
    tot = [0, 0]
    for s in (1993, 1994, 1995):
        x, y = corr[a][s], corr[b][s]
        n10, n01 = int((x & ~y).sum()), int((~x & y).sum()); tot[0] += n10; tot[1] += n01
        print(f"  seed {s}: diff {100 * (x.mean() - y.mean()):+.2f}, chi {a} dung {n10}, chi {b} dung {n01}, McNemar p = {binomtest(n10, n10 + n01).pvalue:.3f}")
    n = len(corr[a][1993])
    diffs = []
    for _ in range(10000):
        diffs.append(np.mean([(corr[a][s][i] .mean() - corr[b][s][i].mean()) for s in (1993, 1994, 1995)
                              for i in [rng.integers(0, len(corr[a][s]), len(corr[a][s]))]]))
    diffs = 100 * np.array(diffs)
    print(f"  gop 3 seed: McNemar {tot[0]} vs {tot[1]}, p = {binomtest(tot[0], sum(tot)).pvalue:.3f}; bootstrap diff tb "
          f"{diffs.mean():+.2f}, KTC95% [{np.percentile(diffs, 2.5):+.2f}, {np.percentile(diffs, 97.5):+.2f}], P(diff<=0) = {(diffs <= 0).mean():.3f}")
