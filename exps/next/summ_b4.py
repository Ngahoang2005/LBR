"""Tong hop B4/B6: Avg/Last cua run day du + B3 (acc TB tasks 1-9 voi bo nho f, z, oracle z) + B1 (head/trunk)."""
import glob, json, os, re, sys
import numpy as np
L = "logs/adapter/imageneta/0/20/"
def run(tag):
    f = glob.glob(L + tag + "_*_pretrained_vit_b16_224_in21k_adapter.log")
    if not f: return None
    s = open(f[0]).read()
    if "Average Seed" not in s: return None
    curve = eval(re.findall(r"CNN top1 curve: (\[.*\])", s)[-1])
    return float(re.findall(r"Average Accuracy \(CNN\): ([\d.]+)", s)[-1]), curve[-1]
def b3(tag):
    p = f"exps/claims/b3_{tag}.json"
    if not os.path.exists(p): return None
    r = json.load(open(p))
    return {m: np.mean([v[m]["acc"] for v in r.values()]) for m in ("f_sdc", "z_sdc", "z_oracle", "real")}
def b1(tag):
    p = f"exps/claims/b1_{tag}.json"
    if not os.path.exists(p): return None
    return np.mean([v["ratio_rms_A"] for v in json.load(open(p))["step"].values()])
rows = []
for tag in sys.argv[1:]:
    a, b, c = run(tag), b3(tag), b1(tag)
    if a is None and b is None: continue
    rows.append(tag)
    print(f"{tag:22s} run {a[0] if a else float('nan'):6.2f}/{a[1] if a else float('nan'):6.2f} | " +
          (f"B3 f {b['f_sdc']:6.2f} z {b['z_sdc']:6.2f} or {b['z_oracle']:6.2f} real {b['real']:6.2f} z-f {b['z_sdc']-b['f_sdc']:+.2f}" if b else "B3 -") +
          (f" | h/t {c:.2f}" if c is not None else ""))
