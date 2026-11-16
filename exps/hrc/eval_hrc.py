"""Danh gia run HRC so voi V2 (cung seed): CA tu log, backbone (feature test giong V2), theta* moi task va ensemble
z(CA) + z(LDA) + z(AGM) tren thong ke HRC (covariance = s S nhu replay cua CA, hoac S).

Dung: python exps/hrc/eval_hrc.py [dump_hrc] [log_hrc]
"""
import glob
import re
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "exps/agm")
from final_eval import AGM_M, LAMBDA_RATIO, acc, lda_logits, z  # noqa: E402
from models.agm import AGMHead  # noqa: E402

D_HRC = sys.argv[1] if len(sys.argv) > 1 else "exps/hrc/dump/imageneta_hrc_s1993"
L_HRC = sys.argv[2] if len(sys.argv) > 2 else "logs/adapter/imageneta/0/20/hrc_s1993_1993_pretrained_vit_b16_224_in21k_adapter.log"
D_V2 = "exps/sts/dump/imageneta_RSIAT_raests_end"
L_V2 = "logs/adapter/imageneta/0/20/RSIAT_raests_end_1993_pretrained_vit_b16_224_in21k_adapter.log"
V2_ENS = [87.43, 84.44, 80.88, 78.51, 75.62, 74.87, 71.63, 68.99, 67.24, 66.69]     # final_eval tren dump V2


def curve(log):
    lines = [l for l in open(log) if "CNN top1 curve" in l]
    return [float(x) for x in re.findall(r"[\d.]+", lines[-1].split("curve:")[1])]


def heads(r, covs):
    y, T = r["test_label"], r["total_classes"]
    means = r["class_means"].double()
    nb = np.full(T, int(np.round(r["n_train_per_class"].numpy().mean())))
    ca = r["head_logits"].float()
    lda = lda_logits(means, covs, r["test_feat"]).float()
    agm = AGMHead(768, M=AGM_M, device="cuda", lin_gamma=1.0, bias=10.0)
    agm.fit(means, covs, nb)
    agm.solve(LAMBDA_RATIO * torch.trace(agm.G).item() / agm.M)
    la = agm.logits(r["test_feat"]).float().cpu()
    del agm
    torch.cuda.empty_cache()
    return {"CA": acc(ca, y), "LDA": acc(lda, y), "AGM": acc(la, y),
            "ENS": acc(z(ca.double()) + z(lda.double()) + z(la.double()), y)}


if __name__ == "__main__":
    ca_h, ca_v = curve(L_HRC), curve(L_V2)
    n = len(ca_h)
    print("CA (log)      " + " ".join(f"{x:6.2f}" for x in ca_h) + f" | Avg {np.mean(ca_h):.2f} Last {ca_h[-1]:.2f}")
    print("V2-CA (log)   " + " ".join(f"{x:6.2f}" for x in ca_v[:n]) + f" | Avg {np.mean(ca_v[:n]):.2f} Last {ca_v[n - 1]:.2f}")
    print("HRC - V2      " + " ".join(f"{a - b:+6.2f}" for a, b in zip(ca_h, ca_v)) +
          f" | Avg {np.mean(ca_h) - np.mean(ca_v[:n]):+.2f} Last {ca_h[-1] - ca_v[n - 1]:+.2f}")
    same, rows = 0, {"sS": [], "S": []}
    for t in range(len(glob.glob(f"{D_HRC}/task_*.pt"))):
        r = torch.load(f"{D_HRC}/task_{t}.pt", map_location="cpu", weights_only=False)
        r2 = torch.load(f"{D_V2}/task_{t}.pt", map_location="cpu", weights_only=False)
        same += int(torch.equal(r["test_feat"], r2["test_feat"]))
        sel = r.get("hrc_select")
        if sel is not None:
            print(f"task {t}: theta* (n0, tau, s) = {sel['theta']}, log-loss O {sel['best'][0]:.4f} (V2 {sel['v2'][0]:.4f}),"
                  f" sau temperature {sel['best'][-1]:.4f} (V2 {sel['v2'][-1]:.4f}),"
                  f" loi O {sel['best'][1]:.2f}% (V2 {sel['v2'][1]:.2f}%), bien: "
                  + (",".join(k for k, v in sel["edge"].items() if v) or "khong"))
        S = r["class_covs_full"].double()
        rows["sS"].append(heads(r, S * r["class_s"].double()[:, None, None]))
        rows["S"].append(heads(r, S))
    print(f"backbone HRC = V2 (feature test giong het): {same}/{len(rows['S'])} task")
    for k, R in rows.items():
        print(f"\n== ensemble, covariance {'s S (nhu replay)' if k == 'sS' else 'S'}")
        for key in ("CA", "LDA", "AGM", "ENS"):
            v = [x[key] for x in R]
            print(f"{key:4s} " + " ".join(f"{x:6.2f}" for x in v) + f" | Avg {np.mean(v):.2f} Last {v[-1]:.2f}")
    v = [x["ENS"] for x in rows["sS"]]
    m = len(v)
    print(f"V2 ens " + " ".join(f"{x:6.2f}" for x in V2_ENS[:m]) + f" | Avg {np.mean(V2_ENS[:m]):.2f} Last {V2_ENS[m - 1]:.2f}")
