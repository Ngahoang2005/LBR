"""Thu offline: co covariance tung lop bang OAS (Chen et al. 2010) truoc LDA/AGM, roi tinh lai z CA+LDA+AGM.

OAS khong co sieu tham so: he so co rho tinh tu chinh covariance va so mau train cua lop.
Dung: python exps/v2/oas_eval.py <ten>=<thu_muc_dump> ...
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "exps/agm")
from final_eval import AGM_M, LAMBDA_RATIO, acc, lda_logits, z  # noqa: E402
from models.agm import AGMHead  # noqa: E402


def oas(S, n):
    """Sigma = (1 - rho) S + rho * tr(S)/p * I, rho theo cong thuc OAS."""
    p = S.size(-1)
    tr, tr2 = torch.diagonal(S).sum(), (S * S).sum()
    rho = ((1 - 2 / p) * tr2 + tr ** 2) / ((n + 1 - 2 / p) * (tr2 - tr ** 2 / p))
    rho = float(rho.clamp(0, 1))
    return (1 - rho) * S + rho * tr / p * torch.eye(p, dtype=S.dtype), rho


def evaluate(dump_dir):
    rows = []
    for t in range(len(glob.glob(f"{dump_dir}/task_*.pt"))):
        r = torch.load(f"{dump_dir}/task_{t}.pt", map_location="cpu", weights_only=False)
        y = r["test_label"]
        means, covs = r["class_means"].double(), r["class_covs_full"].double()
        T = means.size(0)
        n = float(np.round(r["n_train_per_class"].numpy().mean()))
        shr = [oas(covs[c], n) for c in range(T)]
        covs_o = torch.stack([s for s, _ in shr])
        ca = r["head_logits"].float()
        row = dict(task=t, rho=np.mean([p for _, p in shr]), CA=acc(ca, y))
        for tag, C in (("base", covs), ("oas", covs_o)):
            lda = lda_logits(means, C, r["test_feat"]).float()
            agm = AGMHead(768, M=AGM_M, device="cuda", lin_gamma=1.0, bias=10.0)
            agm.fit(means, C, np.full(T, int(n)))
            agm.solve(LAMBDA_RATIO * torch.trace(agm.G).item() / agm.M)
            la = agm.logits(r["test_feat"]).float().cpu()
            del agm
            torch.cuda.empty_cache()
            row[f"LDA_{tag}"] = acc(lda, y)
            row[f"AGM_{tag}"] = acc(la, y)
            row[f"ens_{tag}"] = acc(z(ca.double()) + z(lda.double()) + z(la.double()), y)
        rows.append(row)
        print(name, t, " ".join(f"{k}={v:.2f}" for k, v in row.items() if k != "task"), flush=True)
    return rows


for arg in sys.argv[1:]:
    name, d = arg.split("=", 1)
    rows = evaluate(d)
    keys = [k for k in rows[0] if k not in ("task", "rho")]
    print(f"== {name}  rho TB {np.mean([r['rho'] for r in rows]):.3f}")
    print("Avg  " + " ".join(f"{k}={np.mean([r[k] for r in rows]):.2f}" for k in keys))
    print("Last " + " ".join(f"{k}={rows[-1][k]:.2f}" for k in keys))
