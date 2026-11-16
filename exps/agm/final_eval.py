"""Ket qua cuoi cung: z(CA) + z(LDA) + z(AGM), AGM voi lambda = 0.1 * tr(G) / M. Offline tren dump, khong train.

Dung (sau khi da chay main.py cho cac config simple, dump duoc ghi ra exps/*/dump):
    python exps/agm/final_eval.py ina_1993=exps/simple/dump/imageneta_RSIAT_simple vtab_1993=exps/multi/dump/vtab_simple_s1993
Moi tham so la <ten>=<thu_muc_dump>. Ket qua in ra va luu vao exps/agm/results/<ten>.json.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from models.agm import AGMHead  # noqa: E402

LAMBDA_RATIO = 0.1
AGM_M = 4000
z = lambda x: (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp_min(1e-12)
acc = lambda s, y: 100 * (s.argmax(1) == y).float().mean().item()


def lda_logits(means, covs, X, weights=None):
    """LDA: covariance chung (trung binh deu cac covariance lop; V3: trung binh co trong so weights = n0 + n - 1,
    gop theo do tin cay nhu ESL 4.3) + ridge, logit tuyen tinh tu mean lop."""
    if weights is None:
        S = covs.mean(0)
    else:
        w = weights.to(covs.dtype)
        S = torch.einsum("c,cij->ij", w / w.sum(), covs)
    P = torch.linalg.inv(S + S.diagonal().mean() * torch.eye(S.size(0), dtype=S.dtype))
    A = means @ P
    return X.double() @ A.t() - 0.5 * (A * means).sum(1)


def evaluate(dump_dir):
    rows = []
    for t in range(len(glob.glob(f"{dump_dir}/task_*.pt"))):
        r = torch.load(f"{dump_dir}/task_{t}.pt", map_location="cpu", weights_only=False)
        K, T, y = r["known_classes"], r["total_classes"], r["test_label"]
        means, covs = r["class_means"].double(), r["class_covs_full"].double()
        nb = np.full(T, int(np.round(r["n_train_per_class"].numpy().mean())))

        ca = r["head_logits"].float()
        lda = lda_logits(means, covs, r["test_feat"], r.get("lda_weight")).float()
        agm = AGMHead(768, M=AGM_M, device="cuda", lin_gamma=1.0, bias=10.0)
        agm.fit(means, covs, nb)
        agm.solve(LAMBDA_RATIO * torch.trace(agm.G).item() / agm.M)
        la = agm.logits(r["test_feat"]).float().cpu()
        del agm
        torch.cuda.empty_cache()

        rows.append(dict(
            task=t, K=K, T=T,
            CA=acc(ca, y), LDA=acc(lda, y), AGM=acc(la, y),
            zCA_AGM=acc(z(ca.double()) + z(la.double()), y),
            zCA_LDA_AGM=acc(z(ca.double()) + z(lda.double()) + z(la.double()), y),
        ))
    return rows


def report(name, rows):
    keys = ["CA", "LDA", "AGM", "zCA_AGM", "zCA_LDA_AGM"]
    print(f"== {name}")
    print("| task | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for r in rows:
        print(f"| {r['task']} | " + " | ".join(f"{r[k]:.2f}" for k in keys) + " |")
    print("| Avg | " + " | ".join(f"{np.mean([r[k] for r in rows]):.2f}" for k in keys) + " |")
    print("| Last | " + " | ".join(f"{rows[-1][k]:.2f}" for k in keys) + " |")


if __name__ == "__main__":
    os.makedirs("exps/agm/results", exist_ok=True)
    for arg in sys.argv[1:]:
        name, dump_dir = arg.split("=", 1)
        rows = evaluate(dump_dir)
        report(name, rows)
        with open(f"exps/agm/results/{name}.json", "w") as f:
            json.dump(rows, f, indent=1)
