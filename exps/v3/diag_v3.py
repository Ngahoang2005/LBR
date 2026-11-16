"""Chan doan V3 so voi V2 tren cung backbone (feature test giong het), offline tren dump. Kiem 4 nguyen nhan:
  N2  do chinh xac lop cu / lop moi cua tung head (CA, LDA, AGM, ensemble): V3 co keo anh mo ho ve lop cu khong.
  N3  tr(U)/tr(S) va ty le sau khi lam trang theo S (tr(S^-1 U)/d) theo tuoi lop; sai so mean that sau STS so voi U
      (mean that: feature train cua lop cu qua model hien tai, lay tu dump V2 co diag_replay - cung backbone).
  N4  khoi luong trach nhiem cua lop cu va phan nen o buoc E dau task (sts_mass), V2 so voi V3.
Dung: python exps/v3/diag_v3.py <dump_v2 (co train_feat_all)> <ten>=<dump_v3> ...
"""
import glob
import sys

import torch

sys.path.insert(0, ".")
from exps.agm.final_eval import AGM_M, LAMBDA_RATIO, lda_logits, z  # noqa: E402
from models.agm import AGMHead  # noqa: E402


def load(d, t):
    return torch.load(f"{d}/task_{t}.pt", map_location="cpu", weights_only=False)


def heads(r):
    """Du doan cua CA, LDA, AGM, ensemble (giong exps/agm/final_eval.py)."""
    means, covs = r["class_means"].double(), r["class_covs_full"].double()
    T = r["total_classes"]
    nb = torch.full((T,), float(round(r["n_train_per_class"].double().mean().item())))
    ca = r["head_logits"].double()
    lda = lda_logits(means, covs, r["test_feat"], r.get("lda_weight")).double()
    agm = AGMHead(768, M=AGM_M, device="cuda", lin_gamma=1.0, bias=10.0)
    agm.fit(means, covs, nb)
    agm.solve(LAMBDA_RATIO * torch.trace(agm.G).item() / agm.M)
    la = agm.logits(r["test_feat"]).double().cpu()
    del agm
    torch.cuda.empty_cache()
    return {"CA": ca, "LDA": lda, "AGM": la, "ENS": z(ca) + z(lda) + z(la)}


def split_acc(logits, y, K):
    p = logits.argmax(1)
    old, new = y < K, y >= K
    acc = lambda m: 100.0 * (p[m] == y[m]).double().mean().item() if m.any() else float("nan")
    to_old = 100.0 * (p[new] < K).double().mean().item() if new.any() else float("nan")     # lop moi doan thanh lop cu
    return acc(old), acc(new), to_old


v2_dir = sys.argv[1]
runs = dict(a.split("=", 1) for a in sys.argv[2:])
T_n = len(glob.glob(f"{v2_dir}/task_*.pt"))
print("== N2: do chinh xac lop cu / lop moi (ensemble va tung head), task 1..%d" % (T_n - 1))
rows = {}
for name, d in [("V2", v2_dir)] + list(runs.items()):
    rows[name] = []
    for t in range(1, T_n):
        r = load(d, t)
        K, y = r["known_classes"], r["test_label"]
        h = heads(r)
        rows[name].append({k: split_acc(v, y, K) for k, v in h.items()})
for k in ["ENS", "CA", "LDA", "AGM"]:
    print(f"-- {k}: cu / moi / % anh lop moi doan nham thanh lop cu (TB task 1..{T_n - 1}; task cuoi)")
    for name in rows:
        R = rows[name]
        avg = [sum(x[k][i] for x in R) / len(R) for i in range(3)]
        last = R[-1][k]
        print(f"   {name:<10} TB {avg[0]:6.2f} / {avg[1]:6.2f} / {avg[2]:5.2f}   | cuoi {last[0]:6.2f} / {last[1]:6.2f} / "
              f"{last[2]:5.2f}")

print("\n== N3: U so voi S theo tuoi lop (task cuoi) va sai so mean that sau STS")
r2_last = load(v2_dir, T_n - 1)
assert "train_feat_all" in r2_last, "dump V2 can diag_replay (train_feat_all)"
for name, d in runs.items():
    r = load(d, T_n - 1)
    if "class_U_full" not in r:
        continue
    K, task_size = r["known_classes"], r["total_classes"] // T_n
    S, U = r["class_S_full"].double(), r["class_U_full"].double()
    trS, trU = S.diagonal(dim1=1, dim2=2).sum(1), U.diagonal(dim1=1, dim2=2).sum(1)
    white, top = [], []
    for c in range(K):
        L = torch.linalg.cholesky(S[c].cuda())
        W = torch.linalg.solve_triangular(L, torch.linalg.solve_triangular(L, U[c].cuda(), upper=False).t(), upper=False)
        ev = torch.linalg.eigvalsh(0.5 * (W + W.t()))
        white.append(ev.mean().item())
        top.append(ev.max().item())
    white, top = torch.tensor(white), torch.tensor(top)
    print(f"-- {name}: theo task hoc lop (tuoi giam dan)")
    print("   task hoc | tr U/tr S | tr(S^-1 U)/d | tri rieng max S^-1/2 U S^-1/2")
    for a in range(0, K // task_size):
        m = slice(a * task_size, (a + 1) * task_size)
        print(f"   {a:>8} | {(trU[m] / trS[m]).mean():9.4f} | {white[m].mean():12.3f} | {top[m].mean():10.3f}")
    # sai so mean that sau STS (lop cu o moi task) so voi tr U
    err, pred = [], []
    for t in range(1, T_n):
        r2, r3 = load(v2_dir, t), load(d, t)
        Kt = r3["known_classes"]
        ftr, ytr = r2["train_feat_all"].double(), r2["train_label_all"]
        mu_true = torch.stack([ftr[ytr == c].mean(0) for c in range(Kt)])
        e = r3["class_means"][:Kt].double() - mu_true
        err.append((e * e).sum(1).mean().item())
        pred.append(r3["class_trU"][:Kt].mean().item())
    print("   task      : " + " ".join(f"{t:7d}" for t in range(1, T_n)))
    print("   ||e||^2 TB: " + " ".join(f"{v:7.2f}" for v in err) + "   (sai so mean that sau STS)")
    print("   tr U TB   : " + " ".join(f"{v:7.2f}" for v in pred) + "   (U du doan: ca phan lop rong ra lan sai so mean)")

print("\n== N4: buoc E dau task: khoi luong trach nhiem TB moi lop cu (sts_mass) va so lop < 5")
for name, d in [("V2", v2_dir)] + list(runs.items()):
    vals = []
    for t in range(1, T_n):
        m = load(d, t)["sts_mass"].double()
        vals.append(f"{m.mean():.2f}/{int((m < 5).sum())}")
    print(f"   {name:<10} " + " ".join(f"{v:>9}" for v in vals))
