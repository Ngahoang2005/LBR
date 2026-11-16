"""Du doan offline CA cua mot day theta theo task (moi lop dung theta cua task no duoc hoc), cung cach dung lai thong ke
cua exps/hrc/g0_offline.py (dump V2, cung backbone). Dung de doc nhanh ket qua cua mot day theta* (vd. cua run HRC).

Dung: python exps/hrc/g0_pertask.py '<ten>=n0,tau,s;n0,tau,s;...' ...   (1 bo: dung cho moi task; it hon 10: bo cuoi
lap lai cho cac task sau). Mac dinh them V2 (0,0,1). In Avg / Last va chenh lech moi task so voi V2, trung binh seed.
"""
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "exps/hrc")
sys.path.insert(0, ".")
import g0_offline as g  # noqa: E402


def mixing_pt(meta, th):
    Tn = meta["T"][-1]
    M = torch.zeros(Tn, Tn, dtype=torch.float64)
    rho = torch.zeros(Tn, dtype=torch.float64)
    for t in range(g.N_TASKS):
        n0, tau = th[t][0], th[t][1]
        K, T = meta["K"][t], meta["T"][t]
        mu = F.normalize(meta["mu_learn"][t].double(), dim=1)
        for c in range(K, T):
            nc = float(meta["n"][c])
            if n0 <= 0.0:
                M[c, c], rho[c] = 1.0, g.RIDGE
                continue
            logits = tau * (mu[c] @ mu[:T].t())
            logits[c] = float("-inf")
            w = torch.softmax(logits, 0)
            a, b = n0 / (n0 + nc - 1.0), (nc - 1.0) / (n0 + nc - 1.0)
            row = a * (w[:K] @ M[:K])
            row[K:T] += a * w[K:T]
            row[c] += b
            M[c], rho[c] = row, a * float(w[:K] @ rho[:K]) + g.RIDGE
    return M, rho


def parse(spec):
    th = [tuple(float(v) for v in part.split(",")) for part in spec.split(";")]
    return th + [th[-1]] * (g.N_TASKS - len(th))


def main(cands, seeds=(0, 1, 2)):
    meta = g.load_meta()
    names = ["V2"] + [c.split("=", 1)[0] for c in cands]
    ths = [parse("0,0,1")] + [parse(c.split("=", 1)[1]) for c in cands]
    mix = [mixing_pt(meta, th) for th in ths]
    cfgs = [{"kappa": "rsiat"} for _ in ths]

    def get_task(t):
        r = g.load(g.V2_DUMP, t)
        T = meta["T"][t]
        Ls = []
        for (M, rho), th in zip(mix, ths):
            s_cls = torch.tensor([th[c // 20 if t > 0 else 0][2] for c in range(T)], device=g.dev)
            S = g.shapes(M, rho, r["class_covs_full"])
            Ls.append(g.chol(S) * s_cls.sqrt()[:, None, None])
            del S
        G = len(ths)
        return (torch.stack(Ls), torch.arange(G, device=g.dev), torch.ones(G, device=g.dev),
                r["class_means"].float().to(g.dev)[None], torch.zeros(G, dtype=torch.long, device=g.dev))

    acc = np.array([[r["acc"] for r in g.run_chain(cfgs, get_task, s)] for s in seeds])   # seeds x cand x task
    m = acc.mean(0)
    for i, n in enumerate(names):
        d = (acc[:, i] - acc[:, 0]).mean(0)
        print(f"{n:14s} Avg {m[i].mean():.2f} Last {m[i][-1]:.2f} | dAvg {d.mean():+.2f} dLast {d[-1]:+.2f} | "
              + " ".join(f"{x:+.2f}" for x in d) + "  | theta " + ";".join(",".join(f"{v:g}" for v in x) for x in ths[i][:3])
              + "...", flush=True)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    main(sys.argv[1:])
