"""Phan tich gia thuyet (tren cac buoc mo phong da luu cua SimCal nhung): lech cua SimCal ve phia replay rong (s lon)
den tu viec head lop cu gia trong mo phong chua qua CA, trong khi o pipeline that CA o task t khoi tao tu head DA QUA CA.

So sanh hai cach cham theta tren cung cac buoc mo phong:
  mot buoc : CA mot lan tren moi lop (nhu SimCal hien tai), khoi tao tu head train cua A1 / A2.
  hai buoc : CA mo phong o buoc S_cu truoc (lop cu that + S_cu), roi CA o buoc S_moi khoi tao tu head do (dung chuoi
             head cua pipeline that). Thong ke dung cho CA buoc 1 lay o khong gian A2 (xap xi, vi chi luu thong ke sau
             van chuyen).
Dung: python exps/hrc/simcal_chain.py [simcal_sims.pt]
"""
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from models.hrc import _chol, train_ca_batch  # noqa: E402
from models.simcal import _scores, sim_shapes  # noqa: E402

dev = "cuda"
path = sys.argv[1] if len(sys.argv) > 1 else "exps/hrc/dump/imageneta_simcal3_s1993/simcal_sims.pt"
D = torch.load(path, map_location="cpu", weights_only=False)
sims, ca, p_old = D["sims"], D["ca"], D["p_old"]
COMBOS = [(0.0, 0.0), (20.0, 0.0), (20.0, 256.0), (30.0, 1024.0), (100.0, 0.0), (100.0, 256.0)]
CONFS = [(s, k) for k in ("rsiat", "one") for s in (1.0, 2.0, 3.0, 5.0)]
G = len(CONFS)
sq = torch.tensor([math.sqrt(s) for s, _ in CONFS], device=dev)
torch.backends.cuda.matmul.allow_tf32 = False


def kappa(age, t_cur, mode):
    return (0.9 + (age + 1) / (t_cur + 1) * 0.1) if mode == "rsiat" else torch.ones_like(age)


def run(two_stage, seeds=(0, 1)):
    table = {}
    for n0, tau in COMBOS:
        lg_all, ys, ws, olds = [], [], [], []
        for sim in sims:
            K, n_old, n_new = sim["K"], sim["n_old"], len(sim["n_n"])
            Ls = _chol(sim_shapes(sim, n0, tau, dev), dev)
            mu = torch.cat([sim["mu_r"], sim["mu_o"], sim["mu_n"]]).double()
            tsim = len(set(sim["task_r"]))
            age = torch.tensor(list(sim["task_r"]) + [tsim] * n_old + [tsim + 1] * n_new, dtype=torch.float64)
            y = sim["yO"].to(dev)
            old = y < K + n_old
            w = torch.where(old, p_old / old.sum(), (1 - p_old) / (~old).sum()).float() / len(sims)
            X = F.normalize(sim["fO"].to(dev), dim=1)
            W0 = sim["W0"].to(dev)
            for k in seeds:
                gen = torch.Generator(device=dev).manual_seed(777 + k)
                W_init = W0[None].expand(G, -1, -1).contiguous()
                if two_stage:                           # CA o buoc S_cu: lop cu that + S_cu, t_cur = tsim
                    m = K + n_old
                    MU1 = torch.stack([(mu[:m] * kappa(age[:m], tsim, kk)[:, None]).float() for _, kk in CONFS]).to(dev)
                    W1 = train_ca_batch(W_init[:, :m].contiguous(), MU1, Ls[None, :m].contiguous(),
                                        torch.zeros(G, dtype=torch.long, device=dev), sq,
                                        torch.arange(m, device=dev).repeat_interleave(256), ca, gen)
                    W_init = torch.cat([W1, W_init[:, m:]], 1)
                MU = torch.stack([(mu * kappa(age, tsim + 1, kk)[:, None]).float() for _, kk in CONFS]).to(dev)
                W = train_ca_batch(W_init, MU, Ls[None], torch.zeros(G, dtype=torch.long, device=dev), sq,
                                   torch.arange(mu.size(0), device=dev).repeat_interleave(256), ca, gen)
                lg_all.append(ca["scale"] * torch.einsum("nd,gcd->gnc", X, F.normalize(W, dim=-1)))
                ys.append(y)
                ws.append(w / len(seeds))
                olds.append(old)
            del Ls
        lg, y, w, old = torch.cat(lg_all, 1), torch.cat(ys), torch.cat(ws), torch.cat(olds)
        ref, err, nll = _scores(lg, y, w)
        for g, (s, kk) in enumerate(CONFS):
            table[(n0, tau, s, kk)] = (float(ref[g]), 100 * float(err[g]))
    return table


for name, two in (("mot buoc (nhu SimCal)", False), ("hai buoc (chuoi head that)", True)):
    tab = run(two)
    print(f"== {name}")
    for i, crit in enumerate(("refinement", "loi")):
        order = sorted(tab, key=lambda th: tab[th][i])
        print(f"  theo {crit}: 5 theta tot nhat " + "; ".join(f"{th}" for th in order[:5]))
        for ref_th in ((20.0, 256.0, 2.0, "rsiat"), (0.0, 0.0, 1.0, "rsiat"), (100.0, 0.0, 5.0, "one")):
            print(f"     hang cua {ref_th}: {order.index(ref_th) + 1}/{len(order)} ({tab[ref_th][i]:.4f})")
    print("  loi theo s (n0 20, tau 256, kappa rsiat):",
          " ".join(f"s={s:g}: {tab[(20.0, 256.0, s, 'rsiat')][1]:.2f}" for s in (1.0, 2.0, 3.0, 5.0)), flush=True)
