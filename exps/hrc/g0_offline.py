"""HRC - cong G0 (offline tren dump; chi de phan tich, co dung tap test).

(a) V3-specgrid voi U = 0: train lai CA tren dump V3-specgrid voi covariance (1 + 1/n) S (bo U), so voi chinh run
    (Sigma_pred = (1 + 1/n) S + U) va V2-CA; cung chuoi head, cung nhieu lay mau.
(b) Tran: quet theta = (n0, tau, s) cua ho phan tan V3
        Sigma_c(theta) = s [n0 P_c(tau) + (n_c - 1) C_c] / (n0 + n_c - 1) + 1e-3 I    (s nhan ca ridge)
    tren dump V2 (cung backbone), chon theta bang tap test.

Cach dung lai thong ke ma khong chay lai STS: STS van chuyen covariance tuyen tinh (A^T Sigma A) va voi procrustes moi
lop dung chung anh xa, nen hinh dang vay o moi task la to hop tuyen tinh co dinh cua covariance mau da van chuyen:
    S_c^(t) = sum_k M_ck C_k^(t) + rho_c I,     C_k^(t) = Sigma_k^(V2, t) - 1e-3 I
(M, rho tinh mot lan tu trong so vay luc lop duoc hoc; ridge da van chuyen xap xi 1e-3 I, sai khac khong dang ke khi
n0 > 0, va n0 = 0 cho dung Sigma^(V2)). Mean lop cu lay cua V2: bo qua anh huong cua theta len buoc E cua STS.
CA: ban sao base._stage2_compact_classifier chay song song nhieu cau hinh (cung mau chuan va hoan vi -> so sanh theo
cap); head cu khoi tao tu ket qua CA task truoc cua chinh cau hinh, head moi tu pre_ca_fc cua V2 (giong nhau giua cac
run vi backbone va head moi khong phu thuoc thong ke).

Dung: python exps/hrc/g0_offline.py [--seeds 0 1 2] [--out exps/hrc/g0.json]
"""
import argparse
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
dev = "cuda"
RIDGE = 1e-3
V2_DUMP = "exps/sts/dump/imageneta_RSIAT_raests_end"
SPEC_DUMP = "exps/v3/dump/imageneta_v3specgrid_s1993"
N_TASKS = 10
N0_GRID = (0.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
TAU_GRID = (0.0, 4.0, 16.0)
S_GRID = (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0)
KAPPAS = ("rsiat", "one")


def load(d, t):
    return torch.load(f"{d}/task_{t}.pt", map_location="cpu", weights_only=False)


def kappa_vec(mode, T, task_size, t):
    if mode == "one":
        return torch.ones(T)
    tid = torch.arange(T) // task_size
    return 0.9 + (tid + 1).float() / (t + 1) * 0.1                   # nhu base.py (decay theo task cua lop)


# ---------------------------------------------------------------------- ho phan tan
def mixing(meta, n0, tau):
    """M (T x T), rho (T): S_c = sum_k M_ck C_k + rho_c I, dung nhu models/v3stats.new_class_shapes (nguon: S cua lop
    cu + covariance mau cua lop moi khac, trong so softmax(tau cos) theo mean luc hoc, bo chinh no)."""
    Tn = meta["T"][-1]
    M = torch.zeros(Tn, Tn, dtype=torch.float64)
    rho = torch.zeros(Tn, dtype=torch.float64)
    for t in range(N_TASKS):
        K, T = meta["K"][t], meta["T"][t]
        mu = F.normalize(meta["mu_learn"][t].double(), dim=1)
        for c in range(K, T):
            nc = float(meta["n"][c])
            if n0 <= 0.0:
                M[c, c], rho[c] = 1.0, RIDGE
                continue
            logits = tau * (mu[c] @ mu[:T].t())
            logits[c] = float("-inf")
            w = torch.softmax(logits, 0)
            a, b = n0 / (n0 + nc - 1.0), (nc - 1.0) / (n0 + nc - 1.0)
            row = a * (w[:K] @ M[:K])
            row[K:T] += a * w[K:T]
            row[c] += b
            M[c], rho[c] = row, a * float(w[:K] @ rho[:K]) + RIDGE
    return M, rho


@torch.no_grad()
def shapes(M, rho, cov_v2):
    """S^(t) (T x d x d, float32, dev) tu covariance V2 da van chuyen o task t."""
    T, d = cov_v2.shape[0], cov_v2.shape[1]
    C = cov_v2.to(dev) - RIDGE * torch.eye(d, device=dev)
    S = (M[:T, :T].float().to(dev) @ C.reshape(T, -1)).reshape(T, d, d)
    del C
    S += rho[:T].float().to(dev)[:, None, None] * torch.eye(d, device=dev)
    return S


@torch.no_grad()
def chol(S):
    L, info = torch.linalg.cholesky_ex(S.double())
    assert int(info.abs().max()) == 0, "covariance khong xac dinh duong"
    return L.float()


# ---------------------------------------------------------------------- CA song song
def train_ca_batch(W0, MU, Ls, L_idx, sq, Y_all, a, gen):
    """G cau hinh doc lap (tong loss = tong loss tung cau hinh -> moi cau hinh nhu chay rieng). W0, MU: G x T x d;
    Ls: nL x T x d x d; cau hinh g lay mau MU[g] + sq[g] Ls[L_idx[g]] z voi z chung cho moi cau hinh."""
    G, T, d = W0.shape
    W = W0.clone().requires_grad_(True)
    opt = torch.optim.SGD([W], lr=a["init_lr"], momentum=0.9, weight_decay=a["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a["ca_epochs"])
    B = 256
    for _ in range(a["ca_epochs"]):
        with torch.no_grad():
            Z = torch.randn(T, B, d, device=dev, generator=gen)
            LZ = torch.einsum("lcij,cnj->lcni", Ls, Z)
            X = (MU[:, :, None, :] + sq[:, None, None, None] * LZ[L_idx]).reshape(G, T * B, d)
            del LZ, Z
            p = torch.randperm(T * B, device=dev, generator=gen)
            X = F.normalize(X[:, p], dim=-1)
            Y = Y_all[p]
        for i in range(T):
            x, y = X[:, i * B:(i + 1) * B], Y[i * B:(i + 1) * B]
            logits = a["scale"] * torch.bmm(x, F.normalize(W, dim=-1).transpose(1, 2))
            loss = F.cross_entropy(logits.reshape(G * B, T), y.repeat(G)) * G
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
        del X
    return W.detach()


@torch.no_grad()
def accs(W, Xte, yte, K):
    pred = torch.bmm(F.normalize(Xte, dim=-1)[None].expand(W.size(0), -1, -1),
                     F.normalize(W, dim=-1).transpose(1, 2)).argmax(-1)
    ok = (pred == yte[None]).float()
    old = ok[:, yte < K].mean(1) * 100 if K > 0 else torch.full((W.size(0),), float("nan"), device=dev)
    return (ok.mean(1) * 100).tolist(), old.tolist(), (ok[:, yte >= K].mean(1) * 100).tolist()


def run_chain(cfgs, get_task, seed):
    """cfgs: danh sach cau hinh; get_task(t) -> (Ls, L_idx, sq, MU_base, mu_idx). Tra ve acc moi task cho moi cau hinh."""
    G = len(cfgs)
    res = [{"acc": [], "old": [], "new": []} for _ in range(G)]
    W_prev = None
    for t in range(N_TASKS):
        r = TASK[t]
        K, T = r["K"], r["T"]
        if t == 0:                                    # task 0 khong co CA: head train truc tiep
            acc0 = (r["head_logits"].float().argmax(1) == r["test_label"]).float().mean().item() * 100
            for g in range(G):
                res[g]["acc"].append(acc0)
                res[g]["old"].append(float("nan"))
                res[g]["new"].append(acc0)
            W_prev = r["head_new"].to(dev)[None].expand(G, -1, -1).clone()
            continue
        Ls, L_idx, sq, MUb, mu_idx = get_task(t)
        kap = torch.stack([kappa_vec(c["kappa"], T, r["task_size"], t) for c in cfgs]).to(dev)
        MU = MUb[mu_idx] * kap[:, :, None]
        W0 = torch.cat([W_prev[:, :K], r["head_new"].to(dev)[None].expand(G, -1, -1)], 1)
        gen = torch.Generator(device=dev).manual_seed(1000 * seed + t)
        Y_all = torch.arange(T, device=dev).repeat_interleave(256)
        W = train_ca_batch(W0, MU, Ls, L_idx, sq, Y_all, r["ca_args"], gen)
        a, o, n = accs(W, r["test_x"], r["test_y"], K)
        for g in range(G):
            res[g]["acc"].append(a[g])
            res[g]["old"].append(o[g])
            res[g]["new"].append(n[g])
        W_prev = W
        del Ls, MU, W0
        torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------------- du lieu
def load_meta():
    """Thong tin nhe cua moi task tu dump V2 (covariance doc lai khi can)."""
    meta = {"K": [], "T": [], "mu_learn": [], "n": None}
    for t in range(N_TASKS):
        r = load(V2_DUMP, t)
        K, T = r["known_classes"], r["total_classes"]
        meta["K"].append(K)
        meta["T"].append(T)
        meta["mu_learn"].append(r["class_means"].clone())
        meta["n"] = r["n_train_per_class"].clone()
        fc = r["pre_ca_fc"] if t > 0 else None
        head_new = (fc[f"heads.{t}.0.weight"] if t > 0 else None)
        TASK.append({"K": K, "T": T, "task_size": T - K if t > 0 else T, "ca_args": r.get("ca_args"),
                     "test_x": r["test_feat"].float().to(dev), "test_y": r["test_label"].to(dev),
                     "test_label": r["test_label"], "head_logits": r["head_logits"],
                     "head_new": head_new, "means": r["class_means"].clone()})
        if t == 1:                                    # head task 0 (khong CA o task 0)
            TASK[0]["head_new"] = fc["heads.0.0.weight"].clone()
        del r
    return meta


TASK = []


def summarize(res):
    acc = res["acc"]
    return float(np.mean(acc)), acc[-1]


def part_a(seeds, out):
    """V3-specgrid: Sigma_pred (run) / (1 + 1/n) S (U = 0) / V2."""
    for t in range(N_TASKS):
        a, b = TASK[t]["test_x"], load(SPEC_DUMP, t)["test_feat"].float().to(dev)
        assert torch.equal(a, b), f"feature test khac nhau o task {t}"
    print("feature test V2 = V3-specgrid o ca 10 task", flush=True)
    cfgs = [{"name": "V2-CA", "kappa": "rsiat"}, {"name": "V3spec (U, run)", "kappa": "rsiat"},
            {"name": "V3spec U=0", "kappa": "rsiat"}]

    def get_task(t):
        r2, rs = load(V2_DUMP, t), load(SPEC_DUMP, t)
        f = (1.0 + 1.0 / rs["class_n"]).float()
        covs = [r2["class_covs_full"], rs["class_covs_full"], rs["class_S_full"] * f[:, None, None]]
        Ls = torch.stack([chol(c.to(dev)) for c in covs])
        MUb = torch.stack([r2["class_means"], rs["class_means"]]).float().to(dev)
        return Ls, torch.tensor([0, 1, 2], device=dev), torch.ones(3, device=dev), MUb, \
            torch.tensor([0, 1, 1], device=dev)

    out["a"] = {}
    for seed in seeds:
        t0 = time.time()
        res = run_chain(cfgs, get_task, seed)
        out["a"][seed] = {c["name"]: r for c, r in zip(cfgs, res)}
        print(f"[a] seed {seed} ({time.time() - t0:.0f}s): " +
              " | ".join(f"{c['name']} {summarize(r)[0]:.2f}/{summarize(r)[1]:.2f}" for c, r in zip(cfgs, res)),
              flush=True)


def check_mixing(meta):
    """Kiem tra cach dung lai: S cua V3-specgrid (n0 = 100, tau = 4) tu dump V2."""
    M, rho = mixing(meta, 100.0, 4.0)
    for t in (0, 1, 5, 9):
        S_hat = shapes(M, rho, load(V2_DUMP, t)["class_covs_full"])
        S_run = load(SPEC_DUMP, t)["class_S_full"].to(dev)
        K, T = TASK[t]["K"], TASK[t]["T"]
        rel = ((S_hat - S_run).flatten(1).norm(dim=1) / S_run.flatten(1).norm(dim=1)).cpu()
        print(f"  task {t}: sai so tuong doi |S_dung_lai - S_run| / |S_run|: lop moi max {rel[K:T].max():.2e}"
              + (f", lop cu trung vi {rel[:K].median():.2e} max {rel[:K].max():.2e}" if K else ""), flush=True)
        del S_hat, S_run


def part_b(meta, seeds, out):
    cfgs = [{"s": s, "kappa": k} for k in KAPPAS for s in ARGS.s]
    sq = torch.tensor([math.sqrt(c["s"]) for c in cfgs], device=dev)
    out["b"] = out.get("b", {})
    combos = [(0.0, 0.0)] + [(n0, tau) for n0 in ARGS.n0 if n0 > 0 for tau in ARGS.tau]
    for seed in seeds:
        out["b"].setdefault(seed, {})
        for n0, tau in combos:
            key = f"n0={n0:g},tau={tau:g}"
            if key in out["b"][seed]:
                continue
            t0 = time.time()
            M, rho = mixing(meta, n0, tau)

            def get_task(t):
                S = shapes(M, rho, load(V2_DUMP, t)["class_covs_full"])
                Ls = chol(S)[None]
                del S
                MUb = TASK[t]["means"].float().to(dev)[None]
                G = len(cfgs)
                return Ls, torch.zeros(G, dtype=torch.long, device=dev), sq, MUb, torch.zeros(G, dtype=torch.long,
                                                                                               device=dev)

            res = run_chain(cfgs, get_task, seed)
            out["b"][seed][key] = {f"s={c['s']:g},k={c['kappa']}": r for c, r in zip(cfgs, res)}
            best = max(zip(cfgs, res), key=lambda x: summarize(x[1])[0])
            print(f"[b] seed {seed} {key} ({time.time() - t0:.0f}s): tot nhat s={best[0]['s']:g} "
                  f"k={best[0]['kappa']} {summarize(best[1])[0]:.2f}/{summarize(best[1])[1]:.2f}", flush=True)
            json.dump(out, open(ARGS.out, "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="exps/hrc/g0.json")
    ap.add_argument("--parts", default="check,a,b")
    ap.add_argument("--n0", type=float, nargs="+", default=list(N0_GRID))
    ap.add_argument("--tau", type=float, nargs="+", default=list(TAU_GRID))
    ap.add_argument("--s", type=float, nargs="+", default=list(S_GRID))
    ARGS = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    meta = load_meta()
    out = {}
    try:
        out = json.load(open(ARGS.out))
        out["b"] = {int(k): v for k, v in out.get("b", {}).items()}
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    parts = ARGS.parts.split(",")
    if "check" in parts:
        check_mixing(meta)
    if "a" in parts:
        part_a(ARGS.seeds, out)
        json.dump(out, open(ARGS.out, "w"))
    if "b" in parts:
        part_b(meta, ARGS.seeds, out)
    json.dump(out, open(ARGS.out, "w"))
    print("g0 done", flush=True)
