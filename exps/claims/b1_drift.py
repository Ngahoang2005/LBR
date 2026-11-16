"""B1 (go/no-go): drift cua head co dang ke so voi drift cua trunk khong? (Prop. 3, chi de phan tich)

f_t = h_t(z_t). Voi anh train that (transform test) cua lop cu, moi buoc task t-1 -> t:
  thu tu A (Prop. 3): D_head = h_t(z_t) - h_{t-1}(z_t),     D_trunk = h_{t-1}(z_t) - h_{t-1}(z_{t-1})
  thu tu B:           D_head = h_t(z_{t-1}) - h_{t-1}(z_{t-1}), D_trunk = h_t(z_t) - h_t(z_{t-1})
Bao cao: RMS |D| / RMS |f_{t-1}| (theo mau) va tb theo lop |mean_c D| / RMS |f_{t-1}| (dich mean lop, thu SDC theo doi),
ti so head/trunk; them drift tich luy tu model task 0 toi task t tren lop task 0.
Can checkpoint {"net": state_dict} sau moi task: <ckpt_dir>/task_<t>.pt (chi dung phan convnet).

Chay: python exps/claims/b1_drift.py --config <json> --ckpt_dir <dir> --tag <ten>
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from data.data_manager import DataManager  # noqa: E402
from models.lbr import head_from_xprime, xprime  # noqa: E402
from utils.inc_net import get_convnet  # noqa: E402


def load_vit(args, path):
    state = torch.load(path, map_location="cpu", weights_only=False)["net"]
    vit = get_convnet(args)
    vit.load_state_dict({k[len("convnet."):]: v for k, v in state.items() if k.startswith("convnet.")})
    return vit.cuda().eval()


@torch.no_grad()
def decompose(v0, v1, dm, n_cls):
    """Tong binh phuong theo mau va tong vector theo lop cua cac thanh phan drift tren anh train cua n_cls lop dau."""
    ds = dm.get_dataset(np.arange(0, n_cls), source="train", mode="test")
    keys = ["A_head", "A_trunk", "B_head", "B_trunk", "total"]
    sq = {k: 0.0 for k in keys + ["f"]}
    cls = {k: torch.zeros(n_cls, v0.out_dim, dtype=torch.float64, device="cuda") for k in keys}
    cnt = torch.zeros(n_cls, dtype=torch.float64, device="cuda")
    cross = 0.0
    for _, x, y in DataLoader(ds, batch_size=64, shuffle=False, num_workers=4):
        x, y = x.cuda(), y.cuda()
        z0, z1 = xprime(v0, x), xprime(v1, x)
        f00, f01 = head_from_xprime(v0, z0).double(), head_from_xprime(v0, z1).double()
        f11, f10 = head_from_xprime(v1, z1).double(), head_from_xprime(v1, z0).double()
        D = {"A_head": f11 - f01, "A_trunk": f01 - f00, "B_head": f10 - f00, "B_trunk": f11 - f10, "total": f11 - f00}
        for k, v in D.items():
            sq[k] += float(v.pow(2).sum())
            cls[k].index_add_(0, y, v)
        sq["f"] += float(f00.pow(2).sum())
        cross += float(2 * (D["A_head"] * D["A_trunk"]).sum())
        cnt += torch.bincount(y, minlength=n_cls).double()
    N = float(cnt.sum())
    rf = (sq["f"] / N) ** 0.5
    r = {"n_img": int(N), "rms_f": rf, "cross_A_rel": cross / N / rf ** 2}
    for k in keys:
        r[f"rms_{k}"] = (sq[k] / N) ** 0.5 / rf
        r[f"mean_{k}"] = float((cls[k] / cnt[:, None]).norm(dim=1).mean()) / rf
    for o in "AB":
        r[f"ratio_rms_{o}"] = r[f"rms_{o}_head"] / r[f"rms_{o}_trunk"]
        r[f"ratio_mean_{o}"] = r[f"mean_{o}_head"] / r[f"mean_{o}_trunk"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--tag", required=True)
    o = ap.parse_args()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"b1_{o.tag}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s => %(message)s",
                        handlers=[logging.FileHandler(out + ".log", mode="w"), logging.StreamHandler(sys.stdout)])
    args = json.load(open(o.config))
    args["seed"] = args["seed"][0] if isinstance(args["seed"], list) else args["seed"]
    dm = DataManager(args["dataset"], args["shuffle"], args["seed"], args["init_cls"], args["increment"])
    T = dm.nb_tasks
    ck = lambda t: os.path.join(o.ckpt_dir, f"task_{t}.pt")
    assert all(os.path.exists(ck(t)) for t in range(T)), f"thieu checkpoint trong {o.ckpt_dir}"
    order = torch.load(ck(0), map_location="cpu", weights_only=False).get("class_order")
    assert order is None or list(order) == list(dm._class_order), "thu tu lop khac checkpoint"

    res = {"step": {}, "cum": {}}
    v_first = load_vit(args, ck(0))
    v_prev = v_first
    n0 = dm.get_task_size(0)
    hdr = "%-6s %6s | %8s %8s %8s %6s | %8s %8s %6s | %8s %8s %6s"
    row = "%-6s %6d | %8.4f %8.4f %8.4f %6.2f | %8.4f %8.4f %6.2f | %8.4f %8.4f %6.2f"
    logging.info("B1 %s: drift tuong doi (chia RMS |f|), lop cu = anh train that", o.tag)
    logging.info(hdr, "buoc", "n_anh", "rms_tot", "rmsA_hd", "rmsA_tr", "A h/t", "mnA_hd", "mnA_tr", "mA h/t",
                 "rmsB_hd", "rmsB_tr", "B h/t")
    for t in range(1, T):
        v_cur = load_vit(args, ck(t))
        K = sum(dm.get_task_size(s) for s in range(t))
        for kind, v0, ncls in (("step", v_prev, K), ("cum", v_first, n0)):
            if kind == "cum" and t == 1:
                res["cum"][t] = res["step"][t]
                continue
            r = decompose(v0, v_cur, dm, ncls)
            res[kind][t] = r
            logging.info(row, f"{'' if kind == 'step' else '0->'}{t}", r["n_img"], r["rms_total"], r["rms_A_head"],
                         r["rms_A_trunk"], r["ratio_rms_A"], r["mean_A_head"], r["mean_A_trunk"], r["ratio_mean_A"],
                         r["rms_B_head"], r["rms_B_trunk"], r["ratio_rms_B"])
        if v_prev is not v_first:
            del v_prev
        v_prev = v_cur
        torch.cuda.empty_cache()
    for kind in ("step", "cum"):
        rs = list(res[kind].values())
        logging.info("%s: ti so head/trunk tb  RMS-A %.3f  mean-A %.3f  RMS-B %.3f  mean-B %.3f; cheo-A tb %.4f", kind,
                     np.mean([r["ratio_rms_A"] for r in rs]), np.mean([r["ratio_mean_A"] for r in rs]),
                     np.mean([r["ratio_rms_B"] for r in rs]), np.mean([r["ratio_mean_B"] for r in rs]),
                     np.mean([r["cross_A_rel"] for r in rs]))
    json.dump(res, open(out + ".json", "w"), indent=2)


if __name__ == "__main__":
    main()
