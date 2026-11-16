"""Cache x'_CLS (dau vao nhanh MLP / adapter cua khoi cuoi, models/lbr.xprime) cua moi anh qua trunk sau task 0, cho
mo phong offline LBR-FT (exps/lbr/sim.py). Trunk (khoi 1..L-1 + attention khoi cuoi) dong bang sau task 0 nen x' cua
mot anh (cung phep bien doi) khong doi qua cac task.

Ghi (fp16): train_test (anh train, transform test, eval: thong ke lop), test (eval), va V lan bien doi train ngau nhien:
views_train (mode train: dropout cua adapter trunk bat, nhu forward khi train) va views_eval (cung anh bien doi, eval:
dau vao mang cu cua RAE).
Dung: python exps/lbr/sim_cache.py <config.json> <ckpt task_0.pt> <out.pt> [--views 15]
"""
import argparse
import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from data.data_manager import DataManager  # noqa: E402
from models.lbr import xprime  # noqa: E402
from utils.inc_net import SimpleVitNet  # noqa: E402


@torch.no_grad()
def run(net, loader, modes, dev):
    out, ys = {m: [] for m in modes}, []
    for _, x, y in loader:
        x = x.to(dev, non_blocking=True)
        for m in modes:
            net.train(m == "train")
            out[m].append(xprime(net.convnet, x).half().cpu())
        ys.append(y.long())
    return {m: torch.cat(v) for m, v in out.items()}, torch.cat(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--views", type=int, default=15)
    a = ap.parse_args()
    args = json.load(open(a.config))
    dev = "cuda"
    torch.manual_seed(0)
    dm = DataManager(args["dataset"], args["shuffle"], args["seed"][0], args["init_cls"], args["increment"])
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    assert list(dm._class_order) == list(ck["class_order"]), "thu tu lop khac checkpoint"
    net = SimpleVitNet(args, True)
    net.update_fc(args["init_cls"])
    net.load_state_dict(ck["net"])
    net.to(dev)
    T = dm.get_total_classnum()
    kw = dict(batch_size=128, shuffle=False, num_workers=8)
    rec = {"class_order": list(dm._class_order), "increments": [dm.get_task_size(t) for t in range(dm.nb_tasks)]}
    xs, rec["train_label"] = run(net, DataLoader(dm.get_dataset(np.arange(T), source="train", mode="test"), **kw),
                                 ("eval",), dev)
    rec["train_test"] = xs["eval"]
    xs, rec["test_label"] = run(net, DataLoader(dm.get_dataset(np.arange(T), source="test", mode="test"), **kw),
                                ("eval",), dev)
    rec["test"] = xs["eval"]
    vt, ve = [], []
    for v in range(a.views):
        xs, y = run(net, DataLoader(dm.get_dataset(np.arange(T), source="train", mode="train"), **kw),
                    ("train", "eval"), dev)
        assert torch.equal(y, rec["train_label"])
        vt.append(xs["train"])
        ve.append(xs["eval"])
        print(f"view {v + 1}/{a.views}", flush=True)
    rec["views_train"], rec["views_eval"] = torch.stack(vt), torch.stack(ve)
    torch.save(rec, a.out)
    print("ghi", a.out, {k: tuple(v.shape) for k, v in rec.items() if torch.is_tensor(v)})


if __name__ == "__main__":
    main()
