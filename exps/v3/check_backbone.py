"""Kiem tra backbone V3 trung V2 (V3 chi doi thong ke va head): feature test giong het tung bit o moi task.
Dung: python exps/v3/check_backbone.py <dump_v2> <dump_v3>"""
import glob
import sys

import torch

a_dir, b_dir = sys.argv[1], sys.argv[2]
T = len(glob.glob(f"{b_dir}/task_*.pt"))
same = []
for t in range(T):
    a = torch.load(f"{a_dir}/task_{t}.pt", map_location="cpu", weights_only=False)
    b = torch.load(f"{b_dir}/task_{t}.pt", map_location="cpu", weights_only=False)
    same.append(torch.equal(a["test_feat"], b["test_feat"]) and torch.equal(a["test_label"], b["test_label"]))
print(f"backbone V3 = V2 (feature test giong het): {sum(same)}/{T} task", "" if all(same) else f"-> khac o task {[t for t, s in enumerate(same) if not s]}")
