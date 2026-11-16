"""Du doan tren tap test (moi lop) tu checkpoint task cuoi -> <out>.npz (y_pred, y_true)."""
import json, sys
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, ".")
from data.data_manager import DataManager
from utils.inc_net import SimpleVitNet
cfg, ckpt, out = sys.argv[1:4]
args = json.load(open(cfg)); args["seed"] = args["seed"][0]
dm = DataManager(args["dataset"], args["shuffle"], args["seed"], args["init_cls"], args["increment"])
net = SimpleVitNet(args, True)
for t in range(dm.nb_tasks):
    net.update_fc(dm.get_task_size(t))
net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["net"])
net = net.cuda().eval()
ds = dm.get_dataset(np.arange(0, dm.get_total_classnum()), source="test", mode="test")
P, Y = [], []
with torch.no_grad():
    for _, x, y in DataLoader(ds, batch_size=64, num_workers=4):
        P.append(net(x.cuda())["logits"].argmax(1).cpu()); Y.append(y)
P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy()
np.savez(out, y_pred=P, y_true=Y); print(out, "acc", 100 * (P == Y).mean())
