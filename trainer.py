import copy
import logging
import os
import sys

import random

import numpy as np
import torch

from data.data_manager import DataManager
from utils import model_factory
from utils.toolkit import count_parameters


def RSIAT_train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    sum_seed = 0.0
    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        sum_seed += _train(args)
    avg_seed = sum_seed / len(seed_list)
    print('Average Seed Accuracy (CNN):', avg_seed)
    logging.info("Average Seed Accuracy (CNN): {}".format(avg_seed))


def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"], args["dataset"], init_cls, args['increment'])
    os.makedirs(logs_name, exist_ok=True)

    logfilename = "{}/{}_{}_{}".format(logs_name, args["prefix"], args["seed"], args["convnet_type"])
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = model_factory.get_model(args["model_name"], args)

    cnn_curve = {"top1": [], "top5": []}
    resume_path = logfilename + ".resume.pt"
    start = _load_resume(resume_path, model, cnn_curve)
    for task in range(start, data_manager.nb_tasks):
        if task >= int(args.get("max_tasks", 10 ** 9)):  # chay sang loc vai task dau
            break
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info("Trainable params: {}".format(count_parameters(model._network, True)))

        model.incremental_train(data_manager)
        cnn_accy = model.eval_task()
        if task == data_manager.nb_tasks - 1:    # du doan tren tap test o task cuoi (McNemar / bootstrap)
            y_pred, y_true = model._eval_cnn(model.test_loader)
            np.savez(logfilename + "_pred.npz", y_pred=y_pred[:, 0], y_true=y_true)
        if args.get("ckpt_dir") and task in args.get("ckpt_tasks", [0]):    # cho mo phong offline (exps/lbr/sim.py)
            os.makedirs(args["ckpt_dir"], exist_ok=True)
            torch.save({"net": model._network.state_dict(), "class_order": list(data_manager._class_order)},
                       os.path.join(args["ckpt_dir"], f"task_{task}.pt"))
        model.after_task()

        logging.info("CNN: {}".format(cnn_accy["grouped"]))
        cnn_curve["top1"].append(cnn_accy["top1"])
        cnn_curve["top5"].append(cnn_accy["top5"])

        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))

        avg = sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
        print('Average Accuracy (CNN):', avg)
        logging.info("Average Accuracy (CNN): {}".format(avg))
        _save_resume(resume_path, model, cnn_curve, task)
    if os.path.exists(resume_path):
        os.remove(resume_path)
    return sum(cnn_curve["top1"]) / len(cnn_curve["top1"])


RESUME_ATTRS = ["_network", "_old_network", "old_ae", "_class_means", "_xp_means", "_xp_covs", "_cur_task",
                "_known_classes", "_total_classes", "_xp_center", "_sdc_prev"]


def _save_resume(path, model, cnn_curve, task):
    """Checkpoint sau moi task (trang thai learner + RNG): run bi dung thi chay lai tu task tiep theo."""
    state = {"task": task, "curve": cnn_curve, "learner": {k: getattr(model, k, None) for k in RESUME_ATTRS},
             "rng": (torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state(), random.getstate())}
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def _load_resume(path, model, cnn_curve):
    if not os.path.exists(path):
        return 0
    state = torch.load(path, map_location="cpu", weights_only=False)
    for k, v in state["learner"].items():
        setattr(model, k, v)
    dev = model._device
    for k in ("_network", "_old_network", "old_ae"):
        if getattr(model, k, None) is not None:
            getattr(model, k).to(dev)
    model.old_network_module_ptr = model._old_network
    cnn_curve.update(state["curve"])
    torch.set_rng_state(state["rng"][0])
    torch.cuda.set_rng_state_all(state["rng"][1])
    np.random.set_state(state["rng"][2])
    random.setstate(state["rng"][3])
    logging.info("Resume tu %s: da xong task %d, curve %s", path, state["task"], cnn_curve["top1"])
    return state["task"] + 1


def _set_device(args):
    args["device"] = [torch.device("cuda:{}".format(d)) for d in args["device"]]


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
