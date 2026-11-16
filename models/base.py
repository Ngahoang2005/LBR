import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.toolkit import accuracy, tensor2numpy

batch_size = 64


class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self.topk = 5
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def _ca_ghost_rng(self, T, ca_epochs, d):
        """Tieu thu RNG chung dung nhu CA goc cua RSIAT (moi epoch 256 mau chuan CUDA moi lop, mot randperm, mot lan tao
        iterator): CA chay voi RNG rieng nen goi ham nay de quy dao training cac task sau giu nguyen nhu RSIAT."""
        dev = self._device
        for _ in range(ca_epochs):
            for _c in range(T):
                torch.empty((256, d), dtype=torch.float32, device=dev).normal_()
            torch.randperm(T * 256)
            torch.empty((), dtype=torch.int64).random_()

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        grouped = {k: float(v) for k, v in grouped.items()}
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = float(np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        ))
        return ret

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        return self._evaluate(y_pred, y_true)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]

            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(self._network.module.extract_vector(_inputs.to(self._device)))
            else:
                _vectors = tensor2numpy(self._network.extract_vector(_inputs.to(self._device)))

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _compute_class_mean(self, data_manager):
        """Mean feature cuoi f cua moi lop moi (anh train, transform test); dung cho L_orth (gamma > 0)."""
        if getattr(self, "_class_means", None) is not None:
            assert self._class_means.shape[0] == self._known_classes
            new_class_means = np.zeros((self._total_classes, self.feature_dim))
            new_class_means[:self._known_classes] = self._class_means
            self._class_means = new_class_means
        else:
            self._class_means = np.zeros((self._total_classes, self.feature_dim))

        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx + 1), source='train',
                                                                  mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            self._class_means[class_idx, :] = np.mean(vectors, axis=0)

    def displacement(self, Y1, Y2, embedding_old, sigma):
        """SDC: dich chuyen trung binh cua feature moi, trong so Gaussian theo khoang cach toi mean lop cu. Tinh theo
        cum lop cu (moi hang doc lap, ket qua y het ban tinh mot lan) de RAM khong tang theo so lop x so anh."""
        DY = Y2 - Y1
        chunk = max(1, int(5e7 // (Y1.shape[0] * Y1.shape[1])))
        out = []
        for i in range(0, embedding_old.shape[0], chunk):
            E = embedding_old[i:i + chunk]
            distance = np.sum((np.tile(Y1[None, :, :], [E.shape[0], 1, 1]) - np.tile(
                E[:, None, :], [1, Y1.shape[0], 1])) ** 2, axis=2)
            W = np.exp(-distance / (2 * sigma ** 2)) + 1e-5
            W_norm = W / np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
            out.append(np.sum(np.tile(W_norm[:, :, None], [
                1, 1, DY.shape[1]]) * np.tile(DY[None, :, :], [W.shape[0], 1, 1]), axis=1))
        return np.concatenate(out)
