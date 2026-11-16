"""RSIAT (adapter ViT) + LBR cho class-incremental learning.

Moi task: train adapter + head (CosFace tren lop moi), chong quen bang RAE (beta * MSE giua f moi va AE(f cu)). L_RS
(lambda_rs, task 0) va L_orth (gamma) giu nhu RSIAT; cau hinh LBR dat ca hai bang 0.
Bo nho lop cu (models/lbr.py): Gauss N(mu'_c, Sigma'_c) cua x'_CLS (dau vao nhanh MLP/Adapter khoi cuoi).
- Lop moi: mu' tu anh train, Sigma' vay hinh dang (lbr_n0, lbr_tau). lbr_borrow_center: cos vay tinh sau khi tru tam
  x' cua du lieu task 0, lbr_tau_match: tau chon lai de so nguon hieu dung nhu ban khong tru tam; lbr_jitter_rel:
  jitter 1e-3 tr(Sigma')/d thay cho 1e-3.
- Lop cu: sau moi task, mu' dich theo SDC tinh tren x' cua anh task moi qua model cu va moi; kernel Gauss bang thong
  lbr_sdc_sigma (so, mac dinh 4 nhu o f) hoac "adaptive" (median khoang cach tu mean lop cu toi anh gan nhat).
- lbr_diag (mac dinh bat, khong doi ket qua): log [DIAG] moi task: SDC (bang thong, so anh hieu dung), do nen cos giua
  cac mean, phan phuong sai cua vai chieu lon nhat va dieu kien cua Sigma', khoang cach h(mu') - mean mau day xuoi, ti
  so khuech dai (|df|/|f|) / (|dz|/|z|) qua khoi cuoi.
- CA: x' gia ~ N(mu', lbr_ca_s Sigma') day qua khoi cuoi hien tai, train lai head (ca_samples mau/lop, mean nhan
  kappa neu lbr_ca_kappa).
Chong quen tach tai x' (f = h(z), z = x'_CLS; mac dinh tat, khi do giu nguyen RAE tren f):
- lbr_trunk ("ae" | "id"): L_trunk = MSE(h_{t-1}(z_t(x)), P(f_{t-1}(x))) tren anh moi, P = RAE (old_ae) hoac dong nhat;
  gradient chi toi adapter khoi 1..L-1. Thay cho beta * MSE(f_t, P(f_{t-1})).
- lbr_md (lambda) > 0: L_MD = T^2 KL(p_{t-1} || p_t) tren z gia ~ N(mu'_c, Sigma'_c), c ~ Unif(lop cu), lbr_md_k mau
  moi buoc (mac dinh = batch), p = softmax(scale / T cos(h(z), W_old)), W_old = head cu (sau CA task truoc), dong
  bang; gradient chi toi adapter khoi cuoi.
Moi tinh toan cua bo nho x' va CA chay voi RNG rieng (preserve_rng / generator rieng, _ca_ghost_rng tieu thu RNG chung
nhu CA goc) nen backbone di dung quy dao cua RSIAT voi cung cau hinh. Mau cua L_MD lay tu generator rieng.
"""
import logging

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from models.lbr import borrowed_shapes, head_from_xprime, kernel_ess, sdc_bandwidth, train_ca_push, xprime
from utils.inc_net import SimpleVitNet
from utils.loss import AngularPenaltySMLoss
from utils.toolkit import AutoencoderSigmoid, log_count_parameter, preserve_rng, tensor2numpy

NUM_WORKERS = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if 'adapter' not in args["convnet_type"]:
            raise NotImplementedError('Adapter requires Adapter backbone')
        self._network = SimpleVitNet(args, True)
        self.args = args
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.tuned_epochs = None
        self.rs_loss_func = RS_Loss(self.args["alpha"], self.args["rs_margin"])
        self.old_ae = None
        self.lbr_n0, self.lbr_tau = float(args.get("lbr_n0", 20.0)), float(args.get("lbr_tau", 256.0))
        self._xp_means, self._xp_covs = None, None
        self.lbr_sdc_sigma = args.get("lbr_sdc_sigma", 4.0)
        assert self.lbr_sdc_sigma == "adaptive" or float(self.lbr_sdc_sigma) > 0, self.lbr_sdc_sigma
        self.lbr_borrow_center = bool(args.get("lbr_borrow_center", False))
        self.lbr_tau_match = bool(args.get("lbr_tau_match", False))
        self.lbr_jitter_rel = bool(args.get("lbr_jitter_rel", False))
        self.lbr_diag = bool(args.get("lbr_diag", True))
        self._xp_center, self._sdc_prev = None, None
        self.lbr_trunk = args.get("lbr_trunk", "none")
        assert self.lbr_trunk in ("none", "ae", "id"), self.lbr_trunk
        self.lbr_md, self.lbr_md_T = float(args.get("lbr_md", 0.0)), float(args.get("lbr_md_T", 2.0))
        self._md = None

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        self.old_network_module_ptr = self._old_network.module if hasattr(self._old_network, "module") else self._old_network

    def extract_features(self, trainloader, model, args):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.cuda()
                label = label.cuda()
                embedding = model.extract_vector(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        return embedding_list, label_list

    def incremental_train(self, data_manager):
        self._cur_task += 1

        if self._cur_task == 1:
            self.old_ae = AutoencoderSigmoid(input_dims=768, code_dims=self.args["ae_code_dims"])
            self.old_ae.to(self._device)

        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(data_manager.get_task_size(self._cur_task))
        self._network_module_ptr = self._network
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train")
        print("The number of training dataset:", len(train_dataset))

        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=NUM_WORKERS)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=NUM_WORKERS)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        if self._cur_task > 0:
            self._network.to(self._device)
            train_embeddings_old, _ = self.extract_features(self.train_loader, self._network, None)
            xp_old = self._xp_embed(data_manager, self._old_network)
            if self.lbr_md > 0:
                self._md_setup()
            if self.args["gamma"] > 0 and self.args.get("lbr_orth_proto", "f") == "push":
                with torch.no_grad():   # prototype lop cu cho L_orth: h_{t-1}(mu_z) tu bo nho dau task
                    self._orth_protos = head_from_xprime(self.old_network_module_ptr.convnet, torch.as_tensor(
                        np.asarray(self._xp_means[:self._known_classes]), dtype=torch.float32).to(self._device)).float()

        self._train(self.train_loader, self.test_loader)
        if self._md is not None:
            logging.info("[LBR] task %d: L_MD tb %.4f (lambda %g, T %g, %d mau/buoc)", self._cur_task,
                         self._md["sum"] / max(self._md["n"], 1), self.lbr_md, self.lbr_md_T, self._md["k"])
            self._md = None
            torch.cuda.empty_cache()

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        if self._cur_task > 0:
            # SDC: dich mean lop cu (f va x') theo dich chuyen feature cua anh task moi qua model cu -> model moi
            K = self._known_classes
            train_embeddings_new, _ = self.extract_features(self.train_loader, self._network, None)
            if self.args['ssca'] is True:
                self._class_means[:K] += self.displacement(train_embeddings_old, train_embeddings_new,
                                                           self._class_means[:K], 4.0)
            self._xp_sdc(xp_old, self._xp_embed(data_manager, self._network))

        self._compute_class_mean(data_manager)
        with preserve_rng():
            self._lbr_class_stats(data_manager)
            if self.lbr_diag:
                self._lbr_diagnose()
        task_size = data_manager.get_task_size(self._cur_task)

        if self._cur_task > 0 and self.args['ca_epochs'] > 0 and self.args['ca'] is True:
            self._ca_xprime(task_size, self.args['ca_epochs'])

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            self.tuned_epochs = self.args["init_epochs"]
            param_groups = [
                {'params': self._network.convnet.blocks[-1].parameters(), 'lr': 0.01,
                 'weight_decay': self.args['weight_decay']},
                {'params': self._network.convnet.blocks[:-1].parameters(), 'lr': 0.01,
                 'weight_decay': self.args['weight_decay']},
                {'params': self._network.fc.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']}
            ]

            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(param_groups, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)
            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epochs, eta_min=self.min_lr)
            log_count_parameter(param_groups)
            self._init_train(train_loader, test_loader, optimizer, scheduler, self.args['warmup_epoch'])
        else:
            self.tuned_epochs = self.args['inc_epochs']
            m = float(self.args.get("lr_mult_last", 1.0))      # B6: LR rieng cho adapter khoi cuoi (A_L)
            last = list(self._network.convnet.blocks[-1].adaptmlp.parameters()) if m != 1.0 else []
            ids = {id(q) for q in last}
            param_groups = [
                {'params': [q for q in self._network.convnet.parameters() if id(q) not in ids] if last
                 else self._network.convnet.parameters(), 'lr': self.init_lr, 'weight_decay': self.weight_decay},
                {'params': self._network.fc.parameters(), 'lr': self.init_lr, 'weight_decay': self.weight_decay},
                {'params': self.old_ae.parameters(), 'lr': self.args['ae_init_lr'],
                 'weight_decay': self.args['ae_weight_decay']},
            ] + ([{'params': last, 'lr': self.init_lr * m, 'weight_decay': self.weight_decay}] if last else [])

            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(param_groups, momentum=0.9)
            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epochs, eta_min=self.min_lr)
            log_count_parameter(param_groups)
            self._init_train(train_loader, test_loader, optimizer, scheduler, self.args['warmup_epoch'])

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, warmup_epoch):
        prog_bar = tqdm(range(self.tuned_epochs))

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            losses_c, losses_rt = 0.0, 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits, loss_c, loss_rt = self._compute_rt_loss(inputs, targets, epoch, warmup_epoch)
                loss = loss_c + loss_rt
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_c += loss_c.item()
                losses_rt += loss_rt.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info ="Task {}, Epoch {}/{} => Loss {:.3f}, Loss_c {:.3f}, Losses_rt {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.tuned_epochs,
                losses / len(train_loader),
                losses_c / len(train_loader),
                losses_rt / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)
        logging.info(info)

    def _inc_loss(self, features, features_old, proj=True):
        """beta * MSE(features, P(features_old)) + L_orth (gamma); P = RAE (old_ae) neu proj, khong thi dong nhat."""
        P = self.old_ae if proj else (lambda u: u)
        features_old = P(features_old)
        loss_align = nn.MSELoss()(features, features_old)
        if self.args["gamma"] == 0:
            return self.args["beta"] * loss_align
        features_old_norm = F.normalize(features_old, p=2, dim=1)
        protos = (self._orth_protos if self.args.get("lbr_orth_proto", "f") == "push"
                  else torch.from_numpy(self._class_means).float().to(self._device, non_blocking=True))
        protos = P(protos)
        protos = F.normalize(protos, p=2, dim=1)
        similarity = torch.matmul(protos, features_old_norm.t())
        loss_orth = similarity.sum() / (similarity.shape[0] * similarity.shape[1])
        return self.args["beta"] * loss_align + self.args["gamma"] * loss_orth

    def _compute_rt_loss(self, inputs, targets, epoch=None, warmup_epoch=10):
        loss_cos = AngularPenaltySMLoss(loss_type='cosface', eps=1e-7, s=self.args["scale"], m=self.args["margin"])
        split = self._cur_task > 0 and self.lbr_trunk != "none"
        if split:
            xp = xprime(self._network_module_ptr.convnet, inputs)
            features = head_from_xprime(self._network_module_ptr.convnet, xp)
        else:
            features = self._network_module_ptr.extract_vector(inputs)
        logits = self._network_module_ptr.fc(features)["logits"]
        loss_c = loss_cos(logits[:, self._known_classes:], targets - self._known_classes)

        if self._cur_task == 0:
            lambda_rs = self.args["lambda_rs"] * min(1.0, epoch / warmup_epoch)
            loss_base = lambda_rs * self.rs_loss_func(features, targets)
            return logits, loss_c, loss_base

        features_old = self.old_network_module_ptr.extract_vector(inputs)
        if split:   # L_trunk: z_t qua head cu dong bang, gradient chi toi khoi 1..L-1
            loss_rt = self._inc_loss(head_from_xprime(self.old_network_module_ptr.convnet, xp), features_old,
                                     proj=self.lbr_trunk == "ae")
        else:
            loss_rt = self._inc_loss(features, features_old)
        if self._md is not None:
            loss_md = self._md_loss()
            self._md["sum"] += loss_md.item()
            self._md["n"] += 1
            loss_rt = loss_rt + self.lbr_md * loss_md
        return logits, loss_c, loss_rt

    def _xp_cholesky(self, n):
        """Cholesky (float, tren GPU) cua Sigma' cua n lop dau; tinh theo cum bang double."""
        d = self.feature_dim
        L = torch.empty(n, d, d, device=self._device)
        for i in range(0, n, 16):
            Li, info = torch.linalg.cholesky_ex(self._xp_covs[i:i + 16].double().to(self._device))
            assert int(info.abs().max()) == 0, "Sigma' khong xac dinh duong"
            L[i:i + 16] = Li.float()
            del Li
        return L

    def _md_setup(self):
        """Bo nho lop cu cho L_MD (vi tri dau task): mu', Cholesky Sigma', head cu W_old, generator rieng."""
        K, t = self._known_classes, self._cur_task
        old = self.old_network_module_ptr
        W_old = torch.cat([h[0].weight for h in old.fc.heads]).detach().float().to(self._device)
        assert W_old.size(0) == K
        self._md = {
            "MU": torch.as_tensor(np.asarray(self._xp_means[:K]), dtype=torch.float32).to(self._device),
            "L": self._xp_cholesky(K),
            "W": F.normalize(W_old, dim=1),
            "k": int(self.args.get("lbr_md_k", self.batch_size)),
            "gen": torch.Generator(device=self._device).manual_seed(1000 * int(self.args.get("ca_seed", 0)) + 500 + t),
            "sum": 0.0, "n": 0,
        }

    def _md_loss(self):
        """T^2 KL(p_{t-1} || p_t) tren old-class posterior cua z gia ~ N(mu'_c, Sigma'_c), c ~ Unif(lop cu)."""
        m, T, s = self._md, self.lbr_md_T, float(self.args["scale"])
        K, d = m["MU"].shape
        c = torch.randint(K, (m["k"],), device=self._device, generator=m["gen"])
        eps = torch.randn(m["k"], d, device=self._device, generator=m["gen"])
        z = m["MU"][c] + torch.einsum("nij,nj->ni", m["L"][c], eps)
        with torch.no_grad():
            f_old = head_from_xprime(self.old_network_module_ptr.convnet, z)
            p_old = F.softmax(s / T * F.linear(F.normalize(f_old, dim=1), m["W"]), dim=1)
        f_new = head_from_xprime(self._network_module_ptr.convnet, z)
        logq = F.log_softmax(s / T * F.linear(F.normalize(f_new, dim=1), m["W"]), dim=1)
        return T * T * F.kl_div(logq, p_old, reduction="batchmean")

    @torch.no_grad()
    def _xp_embed(self, data_manager, network):
        """x'_CLS cua anh train task hien tai (transform test, thu tu co dinh) qua network, RNG giu nguyen."""
        net = network.module if hasattr(network, "module") else network
        was = net.training
        net.eval()
        ds = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test")
        with preserve_rng():
            out = torch.cat([xprime(net.convnet, x.to(self._device)).double().cpu()
                             for _, x, _ in DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)])
        net.train(was)
        return out

    @torch.no_grad()
    def _lbr_class_stats(self, data_manager):
        """Gauss cua x'_CLS cho moi lop moi (anh train, transform test); Sigma' vay hinh dang tu Sigma' lop cu va
        covariance mau cua cac lop moi khac (models/lbr.borrowed_shapes)."""
        K, T, d = self._known_classes, self._total_classes, self.feature_dim
        net = self._network.module if hasattr(self._network, "module") else self._network
        net.eval()
        feats = []
        for c in range(K, T):
            _, _, ds = data_manager.get_dataset(np.arange(c, c + 1), source="train", mode="test", ret_data=True)
            feats.append(torch.cat([xprime(net.convnet, x.to(self._device)).float().cpu()
                                    for _, x, _ in DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)]))
        mu_new = torch.stack([f.double().mean(0) for f in feats])
        mu_old = (torch.as_tensor(self._xp_means[:K], dtype=torch.float64) if K > 0
                  else torch.zeros(0, d, dtype=torch.float64))
        S_old = self._xp_covs[:K] if K > 0 else torch.zeros(0, d, d)
        if self._cur_task == 0:                      # tam chung cua x' (du lieu task 0), co dinh ve sau
            self._xp_center = torch.cat(feats).double().mean(0)
        S_new, info = borrowed_shapes(feats, mu_new, mu_old, S_old, self.lbr_n0, self.lbr_tau, self._device,
                                      center=self._xp_center if self.lbr_borrow_center else None,
                                      tau_match=self.lbr_tau_match, jitter_rel=self.lbr_jitter_rel)
        means, covs = np.zeros((T, d)), torch.zeros((T, d, d))
        if K > 0:
            means[:K], covs[:K] = self._xp_means[:K], self._xp_covs[:K]
        means[K:], covs[K:] = mu_new.numpy(), S_new
        self._xp_means, self._xp_covs = means, covs
        torch.cuda.empty_cache()
        logging.info("[LBR] task %d: Gauss x' cho %d lop moi (n0 = %g, tau = %g), |x'| tb %.1f", self._cur_task, T - K,
                     self.lbr_n0, self.lbr_tau, float(mu_new.norm(dim=1).mean()))
        if info:
            logging.info("[LBR] task %d: vay hinh dang (tru tam %s): tau %.4g, so nguon hieu dung %.2f (khong tru tam, "
                         "tau %g: %.2f)", self._cur_task, self.lbr_borrow_center, info["tau"], info["ppl"],
                         self.lbr_tau, info["ppl_raw"])
        if self.lbr_diag:
            X = torch.cat(feats).double()
            v = X.var(0)
            Xw = torch.cat([f.double() - f.double().mean(0) for f in feats])
            vw = Xw.pow(2).mean(0)
            logging.info("[DIAG] task %d: phuong sai theo chieu cua x': top5 chieu chiem %.1f%% (toan bo), %.1f%% (trong "
                         "lop); chieu lon nhat/median %.1f", self._cur_task, 100 * float(v.topk(5).values.sum() / v.sum()),
                         100 * float(vw.topk(5).values.sum() / vw.sum()), float(vw.max() / vw.median()))

    def _xp_sdc(self, xp_old, xp_new):
        """SDC tai x': dich mu' lop cu theo dich chuyen x' cua anh task moi (model cu -> moi), bang thong co dinh hoac
        thich nghi; luu mean truoc/sau cho chan doan."""
        K = self._known_classes
        mu = np.asarray(self._xp_means[:K])
        s_ad = sdc_bandwidth(xp_old, mu)
        sigma = s_ad if self.lbr_sdc_sigma == "adaptive" else float(self.lbr_sdc_sigma)
        shift = self.displacement(xp_old.numpy(), xp_new.numpy(), mu, sigma)
        self._sdc_prev = mu.copy()
        self._xp_means[:K] = self._xp_means[:K] + shift
        logging.info("[LBR] task %d: SDC x' bang thong %.3g (%s), so anh hieu dung median %.1f / %d (sigma 4: %.1f, "
                     "thich nghi %.3g: %.1f), |dich| tb %.3f, |mu'| tb %.1f", self._cur_task, sigma, self.lbr_sdc_sigma,
                     float(kernel_ess(xp_old, mu, sigma).median()), xp_old.size(0),
                     float(kernel_ess(xp_old, mu, 4.0).median()), s_ad, float(kernel_ess(xp_old, mu, s_ad).median()),
                     float(np.linalg.norm(shift, axis=1).mean()), float(np.linalg.norm(mu, axis=1).mean()))

    @torch.no_grad()
    def _lbr_diagnose(self, n=256):
        """Chan doan bo nho x' (khong doi ket qua: RNG rieng, chi doc): nen cos giua cac mean (tho / tru tam task 0),
        dieu kien Sigma', |h(mu') - mean h(z)| / |mean h(z)| va ti so khuech dai qua khoi cuoi
        (|h(z) - h(mu')| / |h(mu')|) / (|z - mu'| / |mu'|) tren n mau moi lop; voi lop cu them ti so cho dich SDC."""
        T, t = self._total_classes, self._cur_task
        net = self._network.module if hasattr(self._network, "module") else self._network
        was = net.training
        net.eval()
        vit, dev = net.convnet, self._device
        MU = torch.as_tensor(np.asarray(self._xp_means[:T]), dtype=torch.float64).to(dev)
        off = ~torch.eye(T, dtype=torch.bool, device=dev)
        cos_raw = (F.normalize(MU, dim=1) @ F.normalize(MU, dim=1).t())[off]
        Mc = MU - self._xp_center.to(dev)
        cos_c = (F.normalize(Mc, dim=1) @ F.normalize(Mc, dim=1).t())[off]
        gen = torch.Generator(device=dev).manual_seed(7000 + t)
        gap, amp, cond, jit = [], [], [], []
        for c in range(T):
            S = self._xp_covs[c].double().to(dev)
            ev = torch.linalg.eigvalsh(S)
            cond.append(float(ev[-1] / ev[0]))
            jit.append(float(torch.diagonal(S).mean()))
            Z = MU[c] + torch.randn(n, MU.size(1), device=dev, dtype=torch.float64, generator=gen) @ \
                torch.linalg.cholesky(S).t()
            Fz = head_from_xprime(vit, Z.float()).double()
            f0 = head_from_xprime(vit, MU[c:c + 1].float()).double()[0]
            gap.append(float((f0 - Fz.mean(0)).norm() / Fz.mean(0).norm()))
            amp.append(((Fz - f0).norm(dim=1) / f0.norm()) / ((Z - MU[c]).norm(dim=1) / MU[c].norm()))
        amp = torch.cat(amp)
        q = lambda x, p: float(torch.quantile(torch.as_tensor(x, dtype=torch.float64), p))
        logging.info("[DIAG] task %d: cos giua mean x' tho tb %.3f [%.3f, %.3f], tru tam tb %.3f [%.3f, %.3f]", t,
                     float(cos_raw.mean()), float(cos_raw.min()), float(cos_raw.max()), float(cos_c.mean()),
                     float(cos_c.min()), float(cos_c.max()))
        logging.info("[DIAG] task %d: Sigma' dieu kien median %.3g max %.3g, tr/d median %.3g (jitter co dinh 1e-3)", t,
                     q(cond, 0.5), max(cond), q(jit, 0.5))
        logging.info("[DIAG] task %d: |h(mu') - mean h(z)| / |mean h(z)| median %.4f max %.4f; khuech dai mau median "
                     "%.3f p90 %.3f", t, q(gap, 0.5), max(gap), float(amp.median()), q(amp, 0.9))
        if t > 0 and self._sdc_prev is not None:
            K = self._sdc_prev.shape[0]
            P = torch.as_tensor(self._sdc_prev, dtype=torch.float64).to(dev)
            fp = head_from_xprime(vit, P.float()).double()
            fn = head_from_xprime(vit, MU[:K].float()).double()
            r = ((fn - fp).norm(dim=1) / fp.norm(dim=1)) / ((MU[:K] - P).norm(dim=1) / P.norm(dim=1))
            logging.info("[DIAG] task %d: khuech dai cua dich SDC median %.3f p90 %.3f", t, float(r.median()),
                         q(r, 0.9))
        net.train(was)
        torch.cuda.empty_cache()

    def _ca_xprime(self, task_size, ca_epochs):
        """CA o tang x': x' gia cua moi lop ~ N(mu'_c, s Sigma'_c) day qua khoi cuoi hien tai -> f gia, train lai head
        (models/lbr.train_ca_push) voi RNG rieng (1000 * ca_seed + t), roi tieu thu RNG chung nhu CA goc."""
        dev, T, t = self._device, self._total_classes, self._cur_task
        net = self._network.module if hasattr(self._network, "module") else self._network
        fc = net.fc
        for p in fc.parameters():
            p.requires_grad = True
        net.to(dev)
        net.eval()
        sd = fc.state_dict()
        keys = sorted(sd, key=lambda k: int(k.split(".")[1]))
        W0 = torch.cat([sd[k] for k in keys]).float().to(dev)
        kappa = bool(self.args.get("lbr_ca_kappa", True))
        tid = torch.div(torch.arange(T), task_size, rounding_mode="floor").double()
        kap = (0.9 + 0.1 * (tid + 1) / (t + 1)).float() if kappa else None
        MU = torch.as_tensor(np.asarray(self._xp_means[:T]), dtype=torch.float32).to(dev)
        L = self._xp_cholesky(T)
        s_ca = float(self.args.get("lbr_ca_s", 1.0))
        if s_ca != 1.0:
            L.mul_(s_ca ** 0.5)
        ca = {"scale": float(self.args["scale"]), "init_lr": float(self.init_lr), "weight_decay": self.weight_decay,
              "ca_epochs": int(ca_epochs), "samples": int(self.args.get("ca_samples", 256))}
        gen = torch.Generator(device=dev).manual_seed(1000 * int(self.args.get("ca_seed", 0)) + t)
        with preserve_rng():
            W = train_ca_push(net.convnet, W0, MU, L, kap, ca, gen)
        del L
        fc.load_state_dict({k: w.to(sd[k].dtype) for k, w in zip(keys, torch.split(W, [sd[k].size(0) for k in keys]))})
        self._ca_ghost_rng(T, ca_epochs, W0.size(1))
        with preserve_rng():
            acc = self._compute_accuracy(self._network, self.test_loader)
        logging.info("CA Task {} o tang x' ({} mau/lop, kappa {}, s {}): Test_accy {:.3f}".format(
            t, ca["samples"], kappa, s_ca, acc))
        torch.cuda.empty_cache()


class RS_Loss(nn.Module):
    def __init__(self, lamda=0.5, margin=0.5):
        super(RS_Loss, self).__init__()
        self.lamda = lamda
        self.margin = margin

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, p=2, dim=1)
        labels = labels[:, None]
        mask = torch.eq(labels, labels.t()).float().to(device)
        eye = torch.eye(mask.size(0), device=device)
        mask_pos = mask - eye
        mask_neg = 1.0 - mask
        dot_prod = torch.matmul(features, features.t())

        pos_loss = F.relu(1.0 - dot_prod) * mask_pos
        neg_loss = F.relu(dot_prod - self.margin) * mask_neg
        loss = pos_loss.sum() / (mask_pos.sum() + 1e-6) + \
               self.lamda * neg_loss.sum() / (mask_neg.sum() + 1e-6)
        return loss
