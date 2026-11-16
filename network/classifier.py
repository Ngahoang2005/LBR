import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers.weight_init import trunc_normal_


class SimpleContinualLinear(nn.Module):
    """Head tuyen tinh khong bias, mot nhom trong so cho moi task. Logit = cosine(feature, w)."""

    def __init__(self, embed_dim, nb_classes):
        super().__init__()
        self.embed_dim = embed_dim
        head = nn.Sequential(nn.Linear(embed_dim, nb_classes, bias=False))
        self.heads = nn.ModuleList([head])
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)

    def update(self, nb_classes, freeze_old=True):
        _fc = nn.Linear(self.embed_dim, nb_classes, bias=False)
        trunc_normal_(_fc.weight, std=.02)
        if freeze_old:
            for p in self.heads.parameters():
                p.requires_grad = False
        self.heads.append(nn.Sequential(_fc))

    def forward(self, x):
        out = []
        for ti in range(len(self.heads)):
            out.append(F.linear(F.normalize(x, p=2, dim=1), F.normalize(self.heads[ti][0].weight, p=2, dim=1)))
        return {'logits': torch.cat(out, dim=1)}
