import copy

from torch import nn

from network.classifier import SimpleContinualLinear


def get_convnet(args):
    """Backbone ViT-B/16 (IN21k) co adapter AdaptFormer. Trong so ViT tai tu timm, adapter khoi tao ngau nhien."""
    from easydict import EasyDict
    from network import vision_transformer_adapter

    name = args["convnet_type"].lower()
    if '_adapter' not in name or args["model_name"] != "adapter":
        raise NotImplementedError("Unknown type {}".format(name))
    tuning_config = EasyDict(
        ffn_adapt=True,
        ffn_option="parallel",
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        ffn_num=args["ffn_num"],
        d_model=768,
        vpt_on=False,
        vpt_num=0,
    )
    if name == "pretrained_vit_b16_224_adapter":
        model = vision_transformer_adapter.vit_base_patch16_224_adapter(
            num_classes=0, global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
    elif name == "pretrained_vit_b16_224_in21k_adapter":
        model = vision_transformer_adapter.vit_base_patch16_224_in21k_adapter(
            num_classes=0, global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
    else:
        raise NotImplementedError("Unknown type {}".format(name))
    widen_last_adapter(model, tuning_config, args)
    model.out_dim = 768
    return model.eval()


def widen_last_adapter(model, tuning_config, args):
    """B6: bottleneck rieng cho adapter khoi cuoi (A_L) neu args["ffn_num_last"] khac ffn_num (khoi tao lai A_L);
    adapter_scale_last: gamma rieng cua A_L; train_mlp_last: mo fine-tune MLP cua khoi cuoi."""
    from network.vision_transformer_adapter import Adapter
    n = int(args.get("ffn_num_last", tuning_config.ffn_num))
    scalar = str(args.get("adapter_scale_last", tuning_config.ffn_adapter_scalar))
    if n != tuning_config.ffn_num or scalar != tuning_config.ffn_adapter_scalar:
        model.blocks[-1].adaptmlp = Adapter(tuning_config, dropout=0.1, bottleneck=n,
                                            init_option=tuning_config.ffn_adapter_init_option,
                                            adapter_scalar=scalar,
                                            adapter_layernorm_option=tuning_config.ffn_adapter_layernorm_option)
    if args.get("train_mlp_last"):       # B6: mo fine-tune MLP cua khoi cuoi (fc1, fc2)
        for m in (model.blocks[-1].fc1, model.blocks[-1].fc2):
            for q in m.parameters():
                q.requires_grad = True
    return model


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()
        print('This is for the BaseNet initialization.')
        self.convnet = get_convnet(args)
        print('After BaseNet initialization.')
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self


class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, freeze_old=False):
        if self.fc is None:
            self.fc = SimpleContinualLinear(self.feature_dim, nb_classes)
        else:
            self.fc.update(nb_classes, freeze_old=freeze_old)

    def extract_vector(self, x):
        return self.convnet(x)

    def forward(self, x):
        return self.fc(self.convnet(x))

    def ca_forward(self, x):
        return self.fc(x)
