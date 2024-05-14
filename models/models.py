import torch.nn as nn
import timm
from models.linear_resnet import linear_resnet18, linear_resnet34, linear_resnet50
from models.linear_layers import LinearLinear
from torchvision.models import resnet18, resnet34, resnet50
import os
from hydra.utils import get_original_cwd
import numpy as np
import models.vit as vit
import copy
# from ResNet18 import resnet18


def get_models_dict():
    models_dict = {'linear_resnet18': linear_resnet18,
                  'linear_resnet34': linear_resnet34,
                  'linear_resnet50': linear_resnet50,
                  'resnet18': resnet18,
                  'resnet34': resnet34,
                  'resnet50': resnet50}
    return models_dict


def get_model(args, num_classes):
    arch = args.arch
    if 'tangent' in args.trainer.name:
        arch = 'linear_' + arch
        if 'vit' in arch:
            model = _get_linear_vit(arch, args, num_classes)
        else:
            model = _get_linear_resnet(arch, args, num_classes)
    else:
        if 'vit' in arch:
            model = _get_vit(arch, args, num_classes)
        else:
            model = _get_resnet(arch, args, num_classes)

    return model


def _get_resnet(arch, args, num_classes):
    if args.pretrained:
        weights = 'DEFAULT'
    else:
        weights = None

    if arch == 'resnet18':
        model = resnet18(weights=weights)
    elif arch == 'resnet34':
        model = resnet34(weights=weights)
    elif arch == 'resnet50':
        model = resnet50(weights=weights)
    else:
        raise Exception("No such resnet")

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def _get_linear_resnet(arch, args, num_classes):
    choices = {'linear_resnet18': linear_resnet18,
               'linear_resnet34': linear_resnet34,
               'linear_resnet50': linear_resnet50}

    model = choices[arch](pretrained=args.pretrained)
    model.fc = LinearLinear(model.fc.in_features, num_classes)

    return model


def _get_vit(arch, args, num_classes):
    pretrained = args.pretrained
    replace_cls_token = args.replace_cls_token
    if 'vitb16' in arch:
        patch_size = 16
        embed_dim = 768
        depth = 12
        num_heads = 12
    elif 'vitl16' in arch:
        patch_size = 16
        embed_dim = 1024
        depth = 24
        num_heads = 16
    else:
        raise Exception("Not such vit")

    if arch == 'vitb16':
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
    elif arch == 'vitl16':
        model = timm.create_model('vit_large_patch16_224', pretrained=pretrained, num_classes=num_classes)
    elif arch[-6:] == 'blocks':
        from models.vit_parts import Vision_Transformer_Last_N_Blocks
        last_n_blocks = int(arch.split('_')[1])
        model = Vision_Transformer_Last_N_Blocks(
            last_n_blocks,
            replace_cls_token,
            num_classes=num_classes,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )

        if pretrained:
            sd = _get_vit(arch.split('_')[0], pretrained=True, num_classes=num_classes, replace_cls_token=False).state_dict()
            model.load_pretrained_weights(sd, load_fc=False)
    else:
        raise ValueError("No such vit")

    return model



def _get_linear_vit(arch, args, num_classes):
    pretrained = args.pretrained
    replace_cls_token = args.replace_cls_token
    # arch format: linear_vit{b|l}16 or linear_vit{b|l}16_{n_blocks}_blocks
    args_pt = copy.deepcopy(args)
    args_pt.pretrained = True
    if 'vitb16' in arch:
        patch_size = 16
        embed_dim = 768
        depth = 12
        num_heads = 12
        weights = _get_vit('vitb16', args_pt, num_classes).state_dict()
    elif 'vitl16' in arch:
        patch_size = 16
        embed_dim = 1024
        depth = 24
        num_heads = 16
        args_new = copy.deepcopy(args)
        args_new.pretrained = True
        weights = _get_vit('vitl16', args_pt, num_classes).state_dict()
    else:
        raise Exception("Not linear vit")

    if arch[-6:] == 'blocks':
        from models.linear_vit import Linear_Vision_Transformer_Last_N_Blocks
        last_n_blocks = int(arch.split('_')[2])
        model = Linear_Vision_Transformer_Last_N_Blocks(
            last_n_blocks,
            replace_cls_token,
            num_classes=num_classes,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
    else:
        from models.linear_vit import Linear_Vision_Transformer
        model = Linear_Vision_Transformer(
            replace_cls_token,
            args.tangent_unfreeze_layer,
            num_classes=num_classes,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )

    deletes = []
    if not pretrained:
        # Only load weights for nontrainable blocks
        for k,v in weights.items():
            if 'blocks' in k:
                block = int(k.split('.')[1])
                if block >= args.tangent_unfreeze_layer:
                    deletes.append(k)
            elif k.startswith('head') or k.startswith('norm'):
                deletes.append(k)
            elif 'cls' in k or 'pos_embed' in k or 'patch_embed' in k:
                pass
            else:
                print(k)
                raise Exception('unhandled weight')

        for d in deletes:
            del weights[d]

    model.load_pretrained_weights(weights, load_fc=False, deletes=deletes)

    return model

