import torch.nn as nn
import timm
from models.linear_resnet import linear_resnet18, linear_resnet34, linear_resnet50
from models.linear_layers import LinearLinear
from torchvision.models import resnet18, resnet34, resnet50
import os
from hydra.utils import get_original_cwd
import numpy as np
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
        model = _get_linear_resnet(arch, args, num_classes)
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


