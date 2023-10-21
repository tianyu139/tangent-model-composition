#!/usr/bin/env python3
import torch

import torch.nn.functional as F

import json
import logging
from datasets import *
from collections import defaultdict

import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON = 1e-6


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = defaultdict(int)
        self.avg = defaultdict(float)
        self.sum = defaultdict(int)
        self.count = defaultdict(int)

    def update(self, n=1, **val):
        for k in val:
            self.val[k] = val[k]
            self.sum[k] += val[k] * n
            self.count[k] += n
            self.avg[k] = self.sum[k] / self.count[k]


def manual_seed(seed):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seeding {seed}")

############
# TRAINING #
############

def adjust_learning_rate(optimizer, epoch, sched_type, lr, lr_decay, steps):
    if sched_type == 'step':
        steps = [int(k) for k in steps.split(";")]
        lr = lr * lr_decay ** len([x for x in steps if x < epoch])
    elif sched_type == 'exp':
        lr = lr * lr_decay**epoch

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_optim(params, optim_type, lr, momentum, nesterov=False):
    if optim_type == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=0.0, nesterov=nesterov)
    elif optim_type == 'adam':
        optimizer = torch.optim.AdamW(params, lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    else:
        raise ValueError("No such optimizer")

    return optimizer


##############
# PARAMETERS #
##############

def freeze(m):
    for p in m.parameters():
        p.requires_grad = False

def thaw(args, m, mode='regular'):
    assert mode in ['tangent', 'regular']

    freeze(m)
    if mode == 'none':
        return

    for k, p in m.named_parameters():
        if mode == 'regular':
            p.requires_grad = True
        elif mode == 'tangent':
            if 'linear' in k:
                if 'layer' in k:
                    if int(k[5]) >= args.tangent_unfreeze_layer:
                        p.requires_grad = True
                elif 'fc' in k:
                    p.requires_grad = True
                else:
                    raise Exception("No such param")
        else:
            raise Exception(f"No such mode - {mode}")


def collect_params(model, quiet=False):
    parameters = []
    for k, p in model.named_parameters():
        if p.requires_grad:
            if not quiet:
                print(k, p.data.shape)
            parameters.append(p)
    return parameters

def parameter_count(model):
    count = 0
    for p in model.parameters():
        if p.requires_grad:
            count += np.prod(p.shape)
    print(f"Parameter Count: {count}")


##################
# REGULARIZATION #
##################

def get_reg(args):
    if args.reg == 'l1':
        return l1_penalty
    elif args.reg == 'l2':
        return l2_penalty
    else:
        raise ValueError(f"No such arg: {args.reg}")

def l1_penalty(args, model):
    l1_loss = EPSILON
    for k, p in model.named_parameters():
        if p.requires_grad:
            l1_loss += p.abs().sum()
    return l1_loss

def l2_penalty(args, model):
    l2_loss = EPSILON
    for k, p in model.named_parameters():
        if p.requires_grad:
            l2_loss += p.pow(2).sum()
    return l2_loss ** 0.5


##################
# LOSS FUNCTIONS #
##################

def get_loss(args):
    if args.loss == 'ce':
        return ce_loss
    if args.loss == 'mse':
        return mse_loss_label


def ce_loss(args, input, target, reduction='mean'):
    loss = F.cross_entropy(input, target, reduction=reduction)
    return loss

def mse_loss_label(args, input, target, reduction='mean'):
    target = one_hot_embedding(target, args.num_classes)
    target *= args.mse_weight
    loss = F.mse_loss(input, target, reduction=reduction)
    if reduction == 'none':
        loss = loss.mean(1)
    return loss

def one_hot_embedding(labels, num_classes):
    y = torch.eye(int(num_classes))
    return y[list(labels.cpu().numpy())].to(DEVICE)


############
# ACCURACY #
############

def get_error(output, target):
    return 1. - get_acc(output, target)

def get_acc(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    return pred.eq(target.view_as(pred)).float().mean().item()

#########
# MODEL #
#########

def get_output(args, model, input):
    if 'tangent' not in args.trainer.name:
        output = model(input)
    else:
        logits, jvp = model(input)
        output = jvp + logits

    return output


########
# MISC #
########

def write_results(results, filename):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {'meta': None, 'data': []}

    with open(filename, 'w') as f:
        data['data'].append(results)
        json.dump(data, f, default=lambda o: str(o))


def log_metrics(split, metrics, epoch, **kwargs):
    metrics_str = {k: np.round(v, 4) for k,v in metrics.avg.items()}
    print_str = f'[{epoch}] {split} metrics: {metrics_str}'# + json.dumps(metrics.avg)
    logging.info(print_str)


def get_trainer(args, model, train_loader, val_loader, criterion, regularizer):
    if args.trainer.name == 'basic':
        from basic_trainer import BasicTrainer
        trainer = BasicTrainer
    elif args.trainer.name == 'tangent':
        from tangent_trainer import TangentTrainer
        trainer = TangentTrainer
    else:
        raise ValueError("No such trainer")

    return trainer(args, model, train_loader, val_loader, criterion, regularizer)


def compose_models(model, checkpoints):
    n_compose = len(checkpoints)
    compose_weight = None
    for c in checkpoints:
        m = torch.load(c)
        if compose_weight is None:
            compose_weight = m
        else:
            for k in compose_weight.keys():
                compose_weight[k] += m[k]

    for k in compose_weight.keys():
        if compose_weight[k].dtype == torch.long:
            assert 'num_batches_tracked' in k
            compose_weight[k] = torch.div(compose_weight[k], n_compose, rounding_mode='floor')
        else:
            compose_weight[k] = compose_weight[k] / n_compose

    model.load_state_dict(compose_weight, strict=True)
    return model

