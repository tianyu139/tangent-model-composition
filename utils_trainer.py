import torch
import logging
import time

from utils import *
from models.models import get_model


def train_epoch(args, model, train_loader, criterion, regularizer, optimizer=None, wd=1e-4, epoch=0, mode='regular', write_result=True, enable_bn=True):
    if enable_bn:
        model.train()
    else:
        model.eval()

    metrics = AverageMeter()

    start_time = time.time()
    for batch_idx, data in enumerate(train_loader):
        input, target = data
        input, target = input.to(DEVICE), target.to(DEVICE).long()
        output = get_output(args, model, input)

        emp_risk = criterion(args, output, target)
        reg_loss = regularizer(args, model)
        acc = get_acc(output, target)

        loss = emp_risk + wd * reg_loss

        lr_epoch = optimizer.param_groups[-1]['lr']
        print(f'Epoch:{epoch}| LR: {np.round(lr_epoch,7)}| Batch: {batch_idx}/{len(train_loader)} | Acc {acc:.3f} | Loss:{loss:.3f} | Emp risk:{emp_risk:.3f}| Reg:{reg_loss:.3f}', end="\r", flush=True)

        # Update the metrics
        metrics.update(n=input.size(0), loss=loss.item(), emp_risk=emp_risk.item(), acc=acc, reg=reg_loss.item())

        # Update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.test_mode:
            break

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Epoch time: {elapsed:.2f} seconds")

    # Log the metrics
    log_metrics(mode, metrics, epoch)

    if write_result:
        results = {k: v for k, v in metrics.avg.items()}
        results['epoch'] = epoch
        results['time'] = elapsed
        write_results(results, f'train_results.json')
    return metrics


@torch.no_grad()
def validate_epoch(args, model, test_loader, epoch=0, write_result=True, verbose=True, mode='regular'):
    model.eval()

    metrics = AverageMeter()
    for batch_idx, data in enumerate(test_loader):
        input, target = data
        input, target = input.to(DEVICE), target.to(DEVICE).long()
        output = get_output(args, model, input)
        acc = get_acc(output, target)

        metrics.update(n=input.size(0), acc=acc)

        if args.test_mode:
            break

    acc = metrics.avg['acc']

    if write_result:
        results = {
            'val_mode': mode,
            'epoch': epoch,
            'acc': acc,
        }
        write_results(results, f'val_results.json')

    if verbose:
        logging.info(f"[{mode}] Accuracy at epoch {epoch}: {acc:.3f}")

    return acc


@torch.no_grad()
def validate_compose(args, compose_checkpoints, val_loader, write_result=True):
    model_compose = get_model(args, args.num_classes)
    model_compose = model_compose.to(DEVICE)
    model_compose = compose_models(model_compose, compose_checkpoints)
    acc = validate_epoch(args, model_compose, val_loader, epoch=-1, write_result=write_result, mode='compose', verbose=False)
    del model_compose
    return acc
