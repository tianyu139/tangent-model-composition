#!/usr/bin/env python3
import os
import torch
from omegaconf import DictConfig
import hydra
from hydra.utils import get_original_cwd
import logging
from torch.backends import cudnn
from setproctitle import setproctitle

from models.models import get_model
from datasets.datasets import get_dataset
from utils import *
from utils_trainer import validate_compose


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(config_path="conf", config_name="main")
def main(args: DictConfig):
    # Seed the experiment
    manual_seed(args.seed)

    # Name the test point
    if args.name is None:
         args.name = 'test'
    logging.info(f'Checkpoint name: {args.name}')

    setproctitle(args.name)

    # Create directories for saving
    os.makedirs('checkpoints', exist_ok=True)
    cudnn.benchmark = True

    trainset, testset = get_dataset(args.dataset, root=os.path.join(get_original_cwd(), 'data'), n_tasks=args.n_tasks, task_split=args.task_split)
    if not args.augment:
        trainset.dataset.transform = testset.transform
    args.num_classes = testset.num_classes

    loader_args = {'num_workers': 8, 'pin_memory': False}
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, **loader_args)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, **loader_args)

    # Loss function
    criterion = get_loss(args)
    regularizer = get_reg(args)

    base_name = args.name
    for i in range(args.n_tasks):
        logging.info(f"Starting task {i+1}...")
        print(f'Length of train loader: {len(train_loader.dataset)} \t Length of test loader: {len(val_loader.dataset)}')

        if i == 0:
            model: nn.Module = get_model(args, args.num_classes).to(DEVICE)
            torch.save(model.state_dict(), f"checkpoints/{base_name}-init.pt")
        else:
            model.load_state_dict(torch.load(f"checkpoints/{base_name}-init.pt"))

        args.name = base_name + f'_task{i}' if args.n_tasks > 1 else base_name

        trainer = get_trainer(args, model, train_loader, val_loader, criterion, regularizer)
        trainer.train()

        if args.save_epochs != -1:
            torch.save(model.state_dict(), f"checkpoints/{args.name}-final.pt")

        if args.n_tasks > 1:
            train_loader.dataset.current_task += 1
            # Evaluate composition
            model = model.to('cpu')
            compose_checkpoints = [f"checkpoints/{base_name}_task{j}-final.pt" for j in range(i+1)]
            acc = validate_compose(args, compose_checkpoints, val_loader)
            logging.info(f"Compose accuracy at end of task {i+1}: {acc:.3f}")
            model = model.to(DEVICE)


if __name__ == '__main__':
    main()
