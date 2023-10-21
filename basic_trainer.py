import torch
import logging
from utils import *
from utils_trainer import *
from base_trainer import BaseTrainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicTrainer(BaseTrainer):
    def train(self):
        assert self.args.lqf is False
        thaw(self.args, self.model, mode='regular')
        params = collect_params(self.model)
        self.optimizer = get_optim(params, self.args.optim, self.args.lr, self.args.momentum, nesterov=self.args.nesterov)

        for epoch in range(self.args.epochs):
            adjust_learning_rate(self.optimizer, epoch, self.args.sched_type, self.args.lr, self.args.lr_decay, self.args.steps)
            train_epoch(self.args, self.model, self.train_loader, self.criterion, self.regularizer, self.optimizer, wd=self.args.wd, epoch=epoch, mode='regular')

            if self.args.save_epochs != -1 and epoch % self.args.save_epochs == 0 and epoch > 0:
                model_out_path = f"checkpoints/{self.args.name}-{epoch}.pt"
                torch.save(self.model.state_dict(), model_out_path)
                print(f"Saved model to {model_out_path}")

            if epoch % self.args.val_freq == 0 or epoch == self.args.epochs - 1:
                validate_epoch(self.args, self.model, self.val_loader, epoch=epoch)
