
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional

import torch
from torch.optim import Optimizer


class WarmupLRScheduler(_LRScheduler):
    """
    Warmup learning rate until `total_steps`

    Args:
        optimizer (Optimizer): wrapped optimizer.
        configs (DictConfig): configuration set.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps=200,
        init_lr=5e-6,
        peak_lr=1e-4,
        total_steps=5000
    ) -> None:
        self.optimizer = optimizer
        self.init_lr=1e-7
        if warmup_steps != 0:
            warmup_rate = peak_lr - init_lr
            self.warmup_rate = warmup_rate / warmup_steps
        else:
            self.warmup_rate = 0
        self.update_steps = 1
        self.lr = init_lr
        self.warmup_steps = warmup_steps

    @staticmethod
    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g["lr"] = lr

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g["lr"]

    def step(self, val_loss: Optional[torch.FloatTensor] = None):
        if self.update_steps < self.warmup_steps:
            lr = self.init_lr + self.warmup_rate * self.update_steps
            self.set_lr(self.optimizer, lr)
            self.lr = lr
        self.update_steps += 1
        return self.lr