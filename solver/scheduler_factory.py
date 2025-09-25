""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler
from bisect import bisect_right
from math import cos, pi

from torch.optim.lr_scheduler import _LRScheduler

def create_scheduler(cfg, optimizer):
    # num_epochs = cfg.SOLVER.MAX_EPOCHS
    # # type 1
    # # lr_min = 0.01 * cfg.SOLVER.BASE_LR
    # # warmup_lr_init = 0.001 * cfg.SOLVER.BASE_LR
    # # type 2
    # lr_min = 0.002 * cfg.SOLVER.BASE_LR
    # warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR
    # # type 3
    # # lr_min = 0.001 * cfg.SOLVER.BASE_LR
    # # warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR

    # warmup_t = cfg.SOLVER.WARMUP_EPOCHS
    # noise_range = None

    # lr_scheduler = CosineLRScheduler(
    #         optimizer,
    #         t_initial=num_epochs,
    #         lr_min=lr_min,
    #         t_mul= 1.,
    #         decay_rate=0.1,
    #         warmup_lr_init=warmup_lr_init,
    #         warmup_t=warmup_t,
    #         cycle_limit=1,
    #         t_in_epochs=True,
    #         noise_range_t=noise_range,
    #         noise_pct= 0.67,
    #         noise_std= 1.,
    #         noise_seed=42,
    #     )

    # return lr_scheduler
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=(20, 50),
        gamma=0.1,
        warmup_factor=0.1,
        warmup_epochs=5,
        warmup_method='linear',
        total_epochs=100,
        mode='cosine',
        target_lr=0,
        power=0.9,
    )

class LRSchedulerWithWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        mode="step",
        warmup_factor=1.0 / 3,
        warmup_epochs=10,
        warmup_method="linear",
        total_epochs=100,
        target_lr=0,
        power=0.9,
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of"
                " increasing integers. Got {}".format(milestones),
            )
        if mode not in ("step", "exp", "poly", "cosine", "linear"):
            raise ValueError(
                "Only 'step', 'exp', 'poly' or 'cosine' learning rate scheduler accepted"
                "got {}".format(mode)
            )
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.mode = mode
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        self.total_epochs = total_epochs
        self.target_lr = target_lr
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        if self.mode == "step":
            return [
                base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs
            ]

        epoch_ratio = (self.last_epoch - self.warmup_epochs) / (
            self.total_epochs - self.warmup_epochs
        )

        if self.mode == "exp":
            factor = epoch_ratio
            return [base_lr * self.power ** factor for base_lr in self.base_lrs]
        if self.mode == "linear":
            factor = 1 - epoch_ratio
            return [base_lr * factor for base_lr in self.base_lrs]

        if self.mode == "poly":
            factor = 1 - epoch_ratio
            return [
                self.target_lr + (base_lr - self.target_lr) * self.power ** factor
                for base_lr in self.base_lrs
            ]
        if self.mode == "cosine":
            factor = 0.5 * (1 + cos(pi * epoch_ratio))
            return [
                self.target_lr + (base_lr - self.target_lr) * factor
                for base_lr in self.base_lrs
            ]
        raise NotImplementedError

def bisect_right(a, x, lo=0, hi=None):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(x) will
    insert just after the rightmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if x < a[mid]: hi = mid
        else: lo = mid+1
    return lo


