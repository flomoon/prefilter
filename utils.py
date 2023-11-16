import time
import math
import random
import numpy as np
import torch
import torch.optim as optim

from torch.optim.lr_scheduler import _LRScheduler
from torchvision import models
from pytorch_msssim import MS_SSIM


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def set_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 100000 % 2**32)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def create_optimizer(optim_name, lr, parameters):
    if optim_name == 'sgd':
        optimizer = optim.SGD(parameters, lr=lr, momentum=0.9)
    elif optim_name == 'adam':
        optimizer = optim.Adam(parameters, lr=lr)
    else:
        raise NotImplemented("Currently supported optimizers are 'adam' and 'sgd'.")
    return optimizer


def parse_target(target):
    if target == "resnet18":
        resizing_size = 64 * 4
        cropping_size = 224
        module = models.resnet18(pretrained=True)
    elif target == "inception_v3":
        resizing_size = 64 * 5
        cropping_size = 299
        module = models.inception_v3(pretrained=True)
    elif target == "msssim":
        module = MS_SSIM(data_range=1.0, size_average=True, channel=3)
        resizing_size = 64 * 4
        cropping_size = 256
    else:
        raise NotImplementedError("Currently, only 'resnet18', 'inception_v3' and 'msssim' are suppported.")
    target_dict = dict(
        module=module,
        resizing_size=resizing_size,
        cropping_size=cropping_size)
    return target_dict


def get_feasible_qualities(codec_name):
    if codec_name in ['jpeg', 'webp']:
        qualities = range(1, 101)
    elif codec_name == 'surr':
        qualities = range(1, 9)
    elif codec_name == 'vtm':
        qualities = [22, 27, 32, 37, 42, 47]
    return qualities