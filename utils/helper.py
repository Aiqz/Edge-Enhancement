#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import shutil
import torch
import numpy as np
import random
import math
import yaml
from easydict import EasyDict

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    if target.shape == output.shape:
        _, target = target.topk(1, 1, largest=True, sorted=True)

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  ###modified
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, filename, bestfilename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestfilename)


def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_free(optimizer, epoch, init_lr, n_repeats):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // int(math.ceil(30./n_repeats))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_1(optimizer, epoch, init_lr, epochs):
    """The learning rate decay is applied at 50% and 75% of total training steps with decay factor 0.1"""
    if epoch > epochs * 0.75:
        lr = init_lr * (0.1 ** 2)
    elif epoch > epochs * 0.5:
        lr = init_lr * 0.1
    else:
        lr = init_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_attack_success(logits, target_label):
    """
    Compute the attack success rate.
    """

    _, predicted_clean = logits.max(1)

    correct = predicted_clean.eq(target_label).sum().item()

    return correct


def parse_config_file(args):
    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
        
    # Add args parameters to the dict
    for k, v in vars(args).items():
        config[k] = v
        
    # Add the output path
    # config.output_name = '{:s}_step{:d}_eps{:d}_repeat{:d}'.format(args.output_prefix,
    #                      int(config.ADV.fgsm_step), int(config.ADV.clip_eps), 
    #                      config.ADV.n_repeats)
    return config