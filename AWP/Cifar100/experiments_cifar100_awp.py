#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File    :   experiments_cifar100_awp.py
    @Time    :   2021/11/10 20:57:37
    @Author  :   Qingzhong Ai 
    @Contact :   aqz1995@163.com
    @Desc    :   None
'''
import argparse
import os
import sys
sys.path.append("../..")
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from models_cifar100_awp import *
from utils.data_loader import data_loader_cifar100
from utils.attacks import PGD
from utils.helper import AverageMeter, accuracy, save_checkpoint, set_seed, parse_config_file, adjust_learning_rate_1

from managpu import GpuManager
my_gpu = GpuManager()
using_gpu = my_gpu.set_by_memory(1)
print("Using GPU: ", using_gpu)

CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

mu = torch.tensor(CIFAR100_MEAN).view(3,1,1).cuda()
std = torch.tensor(CIFAR100_STD).view(3,1,1).cuda()

def normalize(X):
    return (X - mu)/std

upper_limit, lower_limit = 1,0

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Cifar100 AWP Training')
    parser.add_argument('--data', metavar='DIR', default='/hdd1/aiqingzhong/CIFAR100/',
                        help='path to dataset')
    parser.add_argument('-c', '--config', default='configs.yml', type=str, metavar='Path',
                        help='path to the config file (default: configs.yml)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoitn, (default: None)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--attack_method', default='PGD', type=str, metavar='PATH',
                        help='attack method in validation, (default: PGD)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    return parser.parse_args()

best_prec1 = 0.0

def main():
    global args, best_prec1
    args = parse_config_file(parse_args())
    # print(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    set_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.awp_gamma <= 0.0:
        args.awp_warmup = np.infty

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))

    if args.arch == 'PreActResNet18':
        model = PreActResNet18(dataset="CIFAR100")
        proxy = PreActResNet18(dataset="CIFAR100")
    elif args.arch == 'WideResNet':
        raise NotImplementedError
    else:
        raise NotImplementedError

    # use cuda
    model = model.to(device)
    proxy = proxy.to(device)

    # define loss and optimizer
    if args.l2:
        decay, no_decay = [], []
        for name,param in model.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        params = [{'params':decay, 'weight_decay':args.l2},
                  {'params':no_decay, 'weight_decay': 0 }]
    else:
        params = model.parameters()

    optimizer = optim.SGD(params, lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    proxy_optimizer = optim.SGD(proxy.parameters(), lr=0.01)
    awp_adversary = AdvWeightPerturb(model=model, proxy=proxy, proxy_optim=proxy_optimizer, gamma=args.awp_gamma)

    
    criterion = nn.CrossEntropyLoss().to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading
    train_loader, val_loader = data_loader_cifar100(args.data, args.batch_size, args.workers, args.pin_memory)

    if args.evaluate:
        # PGD10
        print("=> evaluate.tar_num_step:{},step_size:{}".format(args.num_steps_1, args.step_size_1))
        validate(val_loader, model, criterion, args.print_freq, device, args.num_steps_1, args.step_size_1)

        # PGD50
        print("=> evaluate.tar_num_step:{},step_size:{}".format(args.num_steps_2, args.step_size_2))
        validate(val_loader, model, criterion, args.print_freq, device, args.num_steps_2, args.step_size_2)

        # PGD100
        print("=> evaluate.tar_num_step:{},step_size:{}".format(args.num_steps_3, args.step_size_3))
        validate(val_loader, model, criterion, args.print_freq, device, args.num_steps_3, args.step_size_3)
        return

    # Create output file
    cur_dir = os.getcwd()
    dir = cur_dir + '/checkpoint_Cifar100_AWP1/' + str(args.method_name) + '/' + str(args.arch) + '-bs' + str(
        args.batch_size) + '-lr' + str(args.lr) + '-momentum' + str(args.momentum) + '-wd' + str(
        args.weight_decay) + '-seed' + str(args.seed) + '/'
    print("Output dir:" + dir)
    model_dir = dir + 'model_pth/'
    best_model_dir = dir + 'best_model_pth/'
    log_dir = dir + 'log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)

    # Training Process
    for epoch in range(args.start_epoch, args.epochs):
        
        # train for one epoch
        train(train_loader, model, awp_adversary, criterion, optimizer, epoch, args.print_freq, device, log_dir)

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, criterion, args.print_freq, device, args.num_steps_2, args.step_size_1, log_dir)

        # remember the best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best,
            model_dir + 'at_numstep' +  str(args.num_steps_1) + '_epsilon' + str(int(args.epsilon*255)) +
            '_r' + str(args.r) +
            '_canny_sigma' + str(args.sigma) + '_alpha' + str(args.alpha) +
            '-bs' + str(args.batch_size) + '-lr_' + str(args.lr) +
            '-w' + str(args.w) + '-gf' + str(args.gf) +
            '-l' + str(args.low)+ '-h' + str(args.high)+ '_' + str(epoch) + '.pth',
            best_model_dir + 'at_numstep' + str(args.num_steps_1) + '_epsilon' + str(int(args.epsilon*255)) +
            'r' + str(args.r) +
            '_canny_sigma' + str(args.sigma) + '_alpha' + str(args.alpha) +
            '-bs' + str(args.batch_size) + '-lr_' + str(args.lr) +
            '-w' + str(args.w) + '-gf' + str(args.gf) +
            '-l' + str(args.low)+ '-h' + str(args.high) + '.pth'

        )

def train(train_loader, model, awp_adversary, criterion, optimizer, epoch, print_freq, device, log_dir):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):

        target = target.to(device)
        input = input.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        epoch_index = epoch + (i + 1) / len(train_loader)
        adjust_learning_rate_1(optimizer, epoch_index, args.lr, args.epochs)
        
        # compute output
        if args.method_name == 'AT_AWP':
            data_adv = PGD(model, args, input, target, args.num_steps_1, args.step_size_1)
            # delta = attack_pgd(model, input, target, args.epsilon, args.step_size_1, args.num_steps_1, args.restarts, args.norm)
            # delta = delta.detach()
            # data_adv = normalize(torch.clamp(input + delta[:input.size(0)], min=lower_limit, max=upper_limit))
            # calculate adversarial weight perturbation and perturb it
            if epoch >= args.awp_warmup:
                # not compatible to mixup currently.
                # assert (not args.mixup)
                awp = awp_adversary.calc_awp(inputs_adv=data_adv, targets=target)
                awp_adversary.perturb(awp)
            robust_output = model(data_adv)
            # compute loss
            robust_loss = criterion(robust_output, target)
        else:
            raise NotImplementedError('Wrong method name!')
        
        if args.l1:
            for name,param in model.named_parameters():
                if 'bn' not in name and 'bias' not in name:
                    robust_loss += args.l1*param.abs().sum()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        robust_loss.backward()
        optimizer.step()
        
        if epoch >= args.awp_warmup:
                awp_adversary.restore(awp)
        
        # output = model(input)
        # loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(robust_output.data, target, topk=(1, 5))
        losses.update(robust_loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print_content = 'Epoch: [{0}][{1}/{2}]\t' \
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                            'Robust Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                            epoch, i, len(train_loader), batch_time=batch_time,
                            data_time=data_time, loss=losses, top1=top1, top5=top5)
            print(print_content)
            with open(log_dir + 'log.txt', 'a') as f:
                print(print_content, file=f)


def validate(val_loader, model, criterion, print_freq, device, num_steps, step_size, log_dir):
    batch_time = AverageMeter()
    losses_cle = AverageMeter()
    top1_cle = AverageMeter()
    top5_cle = AverageMeter()

    losses_adv = AverageMeter()
    top1_adv = AverageMeter()
    top5_adv = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # target = target.cuda(async=True)
        # input = input.cuda(async=True)

        target = target.to(device)
        input = input.to(device)

        if args.attack_method == 'PGD':
            data_adv = PGD(model, args, input, target, num_steps, step_size)
            # delta = attack_pgd(model, input, target, args.epsilon, step_size, num_steps, args.restarts, args.norm)
            # delta = delta.detach()
            # data_adv = normalize(torch.clamp(input + delta[:input.size(0)], min=lower_limit, max=upper_limit))
        else:
            raise NotImplementedError

        with torch.no_grad():
            # compute output
            # output_clean = model(normalize(input))
            output_clean = model(input)
            output_adv = model(data_adv)

            loss_clean = criterion(output_clean, target)
            loss_adv = criterion(output_adv, target)

            # measure accuracy and record loss
            prec1_cle, prec5_cle = accuracy(output_clean.data, target, topk=(1, 5))
            losses_cle.update(loss_clean.item(), input.size(0))
            top1_cle.update(prec1_cle[0], input.size(0))
            top5_cle.update(prec5_cle[0], input.size(0))

            prec1_adv, prec5_adv = accuracy(output_adv.data, target, topk=(1, 5))
            losses_adv.update(loss_adv.item(), input.size(0))
            top1_adv.update(prec1_adv[0], input.size(0))
            top5_adv.update(prec5_adv[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                test_clean_content = 'Test_clean: [{0}/{1}]\t' \
                                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                    i, len(val_loader), batch_time=batch_time, loss=losses_cle,
                                    top1=top1_cle, top5=top5_cle)
                print(test_clean_content)
                with open(log_dir + 'log.txt', 'a') as f:
                    print(test_clean_content, file=f)
                test_adv_content = 'Test_adv: [{0}/{1}]\t' \
                                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                    i, len(val_loader), batch_time=batch_time, loss=losses_adv,
                                    top1=top1_adv, top5=top5_adv)
                print(test_adv_content)
                with open(log_dir + 'log.txt', 'a') as f:
                    print(test_adv_content, file=f)


    print(' * Clean Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1_cle, top5=top5_cle))
    print(' * Adv Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1_adv, top5=top5_adv))
    with open(log_dir + 'log.txt', 'a') as f:
        print(' * Clean Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1_cle, top5=top5_cle), file=f)
        print(' * Adv Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1_adv, top5=top5_adv), file=f)

    return top1_adv.avg, top5_adv.avg


if __name__ == '__main__':
    main()