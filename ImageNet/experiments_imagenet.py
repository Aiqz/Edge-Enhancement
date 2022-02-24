#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import argparse
import os
import sys
sys.path.append("..")
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from models_imagenet import *
from utils.data_loader import data_loader_imagenet_dataset
from utils.attacks import PGD
from utils.attacks import targeted_PGD, tar_alp_imagenet, targeted_PGD_trick, compute_loss_and_error
from utils.helper import AverageMeter, accuracy, adjust_learning_rate, save_checkpoint, set_seed, parse_config_file

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from managpu import GpuManager
my_gpu = GpuManager()
using_gpu = my_gpu.set_by_memory(4)
print("Using GPU: ", using_gpu)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', metavar='DIR', default='',
                        help='path to dataset')
    parser.add_argument('-c', '--config', default='configs.yml', type=str, metavar='Path',
                        help='path to the config file (default: configs.yml)')
    parser.add_argument('-p', '--pretrained', default=False, dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint, (default: None)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--attack_method', default='PGD', type=str, metavar='PATH',
                        help='attack method in validation, (default: PGD)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    return parser.parse_args()

best_prec1 = 0.0


def main():
    global args, best_prec1
    args = parse_config_file(parse_args())

    device = torch.device("cuda", args.local_rank)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
   

    rank = torch.distributed.get_rank()
    set_seed(args.seed + rank)

    
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))

    if args.arch == 'resnet18':
        model = resnet18(pretrained=args.pretrained)
    elif args.arch == 'resnet34':
        model = resnet34(pretrained=args.pretrained)
    elif args.arch == 'resnet50':
        model = resnet50(pretrained=args.pretrained)
    elif args.arch == 'resnet101':
        model = resnet101(pretrained=args.pretrained)
    elif args.arch == 'resnet152':
        model = resnet152(pretrained=args.pretrained)
    elif args.arch == 'resnet18_EE':
        model = resnet18_EE(pretrained=args.pretrained, cize=args.cize, r=args.r, w=args.w,
                                         with_gf=args.gf, low=args.low, high=args.high, alpha=args.alpha,
                                         sigma=args.sigma)
        print('r:{},w:{},gf:{},low:{},high:{}'.format(args.r, args.w, args.gf, args.low, args.high))
    elif args.arch == 'resnet34_EE':
        model = resnet34_EE(pretrained=args.pretrained, cize=args.cize, r=args.r, w=args.w,
                                         with_gf=args.gf, low=args.low, high=args.high, alpha=args.alpha,
                                         sigma=args.sigma)
        print('r:{},w:{},gf:{},low:{},high:{}'.format(args.r, args.w, args.gf, args.low, args.high))
    elif args.arch == 'resnet50_EE':
        model = resnet50_EE(pretrained=args.pretrained, cize=args.cize, r=args.r, w=args.w,
                                         with_gf=args.gf, low=args.low, high=args.high, alpha=args.alpha,
                                         sigma=args.sigma)
        print('r:{},w:{},gf:{},low:{},high:{}'.format(args.r, args.w, args.gf, args.low, args.high))
    elif args.arch == 'resnet101_EE':
        model = resnet101_EE(pretrained=args.pretrained, cize=args.cize, r=args.r, w=args.w,
                                         with_gf=args.gf, low=args.low, high=args.high, alpha=args.alpha,
                                         sigma=args.sigma)
        print('r:{},w:{},gf:{},low:{},high:{}'.format(args.r, args.w, args.gf, args.low, args.high))
    elif args.arch == 'resnet152_EE':
        model = resnet152_EE(pretrained=args.pretrained, cize=args.cize, r=args.r, w=args.w,
                                         with_gf=args.gf, low=args.low, high=args.high, alpha=args.alpha,
                                         sigma=args.sigma)
        print('r:{},w:{},gf:{},low:{},high:{}'.format(args.r, args.w, args.gf, args.low, args.high))
    elif args.arch == 'resnet18_EE_square':
        model = resnet18_EE_square(pretrained=args.pretrained, cize=args.cize, r=args.r, w=args.w,
                                         with_gf=args.gf, low=args.low, high=args.high, alpha=args.alpha,
                                         sigma=args.sigma, type_canny=args.type_canny, epsilon=args.epsilon, n_queries=args.n_queries)
    
    elif args.arch == 'resnet18_fd':
        model = resnet18_fd(pretrained=args.pretrained)
    elif args.arch == 'resnet34_fd':
        model = resnet34_fd(pretrained=args.pretrained)
    elif args.arch == 'resnet50_fd':
        model = resnet50_fd(pretrained=args.pretrained)
    elif args.arch == 'resnet101_fd':
        model = resnet101_fd(pretrained=args.pretrained)
    elif args.arch == 'resnet152_fd':
        model = resnet152_fd(pretrained=args.pretrained)
    else:
        raise NotImplementedError

      # use cuda
    # model.cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model.cuda(args.local_rank)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.local_rank)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            loc = 'cuda:{}'.format(args.local_rank)
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading
    train_dataset, val_dataset = data_loader_imagenet_dataset(args.data)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(args.batch_size / args.nGPU), shuffle=False,
                                               num_workers=args.workers, pin_memory=args.pin_memory,
                                               sampler=train_sampler)
    val_sampler = DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(args.batch_size / args.nGPU), shuffle=False,
                                             num_workers=args.workers, pin_memory=args.pin_memory,
                                             sampler=val_sampler)
    
    # Create output file
    cur_dir = os.getcwd()
    dir = cur_dir + '/checkpoint_ImageNet/' + str(args.method_name) + '/' + str(args.arch) + '-bs' + str(
        args.batch_size) + '-lr' + str(args.lr) + '-momentum' + str(args.momentum) + '-wd' + str(
        args.weight_decay) + '-seed' + str(args.seed) + '/'
    print("Output dir:" + dir)
    model_dir = dir + 'model_pth/'
    best_model_dir = dir + 'best_model_pth/'
    log_dir = dir + 'log/'
    if args.local_rank == 0:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
    # 进程同步
    torch.distributed.barrier()

    if args.evaluate:
        validate(val_loader, model, criterion, args.print_freq, device, log_dir)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr)
        # 每一个epoch重新切分数据
        train_sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args.print_freq, device, log_dir)

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, criterion, args.print_freq, device, log_dir)

        # remember the best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if args.local_rank == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict()
            }, is_best,
                model_dir + 'at_numstep' + str(args.num_steps_1) + '_epsilon' + str(int(args.epsilon*255)) +
                '_r' + str(args.r) +
                '_canny_sigma' + str(args.sigma) + '_alpha' + str(args.alpha) +
                '-bs' + str(args.batch_size) + '-lr_' + str(args.lr) +
                '-w' + str(args.w) + '-gf' + str(args.gf) +
                '-l' + str(args.low) + '-h' + str(args.high) +
                '_' + str(epoch) + '.pth',
                best_model_dir + 'at_numstep' + str(args.num_steps_1) + '_epsilon' + str(int(args.epsilon*255)) +
                'r' + str(args.r) +
                '_canny_sigma' + str(args.sigma) + '_alpha' + str(args.alpha) +
                '-bs' + str(args.batch_size) + '-lr_' + str(args.lr) +
                '-w' + str(args.w) + '-gf' + str(args.gf) +
                '-l' + str(args.low) + '-h' + str(args.high) + '.pth'
            )


def train(train_loader, model, criterion, optimizer, epoch, print_freq, device, log_dir):
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
        if args.method_name == 'ST':
            data_adv = input
            output = model(data_adv)
        elif args.method_name == 'tarAT' or args.method_name == 'tarFD' or args.method_name == 'tarEE' or args.method_name == 'tarEE_BPDA3_AT_square':
            data_adv, targeted_labels = targeted_PGD(model, args, input, target, args.num_steps_1, args.step_size_1, 1000, device)
            output = model(data_adv)
        elif args.method_name == 'tarFD_trick' or args.method_name == 'tarEE_trick':
            data_adv, targeted_labels = targeted_PGD_trick(model, args, input, target, args.num_steps_1, args.step_size_1, 1000, device)
            output = model(data_adv)
        elif args.method_name == 'tarALP':
            preds = model(input)
            data_adv, targeted_labels = tar_alp_imagenet(model, args, input, target, args.num_steps_1, args.step_size_1, device)
            output = model(data_adv)
        else:
            data_adv = PGD(model, args, input, target, args.num_steps_1, args.step_size_1)
            output = model(data_adv)

        # compute output
        if args.method_name == 'tarALP':
            loss_robust = 0.5 * nn.functional.cross_entropy(preds, target) + 0.5 * nn.functional.cross_entropy(output, target)
            loss_alp = nn.functional.mse_loss(preds, output)
            loss = loss_robust + args.beta * loss_alp
        elif args.method_name == 'tarFD_trick' or args.method_name == 'tarEE_trick':
            loss = compute_loss_and_error(output,target,args.label_smooth)
        else:
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 and args.local_rank == 0:
            print_content = 'Epoch: [{0}][{1}/{2}]\t' \
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
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

    # switch to evaluate model
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        target = target.to(device)
        input = input.to(device)

        if args.attack_method == 'PGD':
            if "tar" in args.method_name:
                data_adv, targeted_labels = targeted_PGD(model, args, input, target, num_steps, step_size, 1000, device)
            else:
                data_adv = PGD(model, args, input, target, num_steps, step_size)
        else:
            raise NotImplementedError

        with torch.no_grad():
            # compute output
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

            if i % print_freq == 0 and args.local_rank == 0:
                test_clean_content = 'Test_clean: [{0}/{1}]\t'\
                                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'\
                                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'\
                                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                         i, len(val_loader), batch_time=batch_time, loss=losses_cle, top1=top1_cle, top5=top5_cle)
                print(test_clean_content)
                with open(log_dir + 'log.txt', 'a') as f:
                    print(test_clean_content, file=f)
                test_adv_content = 'Test_adv: [{0}/{1}]\t'\
                                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'\
                                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'\
                                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                       i, len(val_loader), batch_time=batch_time, loss=losses_adv, top1=top1_adv, top5=top5_adv)
                print(test_adv_content)
                with open(log_dir + 'log.txt', 'a') as f:
                    print(test_adv_content, file=f)

    top1_cle_gather = [torch.ones_like(top1_cle.avg) for _ in range(
        torch.distributed.get_world_size())]
    top5_cle_gather = [torch.ones_like(top5_cle.avg) for _ in range(
        torch.distributed.get_world_size())]
    top1_adv_gather = [torch.ones_like(top1_adv.avg) for _ in range(
        torch.distributed.get_world_size())]
    top5_adv_gather = [torch.ones_like(top5_adv.avg) for _ in range(
        torch.distributed.get_world_size())]
    torch.distributed.all_gather(top1_cle_gather, top1_cle.avg, async_op=False)
    torch.distributed.all_gather(top5_cle_gather, top5_cle.avg, async_op=False)
    torch.distributed.all_gather(top1_adv_gather, top1_adv.avg, async_op=False)
    torch.distributed.all_gather(top5_adv_gather, top5_adv.avg, async_op=False)
    top1_cle_mean = torch.mean(torch.stack(top1_cle_gather))
    top5_cle_mean = torch.mean(torch.stack(top5_cle_gather))
    top1_adv_mean = torch.mean(torch.stack(top1_adv_gather))
    top5_adv_mean = torch.mean(torch.stack(top5_adv_gather))
    if args.local_rank == 0:
        print(' * Clean Prec@1 {0:.3f} Prec@5 {1:.3f}'.format(top1_cle_mean, top5_cle_mean))
        print(' * Adv Prec@1 {0:.3f} Prec@5 {1:.3f}'.format(top1_adv_mean, top5_adv_mean))
        with open(log_dir + 'log.txt', 'a') as f:
            print(' * Clean Prec@1 {0:.3f} Prec@5 {1:.3f}'.format(top1_cle_mean, top5_cle_mean), file=f)
            print(' * Adv Prec@1 {0:.3f} Prec@5 {1:.3f}'.format(top1_adv_mean, top5_adv_mean), file=f)

    return top1_cle_mean, top5_cle_mean

if __name__ == '__main__':
    main()
