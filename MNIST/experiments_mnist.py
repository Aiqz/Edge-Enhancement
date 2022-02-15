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
from models_mnist import *
from utils.data_loader import data_loader_mnist
from utils.attacks import PGD, FGSM, CWLinfAttack, ALP, Trades, AVmixup
from utils.helper import AverageMeter, accuracy, save_checkpoint, set_seed, parse_config_file
from autoattack import AutoAttack
from managpu import GpuManager
my_gpu = GpuManager()
using_gpu = my_gpu.set_by_memory(1)
print("Using GPU: ", using_gpu)

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Mnist Training')
    parser.add_argument('--data', metavar='DIR', default='/hdd/lirong/Frequency_NN/data',
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

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))

    if args.arch == 'Net2':
        model = Net_2()
    elif args.arch == 'Net2_EE':
        model = Net2_EE(r=args.r, w=args.w,
                                     with_gf=args.gf, low=args.low, high=args.high, alpha=args.alpha, sigma=args.sigma)
        print('r:{},w:{},gf:{},low:{},high:{}'.format(args.r, args.w, args.gf, args.low,args.high))
    elif args.arch == 'Net2_EE_square':
        model = Net2_EE_square(r=args.r, w=args.w,
                                     with_gf=args.gf, low=args.low, high=args.high, alpha=args.alpha, sigma=args.sigma, type_canny=args.type_canny, epsilon=args.epsilon, n_queries=args.n_queries)
        print('r:{},w:{},gf:{},low:{},high:{}'.format(args.r, args.w, args.gf, args.low,args.high))

    else:
        raise NotImplementedError

    # use cuda
    model = model.to(device)
    if str(device) == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = False  # if benchmark=True, deterministic will be False
        cudnn.deterministic = True

    # define loss and optimizer

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    if args.method_name == 'ALP':
        criterion = ALP(args.step_size_1, args.epsilon, args.num_steps_1, args.beta)
    elif args.method_name == 'TRADES':
        criterion = Trades(args.step_size_1, args.epsilon, args.num_steps_1, args.beta)
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)
    else:
        criterion = nn.CrossEntropyLoss().to(device)
    

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)


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
    train_loader, val_loader = data_loader_mnist(args.data, args.batch_size, args.workers, args.pin_memory)
    
    # Create output file
    cur_dir = os.getcwd()
    dir = cur_dir + '/checkpoint_MNIST/' + str(args.method_name) + '/' + str(args.arch) + '/' + str(args.type_canny)+ '-bs' + str(
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

    if args.evaluate:
        # PGD40
        print("=> evaluate.tar_num_step:{},step_size:{}".format(args.num_steps_1, args.step_size_1))
        validate(val_loader, model, criterion, args.print_freq, device, args.num_steps_1, args.step_size_1, log_dir)

        # PGD50
        print("=> evaluate.tar_num_step:{},step_size:{}".format(args.num_steps_2, args.step_size_2))
        validate(val_loader, model, criterion, args.print_freq, device, args.num_steps_2, args.step_size_2, log_dir)

        # PGD100
        print("=> evaluate.tar_num_step:{},step_size:{}".format(args.num_steps_3, args.step_size_3))
        validate(val_loader, model, criterion, args.print_freq, device, args.num_steps_3, args.step_size_3, log_dir)
        
        # Auto-attack
        log_dir_aa = log_dir + 'log_aa.txt'
        validate_aa(args, val_loader, model, log_dir_aa)
        return
        


    # Training Process
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args.print_freq, device, log_dir)

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, criterion, args.print_freq, device, args.num_steps_1, args.step_size_1, log_dir)

        scheduler.step()

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

def train(train_loader, model, criterion, optimizer, epoch, print_freq, device, log_dir):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    if args.method_name == 'AVmixup':
        avmixup = AVmixup(args, gamma=2.0, lambda1=1.0, lambda2=0.1, step_size=args.step_size_1,
                      num_steps=args.num_steps_1, num_classes=10, device=device)

    for i, (input, target) in enumerate(train_loader):

        target = target.to(device)
        input = input.to(device)
        # measure data loading time
        data_time.update(time.time() - end)
        
        if args.method_name == 'ST':
            data_adv = input
        elif args.method_name == 'ALP':
            preds = model(input)
            data_adv = criterion.PGD_Linf(model, input, target)
            output = model(data_adv)
        elif  args.method_name == 'TRADES':
            preds = model(input)
            data_adv = criterion.PGD_Linf(model, input, preds)
            output = model(data_adv)
        elif args.method_name == 'AVmixup':
            number_class = 10
            target_onehot = torch.eye(number_class)[target].to(device)
            data_adv, new_target = avmixup.perturb(model, input, target_onehot)
        else:
            data_adv = PGD(model, args, input, target, args.num_steps_1, args.step_size_1)

        # compute output
        if args.method_name == 'ALP':
            loss = criterion.loss(model, preds, output, target, optimizer)
        elif args.method_name == 'TRADES':
            loss = criterion.loss(model, preds, data_adv, target, optimizer)
        elif args.method_name == 'AVmixup':
            output = model(data_adv)
            log_prob = nn.functional.log_softmax(output, dim=1)
            loss = -torch.sum(log_prob * new_target) / input.shape[0]
        else:
            output = model(data_adv)
            loss = criterion(output, target)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
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
        elif args.attack_method == 'FGSM':
            data_adv = FGSM(model, input, target, targeted=False, step_size=step_size)
        elif args.attack_method == 'CW':
            data_adv, _ = CWLinfAttack(x=input, y=target, model=model, magnitude=args.epsilon, previous_p=None, max_eps=args.epsilon, max_iters=20, target=None, cur_device=device)
        else:
            raise NotImplementedError

        with torch.no_grad():
            # compute output
            output_clean = model(input)
            output_adv = model(data_adv)

            if args.method_name == 'ALP' or 'TRADES':
                loss_clean = torch.tensor(0)
                loss_adv = torch.tensor(0)
            else:
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
