#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import argparse
import os
import sys
sys.path.append("..")
import time
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
# import torch.functional as F
from models_tinyimagenet import *
from utils.data_loader import data_loader_tiny_imagenet
from utils.attacks import PGD, FGSM, CWLinfAttack, ALP, Trades, AVmixup
from utils.attacks import targeted_PGD, targeted_ALP
from utils.helper import AverageMeter, accuracy, save_checkpoint, set_seed, parse_config_file, adjust_learning_rate_1
from utils.core import HighFreqSuppress, CannyFilter_pre, get_gaussian_kernel
from managpu import GpuManager
my_gpu = GpuManager()
using_gpu = my_gpu.set_by_memory(1)
print("Using GPU: ", using_gpu)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/hdd/lirong/Frequency_NN/tiny-imagenet-200')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('-c', '--config', default='configs.yml', type=str, metavar='Path',
                        help='path to the config file (default: configs.yml)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoitn, (default: None)')
    parser.add_argument('--n_ex', type=int, default=1000)
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./results/ee')
    parser.add_argument('--log_path', type=str, default='./Log/log_ee_new1_individual.txt')
    parser.add_argument('--version', type=str, default='standard')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')

    global args, best_prec1
    args = parse_config_file(parser.parse_args())
    # print(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    set_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # load model

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
        print('r:{},w:{},gf:{},low:{},high:{}'.format(args.r, args.w, args.gf, args.low,args.high))
    elif args.arch == 'resnet34_EE':
        model = resnet34_EE(pretrained=args.pretrained, cize=args.cize, r=args.r, w=args.w,
                                         with_gf=args.gf, low=args.low, high=args.high, alpha=args.alpha,
                                         sigma=args.sigma)
        print('r:{},w:{},gf:{},low:{},high:{}'.format(args.r, args.w, args.gf, args.low,args.high))
    elif args.arch == 'resnet50_EE':
        model = resnet50_EE(pretrained=args.pretrained, cize=args.cize, r=args.r, w=args.w,
                                         with_gf=args.gf, low=args.low, high=args.high, alpha=args.alpha,
                                         sigma=args.sigma)
        print('r:{},w:{},gf:{},low:{},high:{}'.format(args.r, args.w, args.gf, args.low,args.high))
    elif args.arch == 'resnet101_EE':
        model = resnet101_EE(pretrained=args.pretrained, cize=args.cize, r=args.r, w=args.w,
                                         with_gf=args.gf, low=args.low, high=args.high, alpha=args.alpha,
                                         sigma=args.sigma)
        print('r:{},w:{},gf:{},low:{},high:{}'.format(args.r, args.w, args.gf, args.low,args.high))
    elif args.arch == 'resnet152_EE':
        model = resnet152_EE(pretrained=args.pretrained, cize=args.cize, r=args.r, w=args.w,
                                         with_gf=args.gf, low=args.low, high=args.high, alpha=args.alpha,
                                         sigma=args.sigma)
        print('r:{},w:{},gf:{},low:{},high:{}'.format(args.r, args.w, args.gf, args.low,args.high))
    else:
        raise NotImplementedError
    model = model.to(device)

    if str(device) == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = False  # if benchmark=True, deterministic will be False
        cudnn.deterministic = True

    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    # model.cuda()
    model.eval()
    
    # Data loading
    train_loader, test_loader = data_loader_tiny_imagenet(args.data, args.batch_size, args.workers, args.pin_memory)

    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # load attack    
    from autoattack import AutoAttack
    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=args.log_path,
        version=args.version)
    
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)

    # example of custom version
    if args.version == 'custom':
        adversary.attacks_to_run = ['apgd-ce', 'fab']
        adversary.apgd.n_restarts = 2
        adversary.fab.n_restarts = 2
    
    # #specify a attack square
    # adversary.attacks_to_run = ['square']


    # run attack and save images
    with torch.no_grad():
        if not args.individual:
            # adv_complete = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],
            #     bs=args.batch_size)
            adv_complete = adversary.run_standard_evaluation(x_test, y_test,
                bs=args.batch_size)
            dict_adv = adversary.run_standard_evaluation_individual(x_test, y_test, bs=args.batch_size)

        else:
            # individual version, each attack is run on all test points
            adv_complete = adversary.run_standard_evaluation_individual(x_test[:args.n_ex],
                y_test[:args.n_ex], bs=args.batch_size)
            
            torch.save(adv_complete, '{}/{}_{}_individual_1_{}_eps_{:.5f}_plus_{}_cheap_{}.pth'.format(
                args.save_dir, 'aa', args.version, args.n_ex, args.epsilon))
                

  