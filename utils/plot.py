#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File    :   plot.py
    @Time    :   2021/09/13 10:55:11
    @Author  :   Qingzhong Ai 
    @Contact :   aqz1995@163.com
    @Desc    :   None
'''
from __future__ import print_function

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.switch_backend('agg')
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, transforms
import os
import sys
sys.path.append("..")
import argparse
from ImageNet.models_imagenet import *
import cv2
from data_loader import data_loader_imagenet, data_loader
from plot_core import  Candy_transfrom, generateDataWithDifferentFrequencies_3Channel
from core import HighFreqSuppress, CannyFilter
from read_log import read_loss, read_training_accuracy
# from managpu import GpuManager
# my_gpu = GpuManager()
# using_gpu = my_gpu.set_by_memory(1)
# print("Using GPU: ", using_gpu)

model_names = [
    'alexnet', 'squeezenet1_0', 'squeezenet1_1', 'densenet121',
    'densenet169', 'densenet201', 'densenet201', 'densenet161',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152'
]

parser = argparse.ArgumentParser(description='PyTorch Tiny-ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/home/ubuntu/datasets/seqres',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: alexnet)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='Weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--pin-memory', default=True, dest='pin_memory', action='store_true',
                    help='use pin memory')
parser.add_argument('-p', '--pretrained', default=False, dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--print-freq', '-f', default=50, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoitn, (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--epsilon', type=float, default = 16.0 / 255,
                        help='perturbation')
parser.add_argument('--num-steps-1', type=int, default = 10,
                    help='perturb number of steps')
parser.add_argument('--step-size-1',type=float, default = 2.0 / 255,
                    help='perturb step size')
parser.add_argument('--num-steps-2', type=int, default = 20,
                    help='perturb number of steps')
parser.add_argument('--step-size-2', type=float, default = 1.0 / 255,
                    help='perturb step size')
parser.add_argument('--num-steps-3', type=int, default = 40,
                    help='perturb number of steps')
parser.add_argument('--step-size-3', type=float, default = 1.0 / 255,
                    help='perturb step size')

parser.add_argument('--random',default = True,
                    help='random initialization for PGD')

parser.add_argument('-r', default= 16, type=int, help='radius of our supression module')
best_prec1 = 0.0


def main():
    # re-plot Figure 1
    # read jpg image
    img_bird = cv2.imread('/home/ubuntu/code/Freq_Contour/vizs/imagenet/img_add_2.jpg')
    img_bird = cv2.cvtColor(img_bird, cv2.COLOR_BGR2RGB)
    img_dog = cv2.imread('/home/ubuntu/code/Freq_Contour/vizs/imagenet/img_add_7.jpg')
    img_dog = cv2.cvtColor(img_dog, cv2.COLOR_BGR2RGB)
    img_house = cv2.imread('/home/ubuntu/code/Freq_Contour/vizs/imagenet/img_add_1.jpg')
    img_house = cv2.cvtColor(img_house, cv2.COLOR_BGR2RGB)
    
    img_0 = cv2.imread('/home/ubuntu/code/Freq_Contour/vizs/imagenet/img_add.jpg')
    # img_0 = cv2.cvtColor(img_0[2:226,906:1130,:], cv2.COLOR_BGR2RGB)
    img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)
    # print(img_0.shape)
    img_3 = cv2.imread('/home/ubuntu/code/Freq_Contour/vizs/imagenet/img_add_3.jpg')
    img_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2RGB)
    img_4 = cv2.imread('/home/ubuntu/code/Freq_Contour/vizs/imagenet/img_add_4.jpg')
    img_4 = cv2.cvtColor(img_4, cv2.COLOR_BGR2RGB)
    img_5 = cv2.imread('/home/ubuntu/code/Freq_Contour/vizs/imagenet/img_add_5.jpg')
    img_5 = cv2.cvtColor(img_5, cv2.COLOR_BGR2RGB)
    img_6 = cv2.imread('/home/ubuntu/code/Freq_Contour/vizs/imagenet/img_add_6.jpg')
    img_6 = cv2.cvtColor(img_6, cv2.COLOR_BGR2RGB)
    img_8 = cv2.imread('/home/ubuntu/code/Freq_Contour/vizs/imagenet/img_add_8.jpg')
    img_8 = cv2.cvtColor(img_8, cv2.COLOR_BGR2RGB)


    img_bird = torch.from_numpy(img_bird) / 255.0
    img_dog = torch.from_numpy(img_dog) / 255.0
    img_house = torch.from_numpy(img_house) / 255.0

    img_0 = torch.from_numpy(img_0) / 255.0
    img_3 = torch.from_numpy(img_3) / 255.0
    img_4 = torch.from_numpy(img_4) / 255.0
    img_5 = torch.from_numpy(img_5) / 255.0
    img_6 = torch.from_numpy(img_6) / 255.0
    img_8 = torch.from_numpy(img_8) / 255.0


    img_origin_all = torch.cat((img_0.unsqueeze(0), img_3.unsqueeze(0), img_4.unsqueeze(0), img_5.unsqueeze(0), img_6.unsqueeze(0),img_8.unsqueeze(0)), 0).type(torch.float).permute(0,3,1,2)
    print(img_origin_all.shape)

    torchvision.utils.save_image(img_origin_all, './vizs/add_supp.pdf', nrow=1)
    torchvision.utils.save_image(img_origin_all, './vizs/add_supp.jpg', nrow=1)


        # args = parser.parse_args()
    # torch.manual_seed(args.seed)

    # use_cuda = not args.no_cuda and torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # print('==> Preparing data..')
    # train_loader, val_loader = data_loader_imagenet(args.data, args.batch_size, args.workers, args.pin_memory)

    # for idx, (input, target) in enumerate(train_loader):

    #     if idx == 1: 
    #         break

    #     index = 53
    #     index_img = 3

    #     img_numpy = input.numpy()
    #     img_numpy = np.transpose(img_numpy, (0, 2, 3, 1))

    #     img_fft_low, img_fft_high = generateDataWithDifferentFrequencies_3Channel(img_numpy, args.r)

    #     img_fft_low = img_fft_low.astype(img_numpy.dtype)
    #     img_fft_high = img_fft_high.astype(img_numpy.dtype)

    #     img_high = np.transpose(img_fft_high, (0, 3, 1, 2))
    #     img_high = torch.from_numpy(img_high)
    #     img_high = img_high.type(torch.FloatTensor).to(device)
        

    #     target = target.to(device)
    #     input = input.to(device)
    #     hfs = HighFreqSuppress(224, 224, args.r)
    #     canny = CannyFilter(use_cuda=True).to(device)
        
    #     img_low = hfs(input)
    #     img_canny = canny(input, low_threshold=38.0 / 255, high_threshold=76.0 / 255, hysteresis=True)
    #     # print(img_canny[4,:,:,:])
    #     # print(torch.max(img_low))
    #     # print(img_low.shape)
    #     # print(img_canny.shape)
    #     img_add = img_low + img_canny

    #     # torchvision.utils.save_image(input[:,:,:,:], './vizs/origin.jpg')
    #     # torchvision.utils.save_image(img_canny[:,:,:,:], './vizs/img_canny.jpg')
    #     # torchvision.utils.save_image(img_low[:,:,:,:], './vizs/img_low_freq.jpg')
    #     # torchvision.utils.save_image(img_add[:,:,:,:], './vizs/img_add.jpg')

    #     print(input.shape)
    #     print(img_canny.shape)
    #     print(img_high.shape)
    #     print(img_low.shape)
    #     print(img_add.shape)
    #     img_canny = torch.cat((img_canny, img_canny, img_canny), 1)
    #     print(img_canny.shape)
    #     img_all = torch.cat((input[index,:,:,:].unsqueeze(0), img_high[index,:,:,:].unsqueeze(0), img_low[index,:,:,:].unsqueeze(0), img_canny[index,:,:,:].unsqueeze(0), img_add[index,:,:,:].unsqueeze(0)), 0)
    #     print(img_all.shape)
    #     torchvision.utils.save_image(input[:,:,:,:], './vizs/origin.jpg')
    #     # torchvision.utils.save_image(img_canny[index,:,:,:], './vizs/img_canny_{}.pdf'.format(index_img))
    #     # torchvision.utils.save_image(img_high[index,:,:,:], './vizs/img_high_freq_{}.pdf'.format(index_img))
    #     # torchvision.utils.save_image(img_low[index,:,:,:], './vizs/img_low_freq_{}.pdf'.format(index_img))
    #     # torchvision.utils.save_image(img_add[index,:,:,:], './vizs/img_add_{}.pdf'.format(index_img))
    #     torchvision.utils.save_image(img_all, './vizs/img_all_{}.pdf'.format(index_img))

    #     print("Done!")

def plot_loss_tarAT():

    # Read losses for log
    # tarAT
    log_path_tarAT = '/hdd/helirong/Freq_Contour/Logs_imagenet/tarAT_unify_ddp/resnet18/nopre/'
    fname_tarAT = '0623_tarat_numstep10_eps16_r16_canny_sigma1_alpha0-bs256-lr0.1-wd1e-4-w1-gfF-l38-h76-ty1.log'
    losses_tarAT = read_loss(log_path_tarAT, fname_tarAT)
    acc_tarAT = read_training_accuracy(log_path_tarAT, fname_tarAT)

    # tarAT_hfs_canny
    log_path_tarAT_hfs_canny = '/hdd/helirong/Freq_Contour/Logs_imagenet/tarAT_unify_ddp/resnet18_hfs_canny/nopre/'
    fname_tarAT_hfs_canny = '0622_tarat_numstep10_eps16_r16_canny_sigma1_alpha0-bs256-lr0.1-wd1e-4-w1-gfF-l38-h76-ty1.log'
    losses_tarAT_hfs_canny = read_loss(log_path_tarAT_hfs_canny, fname_tarAT_hfs_canny)
    acc_tarAT_hfs_canny = read_training_accuracy(log_path_tarAT_hfs_canny, fname_tarAT_hfs_canny)

    # tarAT_fd
    log_path_tarAT_fd = '/hdd/helirong/Freq_Contour/Logs_imagenet/tarAT_fd/resnet18_fd/nopre/'
    fname_tarAT_fd = 'log.txt'
    losses_tarAT_fd = read_loss(log_path_tarAT_fd, fname_tarAT_fd)
    acc_tarAT_fd = read_training_accuracy(log_path_tarAT_fd, fname_tarAT_fd)

    # # tarAT + trick
    # log_path_tarAT_fd_trick = '/hdd/helirong/Freq_Contour/Logs_imagenet/tarAT_unify_ddp_trick/resnet18_hfs_canny/nopre/'
    # fname_tarAT_fd_trick = '0719_tarat_numstep10_eps16_r16_canny_sigma1_alpha0-bs256-lr0.1-wd1e-4-w1-gfF-l38-h76-ty1.log'
    # losses_tarAT_fd_trick = read_loss(log_path_tarAT_fd_trick, fname_tarAT_fd_trick)

    epochs = np.arange(1, len(losses_tarAT) + 1)
    losses_tarAT = np.array(losses_tarAT)
    losses_tarAT_hfs_canny = np.array(losses_tarAT_hfs_canny)
    losses_tarAT_fd = np.array(losses_tarAT_fd)
    # losses_tarAT_fd_trick = np.array(losses_tarAT_fd_trick)

    acc_tarAT = np.array(acc_tarAT)
    acc_tarAT_hfs_canny = np.array(acc_tarAT_hfs_canny)
    acc_tarAT_fd = np.array(acc_tarAT_fd)

    plt.plot(epochs, losses_tarAT, color='b', linewidth=3, label="tar-AT")
    plt.plot(epochs, losses_tarAT_fd, color='gold', linewidth=3, label="Feature Denoising")
    plt.plot(epochs, losses_tarAT_hfs_canny, color='r', linewidth=3, label="EE (tar-AT)")

    # plt.plot(epochs, acc_tarAT, color='b', linewidth=3, label="tar-AT")
    # plt.plot(epochs, acc_tarAT_fd, color='gold', linewidth=3, label="Feature Denoising")
    # plt.plot(epochs, acc_tarAT_hfs_canny, color='r', linewidth=3, label="EE (tar-AT)")

    plt.xlim(1, 90)
    x_tricks = np.arange(0, 91, 15)
    plt.tick_params(direction='in', labelsize='large')
    plt.xticks(x_tricks)

    plt.grid()
    plt.legend(fontsize=14)
    # plt.title('Training Loss on Imagenet')
    plt.xlabel('Epochs', fontsize=18)
    # plt.ylabel('Training Robust Accuracy', fontsize=18)
    plt.ylabel('Training Loss', fontsize=18)


    plt.savefig("./vizs/plot_losses/loss_tarAT.pdf")
    # plt.savefig("./vizs/plot_losses/acc_tarAT.pdf")
    # plt.show()


def plot_loss_AT():

    # Read losses for log
    # AT
    log_path_AT = '/hdd/helirong/Freq_Contour/Logs_imagenet/AT_unify/resnet18/nopre/'
    fname_AT = 'log.txt'
    losses_AT = read_loss(log_path_AT, fname_AT)

    # AT_hfs_canny
    log_path_AT_hfs_canny = '/hdd/helirong/Freq_Contour/Logs_imagenet/AT_unify/resnet18_hfs_canny/nopre/'
    fname_AT_hfs_canny = 'log.txt'
    losses_AT_hfs_canny = read_loss(log_path_AT_hfs_canny, fname_AT_hfs_canny)

    epochs = np.arange(1, len(losses_AT) + 1)
    losses_AT = np.array(losses_AT)
    losses_AT_hfs_canny = np.array(losses_AT_hfs_canny)

    plt.plot(epochs, losses_AT, color='b', linewidth=3, label="AT")
    plt.plot(epochs, losses_AT_hfs_canny, color='r', linewidth=3, label="Ours(AT)")

    plt.xlim(1, 90)
    x_tricks = np.arange(0, 91, 15)
    plt.tick_params(direction='in')
    plt.xticks(x_tricks)

    plt.grid()
    plt.legend()
    # plt.title('Training Loss on Imagenet')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.savefig("./vizs/plot_losses/loss_AT.jpg")


def plot_3D():

    # 构造需要显示的值
    # X = np.arange(8, -1, step=-1) # X轴的坐标
    # Y = np.arange(0, 5, step=1) # Y轴的坐标
    w = np.array([0, 0.25, 0.5, 0.75, 1])
    r = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32])
    r = np.flip(r)

    #设置每一个（X，Y）坐标所对应的Z轴的值，在这边Z（X，Y）=X+Y
    acc_adv = [
        [0.500, 13.570, 17.730, 19.480, 20.400, 21.370, 22.360, 22.420, 24.030],
        [16.990, 22.810, 24.290, 23.780, 23.380, 23.030, 23.250, 23.530, 23.890],
        [16.260, 24.060, 25.390, 25.460, 24.460, 23.700, 23.130, 23.410, 23.920],
        [16.100, 26.120, 26.760, 25.430, 25.330, 24.340, 24.060, 23.810, 24.560],
        [16.260, 27.910, 28.150, 27.120, 26.010, 24.010, 24.650, 25.100, 25.990]
    ]
    acc_clean = [
        [0.500, 20.460, 27.270, 30.340, 31.820, 33.100, 34.210, 35.220, 36.540],
        [22.600, 32.580, 34.990, 35.520, 34.540, 34.680, 34.000, 34.950, 35.980],
        [22.520, 32.940, 35.480, 35.700, 35.410, 34.990, 34.780, 34.810, 35.400],
        [21.780, 33.720, 35.950, 35.680, 35.900, 35.500, 35.140, 35.290, 35.290],
        [22.320, 33.590, 36.020, 36.330, 36.010, 35.500, 35.430, 36.050, 36.530]
    ]
    acc_adv = np.array(acc_adv)
    acc_clean = np.array(acc_clean)
    # acc_adv = np.flip(acc_adv, 0)
    # acc_clean = np.flip(acc_clean, 1)
    # Z=np.zeros(shape=(5, 9))
    # for i in range(5):
    #     for j in range(9):
    #         Z[i, j]= 5
    
    xx, yy = np.meshgrid(r, w) #网格化坐标
    X, Y = xx.ravel(), yy.ravel() #矩阵扁平化
    bottom = np.zeros_like(X) #设置柱状图的底端位值
    # Z = acc_clean.ravel()#扁平化矩阵
    Z = acc_clean.ravel()
    
    width = 3.8 #每一个柱子的长和宽
    height = 0.18
    
    #绘图设置
    fig = plt.figure()
    ax = fig.gca(projection='3d')#三维坐标轴
    # colors = 
    # ax.bar3d(X, Y, bottom, width, height, Z, shade=True, color='r')
    # ax.bar3d(X[10:15], Y[10:15], bottom[10:15], width, height, Z[10:15], shade=True, color='gold')
    # ax.bar3d(X[15:20], Y[15:20], bottom[15:20], width, height, Z[0:5], shade=True, color='Cyan')

    ax.bar3d(X[0:9], Y[0:9], bottom[0:9], width, height, Z[0:9], shade=True, color='skyblue')
    ax.bar3d(X[9:18], Y[9:18], bottom[9:18], width, height, Z[9:18], shade=True, color='deepskyblue')
    ax.bar3d(X[18:27], Y[18:27], bottom[18:27], width, height, Z[18:27], shade=True, color='gold')
    ax.bar3d(X[27:36], Y[27:36], bottom[27:36], width, height, Z[27:36], shade=True, color='Cyan')
    ax.bar3d(X[36:45], Y[36:45], bottom[36:45], width, height, Z[36:45], shade=True, color='lightgreen')

    # ax.bar3d(X[0:5], Y[0:5], bottom[0:5], width, height, Z[0:5], shade=True, color='r')
    # ax.bar3d(X[5:10], Y[5:10], bottom[5:10], width, height, Z[5:10], shade=True, color='r')
    # ax.bar3d(X[10:15], Y[10:15], bottom[10:15], width, height, Z[10:15], shade=True, color='gold')
    # ax.bar3d(X[15:20], Y[15:20], bottom[15:20], width, height, Z[15:20], shade=True, color='Cyan')
    # ax.bar3d(X[20:25], Y[20:25], bottom[20:25], width, height, Z[20:25], shade=True, color='Yellow')
    # ax.bar3d(X[25:30], Y[25:30], bottom[25:30], width, height, Z[25:30], shade=True, color='skyblue')
    # ax.bar3d(X[30:35], Y[30:35], bottom[30:35], width, height, Z[30:35], shade=True, color='b')
    # ax.bar3d(X[35:40], Y[35:40], bottom[35:40], width, height, Z[35:40], shade=True, color='deepskyblue')
    
    # tmp_planes = ax.zaxis._PLANES 
    # ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
    #                     tmp_planes[0], tmp_planes[1], 
    #                     tmp_planes[4], tmp_planes[5])
    # view_1 = (25, -135)
    # view_2 = (25, -45)
    # init_view = view_1
    # ax.view_init(*init_view)

    #坐标轴设置
    # ax.set_xtricks([])

    ax.set_xlabel('r')
    ax.set_ylabel('$\lambda$')
    ax.set_zlabel('Clean Accuracy')
    # ax.set_xlim(0, 32, 4)
    # set(ax,'xticklabel',r,'yticklabel',w)
    ax.set_xticklabels(["32", "28", "24", "20", "16", "12", "8", "4", "0"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    # plt.show()
    ax.tick_params(direction='in')
    plt.savefig('./vizs/test_clean.pdf')


if __name__ == '__main__':
    main()