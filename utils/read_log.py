#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File    :   read_log.py
    @Time    :   2021/09/13 11:00:28
    @Author  :   Qingzhong Ai 
    @Contact :   aqz1995@163.com
    @Desc    :   None
'''
import numpy as np


def read_logs_st(path, fname, last=1):
    f = open(path+fname, 'r')
    lines = f.readlines()
    acc_cle_pre1 = []
    acc_cle_pre5 = []
    for l in lines:
        if l.__contains__("*"):
            s = l.split(" ")
            acc_cle_pre1 += [float(s[3])]
            acc_cle_pre5 += [float(s[5])]

    return np.array(acc_cle_pre1[-last:]), np.array(acc_cle_pre5[-last:])


def read_logs_at(path, fname, last=1):
    f = open(path+fname, 'r')
    lines = f.readlines()
    acc_cle_pre1 = []
    acc_cle_pre5 = []
    acc_adv_pre1 = []
    acc_adv_pre5 = []
    acc_who_pre1 = []
    acc_who_pre5 = []
    for l in lines:
        if l.__contains__("* Clean"):
            # s = l.split("/")[0].split(" ")[-1]
            s = l.split(" ")
            acc_cle_pre1 += [float(s[4])]
            acc_cle_pre5 += [float(s[6])]
        elif l.__contains__("* Adv"):
            tmp = l.split(" ")
            acc_adv_pre1 += [float(tmp[4])]
            acc_adv_pre5 += [float(tmp[6])]

        # elif l.__contains__("* Whole"):
        #     who = l.split(" ")
        #     print(who[4])
        #     print(who[6])
        #     acc_who_pre1 += [float(who[4])]
        #     acc_who_pre5 += [float(who[6])]

    return np.array(acc_cle_pre1[-last:]), np.array(acc_cle_pre5[-last:]), np.array(acc_adv_pre1[-last:]), np.array(
        acc_adv_pre5[-last:])
    # return np.array(acc_cle_pre1[-last:]), np.array(acc_cle_pre5[-last:]), np.array(acc_adv_pre1[-last:]), np.array(
    #     acc_adv_pre5[-last:]), np.array(acc_who_pre1[-last:]), np.array(acc_who_pre5[-last:])


def read_log(path, fname, last=1):
    f = open(path+fname, 'r')
    lines = f.readlines()
    acc_cle_pre1 = []
    acc_cle_pre5 = []
    acc_adv_pre1 = []
    acc_adv_pre5 = []
    acc_who_pre1 = []
    acc_who_pre5 = []
    for l in lines:
        if l.__contains__("loss_clean"):
            # s = l.split("/")[0].split(" ")[-1]
            s = l.split(" ")
            acc_cle_pre1 += [float(s[4])]
            acc_cle_pre5 += [float(s[6])]
        elif l.__contains__("loss_adv"):
            tmp = l.split(" ")
            acc_adv_pre1 += [float(tmp[4])]
            acc_adv_pre5 += [float(tmp[6])]

        # elif l.__contains__("* Whole"):
        #     who = l.split(" ")
        #     print(who[4])
        #     print(who[6])
        #     acc_who_pre1 += [float(who[4])]
        #     acc_who_pre5 += [float(who[6])]

    return np.array(acc_cle_pre1[-last:]), np.array(acc_cle_pre5[-last:]), np.array(acc_adv_pre1[-last:]), np.array(
        acc_adv_pre5[-last:])
    # return np.array(acc_cle_pre1[-last:]), np.array(acc_cle_pre5[-last:]), np.array(acc_adv_pre1[-last:]), np.array(
    #     acc_adv_pre5[-last:]), np.array(acc_who_pre1[-last:]), np.array(acc_who_pre5[-last:])


def read_loss(path, fname):
    f = open(path + fname, 'r')
    lines = f.readlines()
    losses = []

    for l in lines:
        if l.__contains__("5000/5005"):
            s = l.split(" ")
            losses.append(float(s[7][1:7]))

    return losses


def read_training_accuracy(path, fname):
    f = open(path + fname, 'r')
    lines = f.readlines()
    acc = []

    for l in lines:
        if l.__contains__("5000/5005"):
            s = l.split(" ")
            acc.append(float(s[9][1:6]))

    return acc

    

# tarAT
# path = '/hdd/helirong/Freq_Contour/Logs_imagenet/tarAT_unify_ddp/resnet18/nopre/'
# fname = '0623_tarat_numstep10_eps16_r16_canny_sigma1_alpha0-bs256-lr0.1-wd1e-4-w1-gfF-l38-h76-ty1.log'
# # tarAT_hfs_canny
# # path = '/hdd/helirong/Freq_Contour/Logs_imagenet/tarAT_unify_ddp/resnet18_hfs_canny/nopre/0622_tarat_numstep10_eps16_r16_canny_sigma1_alpha0-bs256-lr0.1-wd1e-4-w1-gfF-l38-h76-ty1.log'
# # path = '/hdd/helirong/Freq_Contour/Logs_imagenet/tarAT_unify_ddp/resnet18_hfs_canny/nopre/0705_tarat_numstep10_eps16_r16_canny_sigma1_alpha0-bs256-lr0.1-wd1e-4-w1-gfF-l38-h76-ty1.log'

# losses = read_loss(path, fname)
# print(len(losses))
# print(losses)
# acc = read_training_accuracy(path, fname)
# print(acc)
# print(len(acc))

# acc_cle_pre1, acc_cle_pre5, acc_adv_pre1, acc_adv_pre5 = read_logs_at(path, fname, last = 10)

# #
# acc_cle_pre1_mean = np.mean(acc_cle_pre1)
# acc_cle_pre1_cov = np.sqrt(np.var(acc_cle_pre1))

# acc_cle_pre5_mean = np.mean(acc_cle_pre5)
# acc_cle_pre5_cov = np.sqrt(np.var(acc_cle_pre5))

# acc_adv_pre1_mean = np.mean(acc_adv_pre1)
# acc_adv_pre1_cov = np.sqrt(np.var(acc_adv_pre1))

# acc_adv_pre5_mean = np.mean(acc_adv_pre5)
# acc_adv_pre5_cov = np.sqrt(np.var(acc_adv_pre5))


# print('acc_cle_pre1_mean:{:.3f}'.format(acc_cle_pre1_mean))
# print('acc_cle_pre1_cov:{:.3f}'.format(acc_cle_pre1_cov))

# print('acc_cle_pre5_mean:{:.3f}'.format(acc_cle_pre5_mean))
# print('acc_cle_pre5_cov:{:.3f}'.format(acc_cle_pre5_cov))

# print('acc_adv_pre1_mean:{:.3f}'.format(acc_adv_pre1_mean))
# print('acc_adv_pre1_cov:{:.3f}'.format(acc_adv_pre1_cov))

# print('acc_adv_pre5_mean:{:.3f}'.format(acc_adv_pre5_mean))
# print('acc_adv_pre5_cov:{:.3f}'.format(acc_adv_pre5_cov))

