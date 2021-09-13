#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File    :   plot_core.py
    @Time    :   2021/09/13 10:55:52
    @Author  :   Qingzhong Ai 
    @Contact :   aqz1995@163.com
    @Desc    :   None
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from _jit_internal import weak_script_method

import cv2
import numpy as np

def fft(img):
    return np.fft.fft2(img)

def fftshift(img):
    return np.fft.fftshift(fft(img))

def ifft(img):
    return np.fft.ifft2(img)

def ifftshift(img):
    return ifft(np.fft.ifftshift(img))

def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0

def mask_radial(img, r):
    rows, cols = img.shape
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=rows, r=r)
    return mask

def generateDataWithDifferentFrequencies_3Channel(Images, r):
    Images_freq_low = []
    Images_freq_high = []
    mask = mask_radial(np.zeros([Images.shape[1], Images.shape[2]]), r)
    for i in range(Images.shape[0]):
        tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
        for j in range(3):
            fd = fftshift(Images[i, :, :, j])
            fd = fd * mask
            img_low = ifftshift(fd)
            tmp[:,:,j] = np.real(img_low)
        Images_freq_low.append(tmp)
        tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
        for j in range(3):
            fd = fftshift(Images[i, :, :, j])
            fd = fd * (1 - mask)
            img_high = ifftshift(fd)
            tmp[:,:,j] = np.real(img_high)
        Images_freq_high.append(tmp)

    return np.array(Images_freq_low), np.array(Images_freq_high)

def Candy_transfrom(img_in):
    # input img shape: [batch, weight, height, channels]
    img_candy = np.zeros([img_in.shape[0], img_in.shape[1], img_in.shape[2]]).astype(np.uint8)
    for i in range(img_in.shape[0]):
        if img_in.shape[3] == 3:
            # print("zz")
            img_candy[i] = cv2.cvtColor(img_in[i], cv2.COLOR_BGR2GRAY)
            img_candy[i] = cv2.GaussianBlur(img_candy[i], (3, 3), 0)
            img_candy[i] = cv2.Canny(img_candy[i], 100, 200)
        else:
            img_candy[i] = cv2.GaussianBlur(img_in[i], (3, 3), 0)
            img_candy[i] = cv2.Canny(img_candy[i], 100, 200)
    return img_candy



#### A PGD white-box attacker with random target label.
def targeted_PGD(model, args, inputs, labels, num_steps, step_size, device):

    x = inputs.detach()
    #### we consider targeted attacks when evaluating under the white-box settings,
    ### where the targeted class is selected uniformly at random
    label_offset = torch.randint(low=1, high=200, size=labels.shape).to(device)
    target_labels = torch.fmod(labels + label_offset, 200)

    if args.random:
        x = x + torch.zeros_like(x).uniform_(-args.epsilon, args.epsilon)
        x = torch.clamp(x, 0, 1)

    for i in range(num_steps):
        x.requires_grad_()
        with torch.enable_grad():
            logits = model(x)
            loss = F.cross_entropy(logits, target_labels, reduction='sum')
        grad = torch.autograd.grad(loss, [x])[0]
        # x = x.detach() + step_size * torch.sign(grad.detach())  ###wrong
        x = x.detach() - step_size * torch.sign(grad.detach())
        x = torch.min(torch.max(x, inputs - args.epsilon), inputs + args.epsilon)
        x = torch.clamp(x, 0, 1)  # data sclale

    return x, target_labels

def targeted_PGD_1(model, args, inputs, labels, num_steps, step_size, device):

    x = inputs.detach()
    #### we consider targeted attacks when evaluating under the white-box settings,
    ### where the targeted class is selected uniformly at random
    label_offset = torch.randint(low=1, high=200, size=labels.shape).type(torch.LongTensor)
    label_offset = label_offset.to(device)
    target_labels = torch.fmod(labels + label_offset, 200)

    if args.random:
        x = x + torch.zeros_like(x).uniform_(-args.epsilon, args.epsilon)
        x = torch.clamp(x, 0, 1)

    for i in range(num_steps):
        x.requires_grad_()
        with torch.enable_grad():
            logits = model(x)
            loss = F.cross_entropy(logits, target_labels, reduction='sum')
        grad = torch.autograd.grad(loss, [x])[0]
        # x = x.detach() + step_size * torch.sign(grad.detach())  ###wrong
        x = x.detach() - step_size * torch.sign(grad.detach())
        x = torch.min(torch.max(x, inputs - args.epsilon), inputs + args.epsilon)
        x = torch.clamp(x, 0, 1)  # data sclale

    return x, target_labels


def PGD(model, args, inputs, targets, num_steps, step_size):
    x = inputs.detach()

    if args.random:
        x = x + torch.zeros_like(x).uniform_(-args.epsilon, args.epsilon)
        x = torch.clamp(x, 0, 1)

    for i in range(num_steps):
        x.requires_grad_()
        with torch.enable_grad():
            logits = model(x)
            loss = F.cross_entropy(logits, targets, reduction='sum')
        grad = torch.autograd.grad(loss, [x])[0]
        x = x.detach() + step_size * torch.sign(grad.detach())
        x = torch.min(torch.max(x, inputs - args.epsilon), inputs + args.epsilon)
        x = torch.clamp(x, 0, 1)  # data sclale

    return x



def squared_l2_norm(x):
    flattened = x.view(x.shape[0], -1)
    return (flattened ** 2).mean(1)

def l2_norm(x):
    return squared_l2_norm(x).sqrt()

class Trades:
    def __init__(self, step_size=0.003, epsilon=0.047, perturb_steps=5, beta=1.0):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.beta = beta
        self.criterion_kl = nn.KLDivLoss(reduction="batchmean")

    def reset_steps(self, k):
        self.perturb_steps = k

    @weak_script_method
    def PGD_L2(self, model, x_natural, logits):
        model.eval()
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device='cuda').detach()
        prob = F.softmax(logits, dim=-1)

        for _ in range(self.perturb_steps):
            with torch.enable_grad():
                x_adv.requires_grad_()
                loss_kl = self.criterion_kl(F.log_softmax(model(x_adv), dim=1), prob)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0].detach()
            grad /= l2_norm(grad).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + 1e-8
            x_adv = x_adv.detach() + self.step_size * grad

            delta = x_adv - x_natural
            delta_norm = l2_norm(delta)
            cond = delta_norm > self.epsilon
            delta[cond] *= self.epsilon / delta_norm[cond].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            x_adv = x_natural + delta
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv

    @weak_script_method
    def PGD_Linf(self, model, x_natural, logits):
        model.eval()
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device='cuda').detach()
        prob = F.softmax(logits, dim=-1)

        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = self.criterion_kl(F.log_softmax(model(x_adv), dim=1), prob)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0].detach()
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad)
            x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv

    @weak_script_method
    def loss(self, model, logits, x_adv, labels, optimizer):
        model.train()
        optimizer.zero_grad()
        prob = F.softmax(logits, dim=-1)
        loss_natural = F.cross_entropy(logits, labels)
        loss_robust = self.criterion_kl(F.log_softmax(model(x_adv), dim=1), prob)
        loss = loss_natural + self.beta * loss_robust

        return loss


class HighFreqSuppress(torch.nn.Module):
    def __init__(self, w, h, r):
        super(HighFreqSuppress, self).__init__()
        self.w = w
        self.h = h
        self.r = r
        self.templete()

    def templete(self):
        temp = np.zeros((self.w, self.h), "float32")
        cw = self.w // 2
        ch = self.h // 2
        if self.w % 2 == 0:
            dw = self.r
        else:
            dw = self.r + 1

        if self.h % 2 == 0:
            dh = self.r
        else:
            dh = self.r + 1

        temp[cw - self.r:cw + dw, ch - self.r:ch + dh] = 1.0
        temp = np.roll(temp, -cw, axis=0)
        temp = np.roll(temp, -ch, axis=1)
        temp = torch.tensor(temp)
        temp = temp.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        self.temp = temp
        # if torch.cuda.is_available():
        #     self.temp = self.temp.cuda()

    @weak_script_method
    def forward(self, x):
        x_hat = torch.rfft(x, 2, onesided=False)
        x_hat = x_hat * self.temp.cuda()
        y = torch.irfft(x_hat, 2, onesided=False)

        return y

    def extra_repr(self):
        return 'feature_width={}, feature_height={}, radius={}'.format(self.w, self.h, self.r)