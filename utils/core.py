#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._jit_internal import weak_script_method
import cv2
from torch.autograd import Variable


# function to suppress high freqency components
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


def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2D = gaussian_2D / (2 * np.pi *sigma **2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D


def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D


def get_thin_kernels(start=0, end=360, step=45):
        k_thin = 3  # actual size of the directional kernel
        # increase for a while to avoid interpolation when rotating
        k_increased = k_thin + 2

        # get 0° angle directional kernel
        thin_kernel_0 = np.zeros((k_increased, k_increased))
        thin_kernel_0[k_increased // 2, k_increased // 2] = 1
        thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

        # rotate the 0° angle directional kernel to get the other ones
        thin_kernels = []
        for angle in range(start, end, step):
            (h, w) = thin_kernel_0.shape
            # get the center to not rotate around the (0, 0) coord point
            center = (w // 2, h // 2)
            # apply rotation
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

            # get the k=3 kerne
            kernel_angle = kernel_angle_increased[1:-1, 1:-1]
            is_diag = (abs(kernel_angle) == 1)      # because of the interpolation
            kernel_angle = kernel_angle * is_diag   # because of the interpolation
            thin_kernels.append(kernel_angle)
        return thin_kernels


def safeSign(tensor):
    result = torch.sign(tensor)
    result[result==0] = -1
    return result

# Binarized Neural Networks: Training Neural Networks with Weights and Activations Constrained to +1 or -1 [1]
class BinaryConnectDeterministic(torch.autograd.Function):
    """
    Binarizarion deterministic op with backprob.\n
    Forward : \n
    :math:`r_b  = sign(r)`\n
    Backward : \n
    :math:`d r_b/d r = 1_{|r|=<1}`
    """
    @staticmethod
    def forward(ctx, input):
        """
        Apply stochastic binarization on input tensor.
        """
        ctx.save_for_backward(input)
        return safeSign(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute the back propagation of the binarization op.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[torch.abs(input) > 1.001] = 0
        return grad_input


class CannyFilter(nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 use_cuda=False,
                 alpha = 0.0):
        super(CannyFilter, self).__init__()
        # device
        self.device = 'cuda' if use_cuda else 'cpu'
        self.alpha = alpha
        print('CannyFilter; sigma:{}, alpha:{}'.format(sigma, alpha))
        # gaussian
        self.pad_gaussian = nn.ReplicationPad2d(k_gaussian // 2)
        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        gaussian_2D_torch = torch.from_numpy(gaussian_2D).unsqueeze(0).unsqueeze(0).type(torch.float)
        self.weight_gaussian = nn.Parameter(data=gaussian_2D_torch, requires_grad=False)

        # self.gaussian_filter = nn.Conv2d(in_channels=1,
        #                                  out_channels=1,
        #                                  kernel_size=k_gaussian,
        #                                  # padding=k_gaussian // 2,
        #                                  bias=False)
        # self.gaussian_filter.weight[:] = torch.from_numpy(gaussian_2D)

        # sobel
        sobel_2D = get_sobel_kernel(k_sobel)
        self.reflect_pad = nn.ReplicationPad2d(k_sobel // 2)
        sobel_2D_torch_x = torch.from_numpy(sobel_2D).unsqueeze(0).unsqueeze(0).type(torch.float)
        self.weight_sobel_x = nn.Parameter(data=sobel_2D_torch_x, requires_grad=False)

        sobel_2D_torch_y = torch.from_numpy(sobel_2D.T).unsqueeze(0).unsqueeze(0).type(torch.float)
        self.weight_sobel_y = nn.Parameter(data=sobel_2D_torch_y, requires_grad=False)

        # self.sobel_filter_x = nn.Conv2d(in_channels=1,
        #                                 out_channels=1,
        #                                 kernel_size=k_sobel,
        #                                 bias=False)
        # self.sobel_filter_x.weight[:] = torch.from_numpy(sobel_2D)
        # self.sobel_filter_x = F.conv2d()

        # self.sobel_filter_y = nn.Conv2d(in_channels=1,
        #                                 out_channels=1,
        #                                 kernel_size=k_sobel,
        #                                 bias=False)
        # self.sobel_filter_y.weight[:] = torch.from_numpy(sobel_2D.T)

        # thin

        thin_kernels = get_thin_kernels()
        directional_kernels = np.stack(thin_kernels)
        directional_kernels_torch = torch.from_numpy(directional_kernels).unsqueeze(1).type(torch.float)
        self.weight_directional = nn.Parameter(data=directional_kernels_torch, requires_grad=False)
        self.padding_directional = thin_kernels[0].shape[-1] // 2
        # self.directional_filter = nn.Conv2d(in_channels=1,
        #                                     out_channels=8,
        #                                     kernel_size=thin_kernels[0].shape,
        #                                     padding=thin_kernels[0].shape[-1] // 2,
        #                                     bias=False)
        # self.directional_filter.weight[:, 0] = torch.from_numpy(directional_kernels)

        # hysteresis

        hysteresis = np.ones((3, 3)) + 0.25
        hysteresis_torch = torch.from_numpy(hysteresis).unsqueeze(0).unsqueeze(0).type(torch.float)
        self.weight_hysteresis = nn.Parameter(data=hysteresis_torch, requires_grad=False)
        # self.hysteresis = nn.Conv2d(in_channels=1,
        #                             out_channels=1,
        #                             kernel_size=3,
        #                             padding=1,
        #                             bias=False)
        # self.hysteresis.weight[:] = torch.from_numpy(hysteresis)

    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=False):
        # set the setps tensors
        B, C, H, W = img.shape
        blurred = torch.zeros((B, C, H, W)).to(self.device)
        grad_x = torch.zeros((B, 1, H, W)).to(self.device)
        grad_y = torch.zeros((B, 1, H, W)).to(self.device)
        grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device)
        grad_orientation = torch.zeros((B, 1, H, W)).to(self.device)

        # gaussian

        for c in range(C):
            img_1 = img[:, c:c + 1]
            img_pad = self.pad_gaussian(img_1)
            blurred[:, c:c + 1] = F.conv2d(img_pad, self.weight_gaussian)

            # blurred[:, c:c + 1] = self.gaussian_filter(img_pad)
            # blurred[:, c:c+1] = self.gaussian_filter(img[:, c:c+1])

            # paded_blurred = self.reflect_pad(blurred[:, c:c + 1])

            # grad_x= grad_x + F.conv2d(paded_blurred, self.weight_sobel_x)
            # grad_y = grad_y + F.conv2d(paded_blurred, self.weight_sobel_y)
            # # grad_x = grad_x + self.sobel_filter_x(paded_blurred)
            # grad_y = grad_y + self.sobel_filter_y(paded_blurred)
            # grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c+1])
            # grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c+1])

        pad_blurred = self.reflect_pad(blurred)
        grad_x = F.conv2d(pad_blurred, self.weight_sobel_x.repeat(1, C, 1, 1))  ###modify
        grad_y = F.conv2d(pad_blurred, self.weight_sobel_y.repeat(1, C, 1, 1))  ###modify

        # thick edges

        grad_x_1, grad_y_1 = grad_x / C, grad_y / C
        grad_magnitude = (grad_x_1 ** 2 + grad_y_1 ** 2) ** 0.5
        grad_orientation = torch.atan(grad_y_1 / grad_x_1)
        grad_orientation_1 = grad_orientation * (360 / np.pi) + 180  # convert to degree
        grad_orientation_2 = torch.round(grad_orientation_1 / 45) * 45  # keep a split by 45

        # gradient mask new added
        zeros_like = torch.zeros_like(grad_magnitude)
        grad_magnitude = torch.where(grad_magnitude < self.alpha, zeros_like, grad_magnitude)
        # thin edges

        # directional = self.directional_filter(grad_magnitude)
        directional = F.conv2d(grad_magnitude, self.weight_directional, padding=self.padding_directional)
        # get indices of positive and negative directions
        positive_idx = (grad_orientation_2 / 45) % 8
        negative_idx = ((grad_orientation_2 / 45) + 4) % 8
        thin_edges = grad_magnitude.clone()

        # non maximum suppression direction by direction
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (positive_idx == pos_i) * 1
            is_oriented_i_1 = is_oriented_i + (positive_idx == neg_i) * 1
            pos_directional = directional[:, pos_i]
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])

            # get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)

            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i_1) > 0
            thin_edges[to_remove] = 0.0


        # thresholds
        # return thin_edges
        if low_threshold is not None:
            # low = (thin_edges > low_threshold)
            # sign = LBSign_new.apply          ###modify
            # sign = LBSign.apply  ###modify
            sign = BinaryConnectDeterministic.apply
            low = (sign(thin_edges - low_threshold) + 1) / 2


            # ones = torch.ones_like(thin_edges)
            # zeros = torch.zeros_like(thin_edges)
            # low = torch.where(thin_edges > low_threshold, ones, zeros)
            # low = torch.bernoulli(thin_edges)
            # low = torch.gt(thin_edges, low_threshold).float()

            if high_threshold is not None:
                high = (sign(thin_edges - high_threshold) + 1) / 2


                # high = thin_edges > high_threshold
                # get black/gray/white only
                thin_edges = low * 0.5 + high * 0.5

                if hysteresis:
                    # get weaks and check if they are high or not
                    weak = (thin_edges == 0.5) * 1
                    weak_is_high = (F.conv2d(thin_edges, self.weight_hysteresis, padding=1) > 1) * weak
                    thin_edges = high * 1 + weak_is_high * 1

            else:
                thin_edges = low * 1

        return thin_edges
