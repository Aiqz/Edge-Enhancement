#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File    :   data_loader.py
    @Time    :   2021/09/09 16:07:15
    @Author  :   Qingzhong Ai 
    @Contact :   aqz1995@163.com
    @Desc    :   None
'''
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


# data_loader for MNIST
def data_loader_mnist(root, batch_size=64, workers=2, pin_memory=True):
    transform_train = transforms.Compose([
        transforms.ToTensor()])
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(root, train=True, download=True, transform=transform_train,)

    test_dataset = datasets.MNIST(root, train=False, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader (test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=pin_memory)

    return train_loader, test_loader
