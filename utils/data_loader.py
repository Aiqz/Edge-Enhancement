#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


# Data loader for MNIST
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


# Data loader for CIFAR100
def data_loader_cifar100(root, batch_size=64, workers=2, pin_memory=True):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.CIFAR100(root, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR100(root, train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=workers,
    )
    return train_loader, test_loader



# Data loader for Tiny ImageNet
def data_loader_tiny_imagenet(root, batch_size=256, workers=4, pin_memory=True):
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.ToTensor(),
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader


# Dataset loader for ImageNet
def data_loader_imagenet_dataset(root):
    # traindir = os.path.join(root, 'ILSVRC2012_img_train')
    # valdir = os.path.join(root, 'ILSVRC2012_img_val')
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    )
    return train_dataset, val_dataset

