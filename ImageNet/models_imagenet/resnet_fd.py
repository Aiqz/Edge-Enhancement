#!/usr/bin/env python
import os
import math
import torch
import torch.nn as nn


__all__ = ['ResNet_fd', 'resnet18_fd', 'resnet34_fd',
           'resnet50_fd', 'resnet101_fd', 'resnet152_fd']

# you need to download the models to ~/.torch/models
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
models_dir = os.path.expanduser('~/.torch/models')
model_name = {
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'resnet50': 'resnet50-19c8e357.pth',
    'resnet101': 'resnet101-5d3b4d8f.pth',
    'resnet152': 'resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class denoising(nn.Module):
    """
    Feature Denoising in "Feature Denoising for Improving Adversarial Robustness".
    Args:
        embed (bool): whether to use embedding on theta & phi
        softmax (bool): whether to use gaussian (softmax) version or the dot-product version.
    """
    def __init__(self, n_in=64, H=56, W=56, embed=True, softmax=True):
        super(denoising, self).__init__()
        self.embed = embed
        self.softmax = softmax
        self.n_in = n_in
        self.H = H
        self.W = W
        self.conv1 = nn.Conv2d(n_in, int(n_in / 2), kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(n_in, int(n_in / 2), kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(n_in, n_in, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(n_in)

        self.my_softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        if self.embed:
            theta = self.conv1(x)
            phi = self.conv2(x)
            g = x
        else:
            theta, phi, g = x, x, x
        if self.n_in > self.H * self.W or self.softmax:
            f = torch.einsum('niab,nicd->nabcd', theta, phi)
            if self.softmax:
                orig_shape = f.shape
                f = torch.reshape(f, (-1, self.H * self.W, self.H * self.W))
                f = f / torch.sqrt(torch.tensor(theta.shape[1], dtype=torch.float32))
                # f = nn.Softmax(f)
                f = self.my_softmax(f)
                f = torch.reshape(f, orig_shape)
            f = torch.einsum('nabcd,nicd->niab', f, g)
        else:
            f = torch.einsum('nihw,njhw->nij', phi, g)
            f = torch.einsum('nij,nihw->njhw', f, theta)
        if not self.softmax:
            f = f / (self.H * self.W)
        f = torch.reshape(f, x.shape).contiguous()
        # 1*1 conv
        f = self.conv3(f)
        f = self.bn(f)
        x = x + f
        return x


class ResNet_fd(nn.Module):

    def __init__(self, block, denoise_block, layers, num_classes=1000):
        super(ResNet_fd, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.denoise1 = denoise_block(n_in=64, H=56, W=56, embed=False, softmax=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.denoise2 = denoise_block(n_in=128, H=28, W=28, embed=False, softmax=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.denoise3 = denoise_block(n_in=256, H=14, W=14, embed=False, softmax=False)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.denoise4 = denoise_block(n_in=512, H=7, W=7, embed=False, softmax=False)
        
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.denoise1(x)
        x = self.layer2(x)
        x = self.denoise2(x)
        x = self.layer3(x)
        x = self.denoise3(x)
        x = self.layer4(x)
        x = self.denoise4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18_fd(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_fd(BasicBlock, denoising, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            os.path.join(models_dir, model_name['resnet18'])))
    return model


def resnet34_fd(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_fd(BasicBlock, denoising, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            os.path.join(models_dir, model_name['resnet34'])))
    return model


def resnet50_fd(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_fd(Bottleneck, denoising, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            os.path.join(models_dir, model_name['resnet50'])))
    return model


def resnet101_fd(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_fd(Bottleneck, denoising, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            os.path.join(models_dir, model_name['resnet101'])))
    return model


def resnet152_fd(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_fd(Bottleneck, denoising, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            os.path.join(models_dir, model_name['resnet152'])))
    return model
