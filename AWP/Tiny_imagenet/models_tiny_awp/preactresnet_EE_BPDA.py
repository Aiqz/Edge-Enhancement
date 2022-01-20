'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../../..")
from utils.core import HighFreqSuppress, CannyFilter_BPDA, get_gaussian_kernel


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet_EE_BPDA(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, dataset="Tiny-ImageNet", cize= 224, r=16, w=0.5, with_gf=False, low=60.0, high=120.0, alpha=0.0, sigma=1):
        super(PreActResNet_EE_BPDA, self).__init__()

        self.in_planes = 64
        self.dataset = dataset
        self.w = w
        self.with_gf = with_gf

        self.hfs = HighFreqSuppress(cize, cize, r)
        self.canny = CannyFilter_BPDA(sigma=sigma, use_cuda=True, alpha=alpha)

        self.low = low / 255
        self.high = high / 255

        k_gaussian = 3
        gaussian_2D = get_gaussian_kernel(k_gaussian, 0., 1.)
        gaussian_2D_torch = torch.from_numpy(gaussian_2D).unsqueeze(0).unsqueeze(0).type(torch.float)
        self.weight_gaussian = nn.Parameter(data=gaussian_2D_torch, requires_grad=False)


        if dataset == "CIFAR10":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            num_classes = 10
        elif dataset == "CIFAR100":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            num_classes = 100
        elif dataset == "ImageNet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            num_classes = 1000
        elif dataset == "Tiny-ImageNet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            num_classes = 200

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)

        if dataset == "CIFAR10" or dataset == "CIFAR100":
            self.avgpool = nn.AvgPool2d(4, stride=1)
            self.linear = nn.Linear(512*block.expansion, num_classes)
        elif dataset == "ImageNet":
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        elif dataset == "Tiny-ImageNet":
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(
        #             m.weight, mode='fan_out', nonlinearity='relu')
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Remove high frequency
        x_hfs = self.hfs(x)

        # Canny
        x_canny = self.canny(x, low_threshold=self.low, high_threshold=self.high, hysteresis=True)
        # x = x_canny.type(torch.float)

        if self.with_gf:
            x_canny = F.conv2d(x_canny.type(torch.float), self.weight_gaussian, padding=1)
            x = x_hfs + self.w * x_canny
        else:
            x = x_hfs + self.w * x_canny

        x = torch.clamp(x, 0.0, 1.0)

        out = self.conv1(x)
        if self.dataset == "ImageNet":
            out = self.bn1(out)
            out = self.relu(out)
            out = self.maxpool(out)
        if self.dataset == "Tiny-ImageNet":
            out = self.bn1(out)
            out = self.relu(out)
            out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # Why?????
        out = F.relu(self.bn(out))
        
        out = self.avgpool(out)

        out = out.view(out.size(0), -1)
        if self.dataset == "CIFAR10" or self.dataset == "CIFAR100":
            out = self.linear(out)
        elif self.dataset == "ImageNet":
            out = self.fc(out)
        elif self.dataset == "Tiny-ImageNet":
            out = self.fc(out)
        return out


def PreActResNet18_EE_BPDA(dataset="CIFAR10", **kwargs):
    return PreActResNet_EE_BPDA(PreActBlock, [2, 2, 2, 2], dataset=dataset, **kwargs)


def PreActResNet34_EE_BPDA(dataset="CIFAR10", **kwargs):
    return PreActResNet_EE_BPDA(PreActBlock, [3, 4, 6, 3], dataset=dataset, **kwargs)


def PreActResNet50_EE_BPDA(dataset="CIFAR10", **kwargs):
    return PreActResNet_EE_BPDA(PreActBottleneck, [3, 4, 6, 3], dataset=dataset, **kwargs)


def PreActResNet101_EE_BPDA(dataset="CIFAR10", **kwargs):
    return PreActResNet_EE_BPDA(PreActBottleneck, [3, 4, 23, 3], dataset=dataset, **kwargs)


def PreActResNet152_EE_BPDA(dataset="CIFAR10", **kwargs):
    return PreActResNet_EE_BPDA(PreActBottleneck, [3, 8, 36, 3], dataset=dataset, **kwargs)


def test():
    net = PreActResNet18_EE_BPDA(dataset="Tiny-ImageNet", cize=64, r=16, w=0.5, with_gf=False, low=60.0, high=120.0, alpha=0.0, sigma=1).cuda()
    # print(net)
    y = net((torch.zeros(10, 3, 64, 64).cuda()))
    print(y.shape)


# test()
