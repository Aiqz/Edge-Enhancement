import sys
sys.path.append("../..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.core import HighFreqSuppress, CannyFilter, get_gaussian_kernel

class Net2_EE(nn.Module):
    def __init__(self, r=8, w=1, with_gf=False, low=60.0, high=120.0, alpha = 0.0, sigma=1):
        super(Net2_EE, self).__init__()

        self.w = w
        self.with_gf = with_gf
        # hfs and canny
        self.hfs = HighFreqSuppress(28, 28, r)
        self.canny = CannyFilter(sigma= sigma, use_cuda=True, alpha=alpha)

        self.low = low / 255
        self.high = high / 255


        k_gaussian = 3
        gaussian_2D = get_gaussian_kernel(k_gaussian, 0., 1.)
        gaussian_2D_torch = torch.from_numpy(gaussian_2D).unsqueeze(0).unsqueeze(0).type(torch.float)
        self.weight_gaussian = nn.Parameter(data=gaussian_2D_torch, requires_grad=False)


        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(4 * 4 * 64, 1024)
        self.fc2 = nn.Linear(1024, 10)



    def forward(self, x):

        # Remove high frequency
        x_hfs = self.hfs(x)
        # Canny
        x_canny = self.canny(x, low_threshold=self.low, high_threshold=self.high, hysteresis=True)
        # x = x_canny.type(torch.float)
        if self.with_gf:
            x_canny = F.conv2d(x_canny.type(torch.float), self.weight_gaussian, padding=1)
            # print("a")
            x = x_hfs + self.w * x_canny
        else:
            x = x_hfs + self.w * x_canny
        x = torch.clamp(x, 0.0, 1.0)

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x