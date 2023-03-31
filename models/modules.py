'''
Description: sub-modules of RidgeNet
Author: Xiongjun Guan
Date: 2023-03-01 19:45:05
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-03-30 20:28:39

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import numpy as np
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizeModule(nn.Module):
    def __init__(self, m0=0, var0=1, eps=1e-6):
        super(NormalizeModule, self).__init__()
        self.m0 = m0
        self.var0 = var0
        self.eps = eps

    def forward(self, x):
        x_m = x.mean(dim=(1, 2, 3), keepdim=True)
        x_var = x.var(dim=(1, 2, 3), keepdim=True)
        y = (self.var0 * (x - x_m) ** 2 / x_var.clamp_min(self.eps)).sqrt()
        y = torch.where(x > x_m, self.m0 + y, self.m0 - y)
        return y


class ConvBnPRelu(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chn, out_chn, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_chn, eps=0.001, momentum=0.99)
        self.relu = nn.PReLU(out_chn, init=0)

    def forward(self, input):
        y = self.conv(input)
        y = self.bn(y)
        y = self.relu(y)
        return y
    
class ResBlock(nn.Module):
    def __init__(self, chn, kernel_size=3, stride=1, padding=1, dilation=1):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBnPRelu(chn, chn, kernel_size, stride, padding, dilation)
        self.conv2 = ConvBnPRelu(chn, chn, kernel_size, stride, padding, dilation)

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return y + x


def gabor_bank(enh_ksize=25, ori_stride=2, sigma=4.5, Lambda=8, psi=0, gamma=0.5):
    grid_theta, grid_x, grid_y = torch.meshgrid(
        torch.arange(-90, 90, ori_stride, dtype=torch.float32),
        torch.arange(-(enh_ksize // 2), enh_ksize // 2 + 1, dtype=torch.float32),
        torch.arange(-(enh_ksize // 2), enh_ksize // 2 + 1, dtype=torch.float32),
        indexing="ij"
    )
    cos_theta = torch.cos(torch.deg2rad(-grid_theta))
    sin_theta = torch.sin(torch.deg2rad(-grid_theta))

    x_theta = grid_y * sin_theta + grid_x * cos_theta
    y_theta = grid_y * cos_theta - grid_x * sin_theta
    # gabor filters
    sigma_x = sigma
    sigma_y = float(sigma) / gamma
    exp_fn = torch.exp(-0.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2))
    gb_cos = exp_fn * torch.cos(2 * np.pi * x_theta / Lambda + psi)
    gb_sin = exp_fn * torch.sin(2 * np.pi * x_theta / Lambda + psi)

    return gb_cos[:, None], gb_sin[:, None]


def orientation_cycle_gaussian_weight(ang_stride=2, to_tensor=True):
    gaussian_pdf = signal.windows.gaussian(181, 3)
    coord = np.arange(ang_stride / 2, 180, ang_stride)
    delta = np.abs(coord.reshape(1, -1, 1, 1) - coord.reshape(-1, 1, 1, 1))
    delta = np.minimum(delta, 180 - delta) + 90
    if to_tensor:
        return torch.tensor(gaussian_pdf[delta.astype(int)]).float()
    else:
        return gaussian_pdf[delta.astype(int)].astype(np.float32)

def gausslabel(length=180, stride=2):
    gaussian_pdf = signal.gaussian(length+1, 3)
    label = np.reshape(np.arange(int(stride/2), length, stride), [1,1,-1,1])
    y = np.reshape(np.arange(int(stride/2), length, stride), [1,1,1,-1])
    delta = np.array(np.abs(label - y), dtype=int)
    delta = np.minimum(delta, length-delta)+int(length/2)
    return gaussian_pdf[delta]

def orientation_highest_peak(x, ang_stride=2):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    filter_weight = orientation_cycle_gaussian_weight(ang_stride=ang_stride).type_as(x)
    return F.conv2d(x, filter_weight, stride=1, padding=0)

def select_max_orientation(x):
    x = x / torch.max(x, dim=1, keepdim=True).values.clamp_min(1e-8)
    x = torch.where(x > 0.999, x, torch.zeros_like(x))
    x = x / x.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return x


