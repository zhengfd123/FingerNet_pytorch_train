"""
Description: defination of FingerNet. 
Author: Xiongjun Guan
Date: 2023-03-01 19:41:05
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-03-08 11:36:24

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import os
import sys
import cv2
import os.path as osp
import numpy as np
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio
from models.modules import (
    NormalizeModule,
    ConvBnPRelu,
    gabor_bank,
    orientation_highest_peak,
    select_max_orientation,
)


class FingerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_norm = NormalizeModule(m0=0, var0=1)

        # feature extraction VGG
        self.conv1 = nn.Sequential(ConvBnPRelu(1, 64, 3), ConvBnPRelu(64, 64, 3), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(ConvBnPRelu(64, 128, 3), ConvBnPRelu(128, 128, 3), nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(
            ConvBnPRelu(128, 256, 3), ConvBnPRelu(256, 256, 3), ConvBnPRelu(256, 256, 3), nn.MaxPool2d(2, 2)
        )

        # multi-scale ASPP
        self.conv4_1 = ConvBnPRelu(256, 256, 3, padding=1, dilation=1)
        self.ori1 = nn.Sequential(ConvBnPRelu(256, 128, 1, stride=1, padding=0), nn.Conv2d(128, 90, 1, stride=1, padding=0))
        self.seg1 = nn.Sequential(ConvBnPRelu(256, 128, 1, stride=1, padding=0), nn.Conv2d(128, 1, 1, stride=1, padding=0))

        self.conv4_2 = ConvBnPRelu(256, 256, 3, padding=4, dilation=4)
        self.ori2 = nn.Sequential(ConvBnPRelu(256, 128, 1, stride=1, padding=0), nn.Conv2d(128, 90, 1, stride=1, padding=0))
        self.seg2 = nn.Sequential(ConvBnPRelu(256, 128, 1, stride=1, padding=0), nn.Conv2d(128, 1, 1, stride=1, padding=0))

        self.conv4_3 = ConvBnPRelu(256, 256, 3, padding=8, dilation=8)
        self.ori3 = nn.Sequential(ConvBnPRelu(256, 128, 1, stride=1, padding=0), nn.Conv2d(128, 90, 1, stride=1, padding=0))
        self.seg3 = nn.Sequential(ConvBnPRelu(256, 128, 1, stride=1, padding=0), nn.Conv2d(128, 1, 1, stride=1, padding=0))

        # enhance part
        gabor_cos, gabor_sin = gabor_bank(enh_ksize=25, ori_stride=2, Lambda=8)

        self.enh_img_real = nn.Conv2d(gabor_cos.size(1), gabor_cos.size(0), kernel_size=(25, 25), padding=12)
        self.enh_img_real.weight = nn.Parameter(gabor_cos, requires_grad=False)
        self.enh_img_real.bias = nn.Parameter(torch.zeros(gabor_cos.size(0)), requires_grad=False)

        self.enh_img_imag = nn.Conv2d(gabor_sin.size(1), gabor_sin.size(0), kernel_size=(25, 25), padding=12)
        self.enh_img_imag.weight = nn.Parameter(gabor_sin, requires_grad=False)
        self.enh_img_imag.bias = nn.Parameter(torch.zeros(gabor_sin.size(0)), requires_grad=False)

        # mnt part
        self.mnt_conv1 = nn.Sequential(ConvBnPRelu(2, 64, 9, padding=4), nn.MaxPool2d(2, 2))
        self.mnt_conv2 = nn.Sequential(ConvBnPRelu(64, 128, 5, padding=2), nn.MaxPool2d(2, 2))
        self.mnt_conv3 = nn.Sequential(ConvBnPRelu(128, 256, 3, padding=1), nn.MaxPool2d(2, 2))
        self.mnt_o = nn.Sequential(ConvBnPRelu(256 + 90, 256, 1, padding=0), nn.Conv2d(256, 180, 1, padding=0))
        self.mnt_w = nn.Sequential(ConvBnPRelu(256, 256, 1, padding=0), nn.Conv2d(256, 8, 1, padding=0))
        self.mnt_h = nn.Sequential(ConvBnPRelu(256, 256, 1, padding=0), nn.Conv2d(256, 8, 1, padding=0))
        self.mnt_s = nn.Sequential(ConvBnPRelu(256, 256, 1, padding=0), nn.Conv2d(256, 1, 1, padding=0))

    def forward(self, input):
        img_norm = self.img_norm(input)

        # feature extraction VGG
        conv1 = self.conv1(img_norm)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        # multi-scale ASPP
        conv4_1 = self.conv4_1(conv3)
        ori1 = self.ori1(conv4_1)
        seg1 = self.seg1(conv4_1)

        conv4_2 = self.conv4_2(conv3)
        ori2 = self.ori2(conv4_2)
        seg2 = self.seg2(conv4_2)

        conv4_3 = self.conv4_3(conv3)
        ori3 = self.ori3(conv4_3)
        seg3 = self.seg3(conv4_3)

        ori_out = torch.sigmoid(ori1 + ori2 + ori3)
        seg_out = torch.sigmoid(seg1 + seg2 + seg3)

        # enhance part
        enh_real = self.enh_img_real(input)
        enh_imag = self.enh_img_imag(input)
        ori_peak = orientation_highest_peak(ori_out)
        ori_peak = select_max_orientation(ori_peak)
        # ori_up = F.interpolate(ori_peak, scale_factor=8, mode="nearest")
        ori_up = F.interpolate(ori_peak, size=(enh_real.shape[2],enh_real.shape[3]), mode="nearest")
        seg_round = F.softsign(seg_out)
        # seg_up = F.interpolate(seg_round, scale_factor=8, mode="nearest")
        seg_up = F.interpolate(seg_round, size=(enh_real.shape[2],enh_real.shape[3]), mode="nearest")
        enh_real = (enh_real * ori_up).sum(1, keepdim=True)
        enh_imag = (enh_imag * ori_up).sum(1, keepdim=True)
        enh_img = torch.atan2(enh_imag, enh_real)
        enh_seg_img = torch.cat((enh_img, seg_up), dim=1)

        # mnt part
        mnt_conv1 = self.mnt_conv1(enh_seg_img)
        mnt_conv2 = self.mnt_conv2(mnt_conv1)
        mnt_conv3 = self.mnt_conv3(mnt_conv2)

        mnt_o = torch.sigmoid(self.mnt_o(torch.cat((mnt_conv3, ori_out), dim=1)))
        mnt_w = torch.sigmoid(self.mnt_w(mnt_conv3))
        mnt_h = torch.sigmoid(self.mnt_h(mnt_conv3))
        mnt_s = torch.sigmoid(self.mnt_s(mnt_conv3))

        return {
            "real": enh_real,
            "imag":enh_imag,
            "ori": ori_out,
            "seg": seg_out,
            "mnt_o": mnt_o,
            "mnt_w": mnt_w,
            "mnt_h": mnt_h,
            "mnt_s": mnt_s,
        }