'''
Description: basic loss modules
Author: Xiongjun Guan
Date: 2023-03-01 19:41:05
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-03-12 13:32:17

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CoherenceLoss(nn.Module):
    def __init__(self, ang_stride=2, device=None, eps=1e-6):
        super(CoherenceLoss, self).__init__()
        self.eps = eps
        ang_kernel = torch.arange(ang_stride / 2, 180, ang_stride).view(1, -1, 1, 1) / 90.0 * torch.tensor(np.pi)
        self.cos2angle = torch.cos(ang_kernel)
        self.sin2angle = torch.sin(ang_kernel)
        self.coh_kernel = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).view(1, 1, 3, 3) / 9.0
        if device is not None:
            self.cos2angle = self.cos2angle.to(device)
            self.sin2angle = self.sin2angle.to(device)
            self.coh_kernel = self.coh_kernel.to(device)

    def ori2angle(self, ori):
        cos2angle_ori = (ori * self.cos2angle).sum(dim=1, keepdim=True)
        sin2angle_ori = (ori * self.sin2angle).sum(dim=1, keepdim=True)
        modulus_ori = (sin2angle_ori ** 2 + cos2angle_ori ** 2).sqrt()
        return sin2angle_ori, cos2angle_ori, modulus_ori

    def forward(self, pt, mask=None):
        sin2angle_ori, cos2angle_ori, modulus_ori = self.ori2angle(pt)
        cos2angle = F.conv2d(cos2angle_ori, self.coh_kernel, padding=1)
        sin2angle = F.conv2d(sin2angle_ori, self.coh_kernel, padding=1)
        modulus = F.conv2d(modulus_ori, self.coh_kernel, padding=1)
        coherence = (sin2angle ** 2 + cos2angle ** 2).sqrt() / modulus.clamp_min(self.eps)
        if mask is not None:
            loss = mask.sum() / (coherence * mask).sum().clamp_min(self.eps) - 1
        else:
            loss = 1 / coherence.mean().clamp_min(self.eps) - 1
        return loss

class SmoothLoss(nn.Module):
    def __init__(self, device=None,eps=1e-6):
        super(SmoothLoss, self).__init__()
        self.eps = eps
        self.smooth_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).view(1, 1, 3, 3) / 8.0
        if device is not None:
            self.smooth_kernel = self.smooth_kernel.to(device)

    def forward(self, input, mask=None):
        # smoothness
        if mask is None:
            loss = F.conv2d(input, self.smooth_kernel.type_as(input).repeat(1,input.shape[1],1,1), padding=1).abs().mean()
        else:
            loss = F.conv2d(input, self.smooth_kernel.type_as(input).repeat(1,input.shape[1],1,1), padding=1)
            loss = (loss * mask).sum() / mask.sum().clamp_min(self.eps)
        return loss

class CrossEntropyLoss():
    def __init__(self, eps=1e-6):
        super(CrossEntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, y_true, y_pred,mask=None):
        y_pred = y_pred.clamp(self.eps, 1 - self.eps)
        # weighted cross entropy loss
        lamb_pos, lamb_neg = 1., 1.
        logloss = lamb_pos*y_true*torch.log(y_pred)+lamb_neg*(1-y_true)*torch.log(1-y_pred)

        if mask is not None:
            logloss = -torch.sum(logloss*mask) / (torch.sum(mask) + self.eps)
        else:
            logloss = -torch.mean(logloss)

        return logloss

class MultiCEFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, eps=1e-6):
        super(MultiCEFocalLoss, self).__init__()
        self.eps = eps
        self.gamma = gamma

    def forward(self, input, target, mask=None,need_softmax=False):
        if need_softmax:
            p = torch.softmax(input, dim=1)
        else:
            p = input.clamp(self.eps, 1.0 - self.eps)

        pt = (target * p).sum(dim=1, keepdim=True)
        log_pt = (target * torch.log(p)).sum(dim=1, keepdim=True)

        loss = -torch.pow(1 - pt, self.gamma) * log_pt

        if mask is not None:
            loss = (loss * mask).sum() / mask.sum().clamp_min(self.eps)
        else:
            loss = loss.mean()
        return loss
    
class BinaryCEFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, eps=1e-6):
        super(BinaryCEFocalLoss, self).__init__()
        self.eps = eps
        self.gamma = gamma

    def forward(self, input, target, mask=None,need_sigmoid=False):
        if need_sigmoid:
            p = torch.sigmoid(input, dim=1)
        else:
            p = input.clamp(self.eps, 1.0 - self.eps)

        p = torch.cat((p, 1-p), 1)
        target = torch.cat((target, 1-target), 1)

        pt = (target * p).sum(dim=1, keepdim=True)
        log_pt = (target * torch.log(p)).sum(dim=1, keepdim=True)

        loss = -torch.pow(1 - pt, self.gamma) * log_pt

        if mask is not None:
            loss = (loss * mask).sum() / mask.sum().clamp_min(self.eps)
        else:
            loss = loss.mean()
        return loss

