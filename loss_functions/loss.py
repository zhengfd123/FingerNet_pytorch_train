"""
Description: losses for different features
Author: Xiongjun Guan
Date: 2023-03-03 15:03:57
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-03-07 10:26:33

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""
import numpy as np
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss_functions.loss_modules import CoherenceLoss, SmoothLoss, MultiCEFocalLoss,BinaryCEFocalLoss

eps = 1e-6


class MultiCEFocalCoherenceLoss(torch.nn.Module):
    def __init__(self, weight=1.0, device=None):
        super(MultiCEFocalCoherenceLoss, self).__init__()
        self.weight = weight
        self.focal_loss = MultiCEFocalLoss(gamma=2, eps=eps)
        self.coh_loss = CoherenceLoss(ang_stride=2, device=device, eps=eps)

    def forward(self, y_true, y_pred):
        # clip
        y_pred = y_pred.clamp(eps, 1 - eps)
        # get ROI
        label_seg = torch.sum(y_true, dim=1, keepdim=True)
        label_seg = (label_seg > 0).float()

        # focal loss
        focal_loss = self.focal_loss(y_pred, y_true, label_seg)
        # coherence loss, nearby ori should be as near as possible
        coh_loss = self.coh_loss(y_pred, label_seg)
        # total loss
        loss = focal_loss + self.weight * coh_loss

        items = {
            "total": loss.item(),
            "focal": focal_loss.item(),
            "coh": coh_loss.item(),
        }
        return loss, items


class MultiCEFocalRoiLoss(torch.nn.Module):
    def __init__(self):
        super(MultiCEFocalRoiLoss, self).__init__()
        self.focal_loss = MultiCEFocalLoss(gamma=2, eps=eps)

    def forward(self, y_true, y_pred):
        # clip
        y_pred = y_pred.clamp(eps, 1 - eps)
        # get ROI
        label_seg = torch.sum(y_true, dim=1, keepdim=True)
        label_seg = (label_seg > 0).float()

        # focal loss
        focal_loss = self.focal_loss(y_pred, y_true, label_seg)

        items = {"total": focal_loss.item()}
        return focal_loss, items


# class MultiCEFocalSmoothRoiLoss(torch.nn.Module):
#     def __init__(self, weight=1.0, device=None):
#         super(MultiCEFocalSmoothRoiLoss, self).__init__()
#         self.weight = weight
#         self.focal_loss = MultiCEFocalLoss(gamma=2, eps=eps)
#         self.smooth_loss = SmoothLoss(device=device)

#     def forward(self, y_true, y_pred):
#         # clip
#         y_pred = y_pred.clamp(eps, 1 - eps)
#         # get ROI
#         label_seg = torch.sum(y_true, dim=1, keepdim=True)
#         label_seg = (label_seg > 0).float()

#         # focal loss
#         focal_loss = self.focal_loss(y_pred, y_true, label_seg)
#         # smooth loss
#         smooth_loss = self.smooth_loss(y_pred, label_seg)
#         # total loss
#         loss = focal_loss + self.weight * smooth_loss
#         items = {
#             "total": loss.item(),
#             "focal": focal_loss.item(),
#             "smooth": smooth_loss.item(),
#         }
#         return loss, items


class BinaryCEFocalSmoothLoss(torch.nn.Module):
    def __init__(self, weight=1.0, device=None):
        super(BinaryCEFocalSmoothLoss, self).__init__()
        self.weight = weight
        self.focal_loss = BinaryCEFocalLoss(gamma=2, eps=eps)
        self.smooth_loss = SmoothLoss(device=device)

    def forward(self, y_true, y_pred):
        # clip
        y_pred = y_pred.clamp(eps, 1 - eps)

        # focal loss
        focal_loss = self.focal_loss(y_pred, y_true)
        # smooth loss
        smooth_loss = self.smooth_loss(y_pred)
        # total loss
        loss = focal_loss + self.weight * smooth_loss

        items = {
            "total": loss.item(),
            "focal": focal_loss.item(),
            "smooth": smooth_loss.item(),
        }
        return loss, items


class MntSLoss(torch.nn.Module):
    def __init__(self):
        super(MntSLoss, self).__init__()

    def forward(self, y_true, y_pred):
        # clip
        y_pred = y_pred.clamp(eps, 1 - eps)
        # get ROI
        label_seg = (y_true != 0).float()
        y_true[y_true < 0.0] = 0.0  # set -1 -> 0
        # weighted cross entropy loss
        total_elements = torch.sum(label_seg) + eps
        lamb_pos, lamb_neg = 10., .5
        logloss = lamb_pos*y_true*torch.log(y_pred)+lamb_neg*(1-y_true)*torch.log(1-y_pred)
        # apply ROI
        logloss = logloss*label_seg
        logloss = -torch.sum(logloss) / total_elements

        items = {
            "total": logloss.item(),
        }
        return logloss, items


class FinalLoss(torch.nn.Module):
    def __init__(
        self,
        weight_ori_1=0.1,
        weight_ori_2=0.1,
        weight_seg=20.0,
        weight_mnt_w=0.5,
        weight_mnt_h=0.5,
        weight_mnt_o=0.5,
        weight_mnt_s=300,
        device=None,
    ):
        super(FinalLoss, self).__init__()
        self.weight_ori_1 = weight_ori_1
        self.weight_ori_2 = weight_ori_2
        self.weight_seg = weight_seg
        self.weight_mnt_w = weight_mnt_w
        self.weight_mnt_h = weight_mnt_h
        self.weight_mnt_o = weight_mnt_o
        self.weight_mnt_s = weight_mnt_s

        self.focal_coh_loss = MultiCEFocalCoherenceLoss(weight=1.0, device=device)
        self.focalROI_loss = MultiCEFocalRoiLoss()
        self.focal_smooth_loss = BinaryCEFocalSmoothLoss(weight=1.0, device=device)
        self.mnt_s_loss = MntSLoss()

    def forward(self, labels, preds):
        loss_ori_1, _ = self.focal_coh_loss(labels["label_ori"], preds["ori"])
        loss_ori_2, _ = self.focalROI_loss(labels["label_ori_o"], preds["ori"])
        loss_seg, _ = self.focal_smooth_loss(labels["label_seg"], preds["seg"])
        loss_mnt_w, _ = self.focalROI_loss(labels["label_mnt_w"], preds["mnt_w"])
        loss_mnt_h, _ = self.focalROI_loss(labels["label_mnt_h"], preds["mnt_h"])
        loss_mnt_o, _ = self.focalROI_loss(labels["label_mnt_o"], preds["mnt_o"])
        loss_mnt_s, _ = self.mnt_s_loss(labels["label_mnt_s"], preds["mnt_s"])

        loss = (
            self.weight_ori_1 * loss_ori_1
            + self.weight_ori_2 * loss_ori_2
            + self.weight_seg * loss_seg
            + self.weight_mnt_w * loss_mnt_w
            + self.weight_mnt_h * loss_mnt_h
            + self.weight_mnt_o * loss_mnt_o
            + self.weight_mnt_s * loss_mnt_s
        )
        items = {
            "total": loss.item(),
            "loss_ori_1": self.weight_ori_1 *loss_ori_1.item(),
            "loss_ori_2": self.weight_ori_2 *loss_ori_2.item(),
            "loss_seg": self.weight_seg *loss_seg.item(),
            "loss_mnt_w": self.weight_mnt_w * loss_mnt_w.item(),
            "loss_mnt_h": self.weight_mnt_h * loss_mnt_h.item(),
            "loss_mnt_o": self.weight_mnt_o *loss_mnt_o.item(),
            "loss_mnt_s": self.weight_mnt_s *loss_mnt_s.item(),
        }

        return loss, items


class FinalLoss_Ridge(torch.nn.Module):
    def __init__(
        self,
        weight_ori_1=0.1,
        weight_ori_2=0.1,
        weight_seg=10.0,
        device=None,
    ):
        super(FinalLoss_Ridge, self).__init__()
        self.weight_ori_1 = weight_ori_1
        self.weight_ori_2 = weight_ori_2
        self.weight_seg = weight_seg

        self.focal_coh_loss = MultiCEFocalCoherenceLoss(weight=1.0, device=device)
        self.focalROI_loss = MultiCEFocalRoiLoss()
        self.focal_smooth_loss = BinaryCEFocalSmoothLoss(weight=1.0, device=device)

    def forward(self, labels, preds):
        loss_ori_1, _ = self.focal_coh_loss(labels["label_ori"], preds["ori"])
        loss_ori_2, _ = self.focalROI_loss(labels["label_ori_o"], preds["ori"])
        loss_seg, _ = self.focal_smooth_loss(labels["label_seg"], preds["seg"])

        loss = (
            self.weight_ori_1 * loss_ori_1
            + self.weight_ori_2 * loss_ori_2
            + self.weight_seg * loss_seg
        )
        items = {
            "total": float(loss.item()),
            "loss_ori_1": self.weight_ori_1 *loss_ori_1.item(),
            "loss_ori_2": self.weight_ori_2 *loss_ori_2.item(),
            "loss_seg": self.weight_seg *loss_seg.item(),
        }

        return loss, items
