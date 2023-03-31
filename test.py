'''
Description: 
Author: Xiongjun Guan
Date: 2023-03-01 19:41:05
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-03-31 10:12:03

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import argparse
import logging
import sys
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

# import wandb
from tqdm import tqdm
import numpy as np

from data_loader import get_dataloader_test
from models.FingerNet import FingerNet
from glob import glob
import cv2
from utils import (
    postprocessing,
    draw_minutia_on_finger,
    draw_img_with_orientation,
    mnt_writer_verifinger
)


if __name__ == "__main__":
    pth_dir = ""
    
    ext = 'png'
    img_dir = ""
    save_basedir = ""
    
    cuda_ids = [0]
    save_enh_dir = osp.join(save_basedir, 'enh')
    save_mask_dir = osp.join(save_basedir, 'mask')
    save_ori_dir = osp.join(save_basedir, 'ori')
    save_mnt_dir = osp.join(save_basedir, 'mnt')

    if not osp.exists(save_enh_dir):
        os.makedirs(save_enh_dir)
    if not osp.exists(save_mask_dir):
        os.makedirs(save_mask_dir)
    if not osp.exists(save_ori_dir):
        os.makedirs(save_ori_dir)
    if not osp.exists(save_mnt_dir):
        os.makedirs(save_mnt_dir)

    test_dataloader = get_dataloader_test(
        img_dir=img_dir,
        batch_size=1,
        img_fmt="png",
    )

    model = FingerNet()
    model.load_state_dict(torch.load(pth_dir))
    device = torch.device(
        "cuda:{}".format(str(cuda_ids[0])) if torch.cuda.is_available() else "cpu"
    )
    model = torch.nn.DataParallel(
        model, device_ids=cuda_ids, output_device=cuda_ids[0]
    )
    model = model.to(device)
    
    model.eval()
    with torch.no_grad():
        pbar = tqdm(test_dataloader)
        for batch in pbar:
            input_img = batch["img"].to(device)
            ftitles = batch["ftitle"]
            preds = model(input_img)

            for idx in range(len(ftitles)):
                (
                    img_gt,
                    seg_pred,
                    ori_pred,
                    enh_pred,
                    mnt_pred,
                    mntS_pred,
                ) = postprocessing(
                    input_img[idx : idx + 1, :, :, :].detach(),
                    preds["ori"][idx : idx + 1, :, :, :].detach(),
                    preds["seg"][idx : idx + 1, :, :, :].detach(),
                    preds["enh_real"][idx : idx + 1, :, :, :].detach(),
                    seg_gt=None,
                    mnt_o=preds["mnt_o"][idx : idx + 1, :, :, :].detach(),
                    mnt_w=preds["mnt_w"][idx : idx + 1, :, :, :].detach(),
                    mnt_h=preds["mnt_h"][idx : idx + 1, :, :, :].detach(),
                    mnt_s=preds["mnt_s"][idx : idx + 1, :, :, :].detach(),
                )
                ftitle = ftitles[idx]
                cv2.imwrite(
                    osp.join(save_mask_dir, f"{ftitle}.png"),
                    np.uint8(seg_pred*255),
                )
                draw_img_with_orientation(
                    img_gt,
                    ori_pred,
                    osp.join(save_ori_dir, f"{ftitle}.png"),
                )
                cv2.imwrite(osp.join(save_enh_dir, f"{ftitle}.png"), enh_pred)
                draw_minutia_on_finger(
                    img_gt,
                    mnt_pred,
                    osp.join(save_mnt_dir, f"{ftitle}.png"),
                )
                mnt_writer_verifinger(mnt_pred,osp.join(save_mnt_dir,f'{ftitle}.txt'),img_gt.shape)

            
