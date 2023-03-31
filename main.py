"""
Description: 
Author: Xiongjun Guan
Date: 2023-03-01 19:41:05
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-03-06 10:35:12

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""
import argparse
import logging
import sys
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np

from args import get_args, get_args_weight
from train import train
from data_loader import get_dataloader_train, get_dataloader_valid, get_dataloader_test
from loss_functions.loss import FinalLoss,FinalLoss_Ridge
from models.FingerNet import FingerNet
import datetime


if __name__ == "__main__":
    pretrain_dir = None

    info_dir = ""
    train_info = np.load(osp.join(info_dir, "palm_train.npy"), allow_pickle=True).item()
    valid_info = np.load(osp.join(info_dir, "palm_valid.npy"), allow_pickle=True).item()

    save_basedir = "./checkpoints/palm/"

    args = get_args()
    weights = get_args_weight()

    now = datetime.datetime.now()

    save_dir = osp.join(save_basedir, args.train_mode+now.strftime("-%Y-%m-%d-%H-%M-%S"))
    example_dir = osp.join(save_dir, "example")

    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    if not osp.exists(example_dir):
        os.makedirs(example_dir)

    logging_path = osp.join(save_dir, "info.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        filename=logging_path,
        filemode="w",
    )
    logging.info(f"Using device: {args.cuda_ids}")
    logging.info(f"Epochs:{args.epochs}, Batch size:{args.batch_size}, lr:{args.lr}")
    logging.info("Network: {}".format(args.model))
    logging.info("Optimizer: {}".format(args.optimizer))
    logging.info("Scheduler type: {}".format(args.scheduler_type))
    logging.info(
        "Loss weights: ori_1={}, ori_2={},  seg={},\n\t\t\t\t\tmnt_o={}, mnt_s={},mnt_w={},mnt_h={}".format(
            weights.ori_1,
            weights.ori_2,
            weights.seg,
            weights.mnt_o,
            weights.mnt_s,
            weights.mnt_w,
            weights.mnt_h,
        )
    )
    logging.info("Train mode: {}".format(args.train_mode))

    train_loader = get_dataloader_train(
        img_dirs=train_info["img_dirs"][:5000],
        mask_dirs=train_info["seg_dirs"][:5000],
        ridge_dirs=train_info["ridge_dirs"][:5000],
        mnt_dirs=train_info["mnt_dirs"][:5000],
        ftitle_lst=train_info["ftitles"][:5000],
        batch_size=args.batch_size*len(args.cuda_ids),
        img_fmt="bmp",
        aug_prob=0.5,
    )

    valid_loader = get_dataloader_valid(
        img_dirs=valid_info["img_dirs"],
        mask_dirs=valid_info["seg_dirs"],
        ridge_dirs=valid_info["ridge_dirs"],
        mnt_dirs=valid_info["mnt_dirs"],
        ftitle_lst=valid_info["ftitles"],
        batch_size=args.batch_size*len(args.cuda_ids),
        img_fmt="bmp",
    )

    if args.model == "FingerNet":
        model = FingerNet()  
   
    if pretrain_dir is not None:
        model.load_state_dict(torch.load(pretrain_dir))
        logging.info("Pretrain model: {}".format(pretrain_dir))
    else:
        logging.info("Pretrain model: {}".format("None"))
    
    device = torch.device(
        "cuda:{}".format(str(args.cuda_ids[0])) if torch.cuda.is_available() else "cpu"
    )
    model = torch.nn.DataParallel(
        model, device_ids=args.cuda_ids, output_device=args.cuda_ids[0]
    )
    model = model.to(device)

    if args.train_mode == "full":
        loss = FinalLoss(
            weight_ori_1=weights.ori_1,
            weight_ori_2=weights.ori_2,
            weight_seg=weights.seg,
            weight_mnt_w=weights.mnt_w,
            weight_mnt_h=weights.mnt_h,
            weight_mnt_o=weights.mnt_o,
            weight_mnt_s=weights.mnt_s,
            device=device
        )
    elif args.train_mode == "ridge":
        loss = FinalLoss_Ridge(
            weight_ori_1=weights.ori_1,
            weight_ori_2=weights.ori_2,
            weight_seg=weights.seg,
            device=device
        )

    logging.info("******** begin training ********")
    train(
        net=model,
        loss_func=loss,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        device=device,
        cuda_ids=args.cuda_ids,
        num_epoch=args.epochs,
        lr=args.lr,
        train_mode=args.train_mode,
        save_dir=save_dir,
        example_dir = example_dir,
        save_checkpoint=1,
        optim=args.optimizer,
        scheduler_type=args.scheduler_type,
    )
