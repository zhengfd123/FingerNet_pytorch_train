'''
Description: 
Author: Xiongjun Guan
Date: 2023-03-30 20:22:00
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-03-31 10:09:45

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
from cmath import log
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import numpy as np
from tqdm import tqdm
import logging
import os.path as osp
import cv2
from utils import (
    postprocessing,
    draw_minutia_on_finger,
    draw_img_with_orientation,
)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def train(
    net,
    loss_func,
    train_dataloader,
    valid_dataloader,
    device,
    cuda_ids,
    num_epoch,
    lr,
    train_mode,
    save_dir=None,
    example_dir=None,
    save_checkpoint=10,
    optim="adamW",
    scheduler_type="StepLR",
):
    if valid_dataloader is None:
        valid = False
    else:
        valid = True

    # select optimizer
    if optim == "sgd":
        optimizer = torch.optim.SGD(
            (param for param in net.parameters() if param.requires_grad),
            lr=lr,
            weight_decay=0,
        )
    elif optim == "adam":
        optimizer = torch.optim.Adam(
            (param for param in net.parameters() if param.requires_grad),
            lr=lr,
            weight_decay=1e-4,
        )
    elif optim == "adamW":
        optimizer = torch.optim.AdamW(
            (param for param in net.parameters() if param.requires_grad),
            lr=lr,
            weight_decay=1e-4,
        )

    if scheduler_type == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=np.round(num_epoch), eta_min=lr * 1e-2
        )
    elif scheduler_type == "StepLR":
        scheduler = StepLR(optimizer, 15, 0.1)

    # train
    for epoch in range(num_epoch):
        # train phase
        net.train()
        train_losses = {
            "total": 0,
            "loss_ori_1": 0,
            "loss_ori_2": 0,
            "loss_seg": 0,
            "loss_mnt_o": 0,
            "loss_mnt_s": 0,
            "loss_mnt_h": 0,
            "loss_mnt_w": 0,
        }
        pbar = tqdm(train_dataloader, desc=f"epoch:{epoch}, train")
        for batch in pbar:
            input_img = batch["img"].to(device)
            label_ori = batch["label_ori"].to(device)
            label_ori_o = batch["label_ori_o"].to(device)
            label_seg = batch["label_seg"].to(device)
            label_mnt_w = batch["label_mnt_w"].to(device)
            label_mnt_h = batch["label_mnt_h"].to(device)
            label_mnt_o = batch["label_mnt_o"].to(device)
            label_mnt_s = batch["label_mnt_s"].to(device)

            preds = net(input_img)

            if train_mode == "full":
                Loss, items = loss_func(
                    {
                        "label_ori": label_ori,
                        "label_ori_o": label_ori_o,
                        "label_seg": label_seg,
                        "label_mnt_w": label_mnt_w,
                        "label_mnt_h": label_mnt_h,
                        "label_mnt_o": label_mnt_o,
                        "label_mnt_s": label_mnt_s,
                    },
                    preds,
                )
            elif train_mode == "ridge":
                Loss, items = loss_func(
                    {
                        "label_ori": label_ori,
                        "label_ori_o": label_ori_o,
                        "label_seg": label_seg,
                    },
                    preds,
                )

            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

            klist = items.keys()
            for k in klist:
                train_losses[k] += items[k]

            pbar.set_postfix(**{"final loss": Loss.item()})
            del preds, Loss, items, batch

        pbar.close()

        logging.info(
            "epoch: {}, lr:{:.8f}".format(
                epoch, optimizer.state_dict()["param_groups"][0]["lr"]
            )
        )
        scheduler.step()

        klist = train_losses.keys()
        logging_info = "epoch: {}".format(epoch)
        for k in klist:
            train_losses[k] /= len(train_dataloader)
            logging_info = logging_info + ", {}:{:.2f}".format(k, train_losses[k])
        logging.info(logging_info)

        if save_dir is not None and epoch > save_checkpoint:
            if len(cuda_ids) > 1:
                torch.save(
                    net.module.state_dict(),
                    osp.join(save_dir, f"epoch_{epoch}.pth"),
                )
            else:
                torch.save(
                    net.state_dict(),
                    osp.join(save_dir, f"epoch_{epoch}.pth"),
                )

        if valid is True:
            # valid phase
            net.eval()
            with torch.no_grad():
                valid_losses = {
                    "total": 0,
                    "loss_ori_1": 0,
                    "loss_ori_2": 0,
                    "loss_seg": 0,
                    "loss_mnt_o": 0,
                    "loss_mnt_s": 0,
                    "loss_mnt_h": 0,
                    "loss_mnt_w": 0,
                }
                pbar = tqdm(valid_dataloader, desc=f"epoch:{epoch}, val")

                if example_dir is not None:
                    show = True
                else:
                    show = False
                for batch in pbar:
                    input_img = batch["img"].to(device)
                    label_ori = batch["label_ori"].to(device)
                    label_ori_o = batch["label_ori_o"].to(device)
                    label_seg = batch["label_seg"].to(device)
                    label_mnt_w = batch["label_mnt_w"].to(device)
                    label_mnt_h = batch["label_mnt_h"].to(device)
                    label_mnt_o = batch["label_mnt_o"].to(device)
                    label_mnt_s = batch["label_mnt_s"].to(device)

                    preds = net(input_img)

                    if show is True:
                        show = False
                        for idx in range(1):
                            if train_mode == "full":
                                (
                                    img_gt,
                                    seg_gt,
                                    ori_gt,
                                    _,
                                    mnt_gt,
                                    mntS_gt,
                                ) = postprocessing(
                                    input_img[idx : idx + 1, :, :, :].detach(),
                                    label_ori[idx : idx + 1, :, :, :].detach(),
                                    label_seg[idx : idx + 1, :, :, :].detach(),
                                    enh=None,
                                    seg_gt=None,
                                    mnt_o=label_mnt_o[idx : idx + 1, :, :, :].detach(),
                                    mnt_w=label_mnt_w[idx : idx + 1, :, :, :].detach(),
                                    mnt_h=label_mnt_h[idx : idx + 1, :, :, :].detach(),
                                    mnt_s=label_mnt_s[idx : idx + 1, :, :, :].detach(),
                                )
                                cv2.imwrite(osp.join(example_dir, f"{idx}_img.png"), img_gt)
                                cv2.imwrite(osp.join(example_dir, f"{idx}_mask_gt.png"), np.uint8(seg_gt)*255)
                                draw_img_with_orientation(
                                    img_gt,
                                    ori_gt,
                                    osp.join(example_dir, f"{idx}_ori_gt.png"),
                                )
                                draw_minutia_on_finger(
                                    img_gt,
                                    mnt_gt,
                                    osp.join(example_dir, f"{idx}_mnt_gt.png"),
                                )
                                cv2.imwrite(
                                    osp.join(example_dir, f"{idx}_mntS_gt.png"),
                                    np.uint8((mntS_gt + 1) * 255 / 2),
                                )

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
                                    preds["real"][idx : idx + 1, :, :, :].detach(),
                                    seg_gt=label_seg[idx : idx + 1, :, :, :].detach(),
                                    mnt_o=preds["mnt_o"][idx : idx + 1, :, :, :].detach(),
                                    mnt_w=preds["mnt_w"][idx : idx + 1, :, :, :].detach(),
                                    mnt_h=preds["mnt_h"][idx : idx + 1, :, :, :].detach(),
                                    mnt_s=preds["mnt_s"][idx : idx + 1, :, :, :].detach(),
                                )
                                cv2.imwrite(
                                    osp.join(example_dir, f"{idx}_mask_pred.png"),
                                    np.uint8(seg_pred*255),
                                )
                                draw_img_with_orientation(
                                    img_gt,
                                    ori_pred,
                                    osp.join(example_dir, f"{idx}_ori_pred.png"),
                                )

                                cv2.imwrite(osp.join(example_dir, f"{idx}_enh_pred.png"), enh_pred)
                                draw_minutia_on_finger(
                                    img_gt,
                                    mnt_pred,
                                    osp.join(example_dir, f"{idx}_mnt_pred.png"),
                                )
                                cv2.imwrite(
                                    osp.join(example_dir, f"{idx}_mntS_pred.png"),
                                    np.uint8(mntS_pred * 255),
                                )

                            else:
                                (
                                    img_gt,
                                    seg_gt,
                                    ori_gt,
                                    _,
                                    mnt_gt,
                                    mntS_gt,
                                ) = postprocessing(
                                    input_img[idx : idx + 1, :, :, :].detach(),
                                    label_ori[idx : idx + 1, :, :, :].detach(),
                                    label_seg[idx : idx + 1, :, :, :].detach(),
                                    enh=None,
                                    seg_gt=None,
                                    mnt_o=None,
                                    mnt_w=None,
                                    mnt_h=None,
                                    mnt_s=None,
                                )
                                cv2.imwrite(osp.join(example_dir, f"{idx}_img.png"), img_gt)
                                cv2.imwrite(osp.join(example_dir, f"{idx}_mask_gt.png"), np.uint8(seg_gt*255))
                                draw_img_with_orientation(
                                    img_gt,
                                    ori_gt,
                                    osp.join(example_dir, f"{idx}_ori_gt.png"),
                                )

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
                                    preds["real"][idx : idx + 1, :, :, :].detach(),
                                    seg_gt=label_seg[idx : idx + 1, :, :, :].detach(),
                                    mnt_o=None,
                                    mnt_w=None,
                                    mnt_h=None,
                                    mnt_s=None,
                                )
                                cv2.imwrite(
                                    osp.join(example_dir, f"{idx}_mask_pred.png"),
                                    np.uint8(seg_pred*255),
                                )
                                draw_img_with_orientation(
                                    img_gt,
                                    ori_pred,
                                    osp.join(example_dir, f"{idx}_ori_pred.png"),
                                )

                                cv2.imwrite(osp.join(example_dir, f"{idx}_enh_pred.png"), enh_pred)

                    if train_mode == "full":
                        Loss, items = loss_func(
                            {
                                "label_ori": label_ori,
                                "label_ori_o": label_ori_o,

                                "label_seg": label_seg,
                                "label_mnt_w": label_mnt_w,
                                "label_mnt_h": label_mnt_h,
                                "label_mnt_o": label_mnt_o,
                                "label_mnt_s": label_mnt_s,
                            },
                            preds,
                        )
                    elif train_mode == "ridge":
                        Loss, items = loss_func(
                            {
                                "label_ori": label_ori,
                                "label_ori_o": label_ori_o,
                                "label_seg": label_seg,
                            },
                            preds,
                        )

                    klist = items.keys()
                    for k in klist:
                        valid_losses[k] += items[k]

                    pbar.set_postfix(**{"final loss": Loss.item()})

                    del preds, Loss, items, batch

                pbar.close()
                klist = valid_losses.keys()
                logging_info = "eval : {}".format(epoch)
                for k in klist:
                    valid_losses[k] /= len(valid_dataloader)
                    logging_info = logging_info + ", {}:{:.2f}".format(
                        k, valid_losses[k]
                    )
                logging.info(logging_info)

    return
