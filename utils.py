'''
Description: 
Author: Xiongjun Guan
Date: 2023-03-30 20:22:00
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-03-31 10:09:33

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import numpy as np
from scipy import io
import torch
import torch.nn as nn
import cv2
from scipy import ndimage, sparse, spatial
from scipy.ndimage import zoom
from tqdm import tqdm
import os
import os.path as osp
from models.modules import orientation_highest_peak
import matplotlib.pyplot as plt

def mnt_writer_verifinger(mnt, file_name, line):
    f = open(file_name, "w")
    f.write("%d\n" % line[0])
    f.write("%d\n" % line[1])
    f.write("0\n0\n%d\n"% mnt.shape[0])
    for i in range(mnt.shape[0]):
        f.write(
            "%d %d %d %d %d\n"
            % (mnt[i, 0], mnt[i, 1], mnt[i, 2], 1, mnt[i, 3] * 100)
        )
    f.close()
    return


def label2mnt(mnt_s_out, mnt_w_out, mnt_h_out, mnt_o_out, thresh=0.5, max_mnt=1000):
    assert (
        len(mnt_s_out.shape) == 2
        and len(mnt_w_out.shape) == 3
        and len(mnt_h_out.shape) == 3
        and len(mnt_o_out.shape) == 3
    )
    # get cls results
    mnt_sparse = sparse.coo_matrix(mnt_s_out > thresh)
    mnt_list = np.array(tuple(zip(mnt_sparse.row, mnt_sparse.col)), dtype=np.int32)
    if mnt_list.shape[0] > max_mnt:
        return np.zeros((0, 4))
    if mnt_list.shape[0] == 0:
        return np.zeros((0, 4))
    # get regression results
    mnt_w_out = np.argmax(mnt_w_out, axis=0)
    mnt_h_out = np.argmax(mnt_h_out, axis=0)
    mnt_o_out = np.argmax(mnt_o_out, axis=0)
    # get final mnt
    mnt_final = np.zeros((len(mnt_list), 4))
    mnt_final[:, 0] = mnt_sparse.col * 8 + mnt_w_out[mnt_list[:, 0], mnt_list[:, 1]]
    mnt_final[:, 1] = mnt_sparse.row * 8 + mnt_h_out[mnt_list[:, 0], mnt_list[:, 1]]
    mnt_final[:, 2] = mnt_o_out[mnt_list[:, 0], mnt_list[:, 1]] * 2 - 89.0
    mnt_final[mnt_final[:, 2] < 0.0, 2] = mnt_final[mnt_final[:, 2] < 0.0, 2] + 360
    mnt_final[:, 3] = mnt_s_out[mnt_list[:, 0], mnt_list[:, 1]]
    return mnt_final


def angle_delta(A, B, max_D=360):
    delta = np.abs(A - B)
    delta = np.minimum(delta, max_D - delta)
    return delta


def distance(y_true, y_pred, max_D=16, max_O=180 / 6):
    D = spatial.distance.cdist(y_true[:, :2], y_pred[:, :2], "euclidean")
    O = spatial.distance.cdist(
        np.reshape(y_true[:, 2], [-1, 1]),
        np.reshape(y_pred[:, 2], [-1, 1]),
        angle_delta,
    )
    return (D <= max_D) * (O <= max_O)


def nms(mnt):
    if mnt.shape[0] == 0:
        return mnt
    # sort score
    mnt_sort = mnt.tolist()
    mnt_sort.sort(key=lambda x: x[3], reverse=True)
    mnt_sort = np.array(mnt_sort)
    # cal distance
    inrange = distance(mnt_sort, mnt_sort, max_D=16, max_O=180 / 6).astype(np.float32)
    keep_list = np.ones(mnt_sort.shape[0])
    for i in range(mnt_sort.shape[0]):
        if keep_list[i] == 0:
            continue
        keep_list[i + 1 :] = keep_list[i + 1 :] * (1 - inrange[i, i + 1 :])
    return mnt_sort[keep_list.astype(np.bool), :]



def mnt_writer(mnt, file_name, line):
    f = open(file_name, "w")
    f.write("%d %d\n" % (line[1], line[0]))
    f.write("%d\n" % mnt.shape[0])
    for i in range(mnt.shape[0]):
        f.write(
            "%d %d %d %d\n"
            % (mnt[i, 0], mnt[i, 1], mnt[i, 2], mnt[i, 3] * 100)
        )
    f.close()
    return

def mnt_writer_verifinger(mnt, file_name, line):
    f = open(file_name, "w")
    f.write("%d\n" % line[0])
    f.write("%d\n" % line[1])
    f.write("0\n0\n%d\n"% mnt.shape[0])
    for i in range(mnt.shape[0]):
        f.write(
            "%d %d %d %d %d\n"
            % (mnt[i, 0], mnt[i, 1], mnt[i, 2], 1, mnt[i, 3] * 100)
        )
    f.close()
    return


def seg_postprocessing(seg, seg_threshold=0.5):
    """segmentation postprocessing
    Args:
        seg (_type_): [1, 1, h, w] 0~1 mask probability.
        seg_threshold (float, optional): threshold. Defaults to 0.5.

    Returns:
        _type_: [h, w] 0-1 mask.
    """
    round_seg = (seg.cpu().squeeze().numpy() > seg_threshold).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    seg = cv2.morphologyEx(round_seg, cv2.MORPH_OPEN, kernel)
    return seg


def enh_postprocessing(enh,seg):
    """enhancement postprocessing

    Args:
        enh (_type_): [1, 1, h, w] gabor real results.
        seg (_type_): [h, w] values are 0 and 1.
    Returns:
        _type_: [h, w] 0-255 array.
    """
    seg = (seg>0.5)
    enh = enh.cpu().squeeze().numpy()
    enh *= seg
    enh_max = np.max(enh)
    enh_min = np.min(enh)
    enh = (enh - enh_min) * 255 / (enh_max - enh_min)
    return np.uint8(enh)


def phase_postprocessing(Ir, Ii, threshold=1e-5,eps=1e-6):
    """ phase postprocessing

    Args:
        Ir (_type_): [1, 1, h, w] real components after gabor filtering.
        Ii (_type_): [1, 1, h, w] imaginary components after gabor filtering.
        threshold (_type_, optional): confidence based on amplitude. Defaults to 1e-5.

    Returns:
        phase: [h, w] angle.
        intensity: [h, w] amplitude.
    """
    Ir = Ir.cpu().squeeze().numpy()+eps
    Ii = Ii.cpu().squeeze().numpy()+eps  
    Z = Ir + 1j* Ii
    intensity = np.abs(Z)
    phase = np.angle(Z)
    phase[intensity < threshold] = 2 * np.pi  # invalid
    return phase, intensity


def ori_postprocessing(ori_out):
    """orientation postprocessing

    Args:
        ori_out (_type_): [1, 90, h, w] 0~1 otientation classfication probability.

    Returns:
        ori (_type_): [h,w] -90~90 orientation.
        ori_out (_type_): [90, h, w] 0~1 otientation classfication probability.
    """
    ori = orientation_highest_peak(ori_out)
    ori = torch.argmax(ori, dim=1, keepdim=True) * 2.0 - 90
    ori = ori.cpu().squeeze().numpy()
    ori_out = ori_out.cpu().squeeze().numpy()
    return ori, ori_out


def mnt_postprocessing(seg, mnt_o, mnt_w, mnt_h, mnt_s, mnt_threshold=0.5,max_mnt=800):
    """minutiae postprocessing

    Args:
        seg (_type_): [h, w] 0-1 segmentation
        mnt_o (_type_): [1, 90, h, w] minutiae orientation
        mnt_w (_type_): [1, 8, h, w] minutiae weight/8
        mnt_h (_type_): [1, 8, h, w] minutiae height/8
        mnt_s (_type_): [1, 1, h, w] 0~1 minutiae score
        mnt_threshold (float, optional): minutiae threshold. Defaults to 0.5.

    Returns:
        mnt: n * (col, row, ori, score)
    """
    mnt_o = mnt_o.cpu().squeeze().numpy()
    mnt_w = mnt_w.cpu().squeeze().numpy()
    mnt_h = mnt_h.cpu().squeeze().numpy()
    mnt_s = mnt_s.cpu().squeeze().numpy()

    mnt = label2mnt(mnt_s * seg, mnt_w, mnt_h, mnt_o, thresh=mnt_threshold,max_mnt=max_mnt)
    mnt = nms(mnt)
    # mnt[:,:2] = mnt[:,:2] * 180 / np.pi

    return mnt

def postprocessing(img,ori,seg,enh=None,seg_gt=None,mnt_o=None, mnt_w=None, mnt_h=None, mnt_s=None):
    img = np.uint8(img.squeeze().cpu().numpy() * 255)

    seg = seg_postprocessing(seg)
    if seg_gt is not None:
        seg_gt = seg_postprocessing(seg_gt)

    if seg_gt is not None:
        ori, _ = ori_postprocessing(ori)
        ori[seg_gt<0.5]=0
    else:
        ori, _ = ori_postprocessing(ori)
        ori[seg<0.5]=0


    f0,f1 = (img.shape[0]/seg.shape[0]),(img.shape[1]/seg.shape[1])
    seg_rsz = zoom(seg,zoom=(f0,f1),order=0)
    if enh is not None:
        if seg_gt is not None:
            seg_gt_rsz = zoom(seg_gt,zoom=(f0,f1),order=0)
            enh = enh_postprocessing(enh,seg=seg_gt_rsz)
        else:
            enh = enh_postprocessing(enh,seg=seg_rsz)

    if (mnt_o is not None) and (mnt_h is not None) and (mnt_w is not None) and (mnt_s is not None):
        if seg_gt is not None:
            mnt = mnt_postprocessing(
                seg_gt,
                mnt_o,
                mnt_w,
                mnt_h,
                mnt_s,
                mnt_threshold=0.5,
            )
        else:
            mnt = mnt_postprocessing(
                seg,
                mnt_o,
                mnt_w,
                mnt_h,
                mnt_s,
                mnt_threshold=0.5,
            )
        mntS = mnt_s.cpu().squeeze().numpy()
    else:
        mnt,mntS = None, None 
    return img,seg_rsz,ori,enh,mnt,mntS

def draw_minutiae(ax, mnt_lst, arrow_length=20, color="red", linewidth=1.2):
    for mnt in mnt_lst:
        try:
            x, y, ori = mnt[:3]
            dx, dy = arrow_length * np.cos(ori * np.pi / 180), arrow_length * np.sin(ori * np.pi / 180)
            ax.scatter(x, y, marker="s", facecolors="none", edgecolor=color, linewidths=linewidth)
            ax.plot([x, x + dx], [y, y + dy], "-", color=color, linewidth=linewidth)
        except:
            x, y = mnt[:2]
            ax.scatter(x, y, marker="s", facecolors="none", edgecolor=color, linewidths=linewidth)


def mkdir(path):
    if not osp.isdir(path):
        try:
            os.makedirs(path)
        except FileExistsError as err:
            pass


def draw_minutia_on_finger(
    img,
    mnt_lst,
    save_path,
    cmap="gray",
    vmin=None,
    vmax=None,
    arrow_length=20,
    color="red",
    linewidth=1.2,
):
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    draw_minutiae(ax, mnt_lst, arrow_length, color, linewidth)

    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_axis_off()
    fig.tight_layout()
    mkdir(osp.dirname(save_path))
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def draw_orientation(ax, ori, mask=None, factor=8, stride=32, color="lime", linewidth=1.5):
    """draw orientation filed

    Parameters:
        [None]
    Returns:
        [None]
    """
    ori = ori * np.pi / 180
    for ii in range(stride // factor // 2, ori.shape[0], stride // factor):
        for jj in range(stride // factor // 2, ori.shape[1], stride // factor):
            if mask is not None and mask[ii, jj] == 0:
                continue
            x, y, o, r = jj, ii, ori[ii, jj], stride * 0.8
            ax.plot(
                [x * factor - 0.5 * r * np.cos(o), x * factor + 0.5 * r * np.cos(o)],
                [y * factor - 0.5 * r * np.sin(o), y * factor + 0.5 * r * np.sin(o)],
                "-",
                color=color,
                linewidth=linewidth,
            )


def draw_img_with_orientation(
    img,
    ori,
    save_path,
    factor=8,
    stride=16,
    cmap="gray",
    vmin=None,
    vmax=None,
    mask=None,
    color="lime",
    dpi=100,
):
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    draw_orientation(
        ax,
        ori,
        mask=mask,
        factor=factor,
        stride=stride,
        color=color,
        linewidth=dpi / 50,
    )

    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_axis_off()
    fig.tight_layout()
    fig.set_size_inches(img.shape[1] * 1.0 / dpi, img.shape[0] * 1.0 / dpi)
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
