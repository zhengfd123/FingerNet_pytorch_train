'''
Description: build [train / valid / test] dataloader
Author: Xiongjun Guan
Date: 2023-03-01 19:41:05
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-03-30 20:31:03

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import os.path as osp
import os
from random import randint
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import cv2
import scipy.io as scio
from torchvision.transforms import transforms as T
from glob import glob
from scipy import signal
from scipy.ndimage import zoom,rotate,shift,maximum_filter



def mnt_reader(file_name):
    f = open(file_name)
    minutiae = []
    for i, line in enumerate(f):
        if i < 2 or len(line) == 0: 
            continue
        w, h, o = [float(x) for x in line.split()]
        w, h = int(round(w)), int(round(h))
        minutiae.append([w, h, o])
    f.close()
    return minutiae

def point_rot(points, theta, b_size, a_size):
    cosA = np.cos(theta)
    sinA = np.sin(theta)
    b_center = [b_size[1]/2.0, b_size[0]/2.0]
    a_center = [a_size[1]/2.0, a_size[0]/2.0]
    points = np.dot(points-b_center, np.array([[cosA,-sinA],[sinA,cosA]]))+a_center
    return points

def prob_normalization(prob,eps=1e-6):
    prob = prob/np.clip(np.sum(prob,axis=0,keepdims=True),eps,np.inf)
    return prob


class load_dataset_train(Dataset):
    def __init__(self, img_dirs:str, mask_dirs:str, ridge_dirs:str, mnt_dirs:str, ftitle_lst:str, img_fmt="png",aug_prob=0.5):
        self.img_dirs = img_dirs
        self.mask_dirs = mask_dirs
        self.ridge_dirs = ridge_dirs
        self.mnt_dirs = mnt_dirs
        self.ftitle_lst = ftitle_lst
        self.fmt = img_fmt
        self.aug_prob = aug_prob


    def __len__(self):
        return len(self.ftitle_lst)


    def __getitem__(self, idx):
        ftitle = self.ftitle_lst[idx]
        img_dir = self.img_dirs[idx]
        mask_dir = self.mask_dirs[idx]
        ridge_dir = self.ridge_dirs[idx]
        mnt_dir = self.mnt_dirs[idx]

        img = cv2.imread(osp.join(img_dir,ftitle+f'.{self.fmt}'),0)
        seg = cv2.imread(osp.join(mask_dir,ftitle+'.png'),0)
        
        ridge = np.load(osp.join(ridge_dir,ftitle+'.npy'),allow_pickle=True).item()
        ori = ridge['orientation_distribution_map']
        mnt = np.array(mnt_reader(osp.join(mnt_dir,ftitle+'.mnt')), dtype=float)

        img_size = np.array(np.ceil(np.array(img.shape)/8)*8,dtype=np.int32) # let img_size % 8 == 0
        if np.random.rand()<self.aug_prob:
            # random rotation [0 - 360] & translation img_size / 4
            rot = np.random.rand() * 360
            tra = ((np.random.rand(2)-0.5) / 2) * img_size

            img = rotate(img, rot, reshape=False, mode='constant',cval=255)
            img = shift(img, tra, mode='constant',cval=255)
            if np.random.rand() < 0.5:
                img = 255 - img

            seg = rotate(seg, rot, reshape=False, mode='constant',cval=0)
            seg = shift(seg, tra, mode='constant',cval=0)

            ori = np.concatenate((ori[:,:,int(rot%180/2):], ori[:,:,:int(rot%180/2)]), axis=2)
            ori = rotate(ori, rot, reshape=False, mode='constant',cval=0)
            for i in range(90):
                ori[:,:,i] = shift(ori[:,:,i], tra/8, mode='constant',cval=0)

            mnt_r = point_rot(mnt[:, :2], rot/180*np.pi, img.shape, img.shape)
            mnt = np.column_stack((mnt_r+tra[[1, 0]], mnt[:, 2]-rot/180*np.pi))
        
        # only keep mnt that stay in pic & not on border
        mnt = mnt[(8<=mnt[:,0])*(mnt[:,0]<img_size[1]-8)*(8<=mnt[:, 1])*(mnt[:,1]<img_size[0]-8), :]
        ori = np.argmax(ori, axis=-1) * 2

        
        image = (img.astype(np.float32)/255.0)[None,:,:]
        segment = ((seg>0.5).astype(np.float32))[None,:,:]
        label_ori = ori[None,:,:]
        label_seg = segment[:,::8, ::8]

        minutiae_w = np.zeros((1,int(img_size[0]/8), int(img_size[1]/8)))-1
        minutiae_h = np.zeros((1,int(img_size[0]/8), int(img_size[1]/8)))-1
        minutiae_o = np.zeros((1,int(img_size[0]/8), int(img_size[1]/8)))-1
        minutiae_w[0,(mnt[:, 1]/8).astype(int), (mnt[:, 0]/8).astype(int)] = mnt[:, 0] % 8
        minutiae_h[0,(mnt[:, 1]/8).astype(int), (mnt[:, 0]/8).astype(int)] = mnt[:, 1] % 8
        minutiae_o[0,(mnt[:, 1]/8).astype(int), (mnt[:, 0]/8).astype(int)] = mnt[:, 2]

        minutiae_seg = (minutiae_o!=-1).astype(float)
        
        # use minutiae direction to supervise orientation
        minutiae_o = minutiae_o/np.pi*180+90 # [90, 450)
        minutiae_o[minutiae_o>360] = minutiae_o[minutiae_o>360]-360 # to current coordinate system [0, 360)
        minutiae_ori_o = np.copy(minutiae_o) # copy one
        minutiae_ori_o[minutiae_ori_o>=180] = minutiae_ori_o[minutiae_ori_o>=180]-180 # for strong ori label [0,180)


        # ori 2 gaussian
        gaussian_pdf = signal.gaussian(361, 3)
        y = np.reshape(np.arange(1, 180, 2), [-1,1,1])
        delta = np.array(np.abs(label_ori - y), dtype=int)
        delta = np.minimum(delta, 180-delta)+180
        label_ori = gaussian_pdf[delta]
        # ori_o 2 gaussian
        delta = np.array(np.abs(minutiae_ori_o - y), dtype=int)
        delta = np.minimum(delta, 180-delta)+180
        label_ori_o = gaussian_pdf[delta]
        # mnt_o 2 gaussian
        y = np.reshape(np.arange(1, 360, 2), [-1,1,1])
        delta = np.array(np.abs(minutiae_o - y), dtype=int)
        delta = np.minimum(delta, 360-delta)+180
        label_mnt_o = gaussian_pdf[delta]

        # w 2 gaussian & h 2 gaussian
        gaussian_pdf = signal.gaussian(17, 2)
        y = np.reshape(np.arange(0, 8), [-1,1,1])
        delta = (minutiae_w-y+8).astype(int)
        label_mnt_w = gaussian_pdf[delta]
        delta = (minutiae_h-y+8).astype(int)
        label_mnt_h = gaussian_pdf[delta]
        # mnt cls label -1:neg, 0:no care, 1:pos
        label_mnt_s = np.copy(minutiae_seg)
        if label_mnt_s[label_mnt_s==0].shape[0] > 0:
            label_mnt_s[label_mnt_s==0] = -1 # neg to -1
        label_mnt_s = (label_mnt_s+maximum_filter(label_mnt_s, size=(1,3,3)))/2 # around 3*3 pos -> 0
        # apply segmentation
        label_ori = prob_normalization(label_ori) * label_seg
        label_ori_o = prob_normalization(label_ori_o) * minutiae_seg
        label_mnt_o = prob_normalization(label_mnt_o) * minutiae_seg
        label_mnt_w = prob_normalization(label_mnt_w) * minutiae_seg
        label_mnt_h = prob_normalization(label_mnt_h) * minutiae_seg


        return {'img':image.astype(np.float32),
                'ftitle':ftitle,
                'label_ori':label_ori.astype(np.float32),
                'label_ori_o':label_ori_o.astype(np.float32),
                'label_seg':label_seg.astype(np.float32),
                'label_mnt_w':label_mnt_w.astype(np.float32),
                'label_mnt_h':label_mnt_h.astype(np.float32),
                'label_mnt_o':label_mnt_o.astype(np.float32),
                'label_mnt_s':label_mnt_s.astype(np.float32),
                }
    


def get_dataloader_train(img_dirs:str, mask_dirs:str, ridge_dirs:str, mnt_dirs:str, ftitle_lst:str,batch_size=1, img_fmt="png",aug_prob=0.5):
    # Create dataset
    try:
        dataset = load_dataset_train(img_dirs, mask_dirs, ridge_dirs, mnt_dirs, ftitle_lst, img_fmt=img_fmt,aug_prob=aug_prob)
    except Exception as e:
        logging.error("Error in DataLoader: ", repr(e))
        return

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    logging.info(f'n_train:{len(dataset)}')

    return train_loader

def get_dataloader_valid(img_dirs:str, mask_dirs:str, ridge_dirs:str, mnt_dirs:str, ftitle_lst:str,batch_size=1, img_fmt="png"):
    # Create dataset
    try:
        dataset = load_dataset_train(img_dirs, mask_dirs, ridge_dirs, mnt_dirs, ftitle_lst, img_fmt=img_fmt,aug_prob=0.0)
    except Exception as e:
        logging.error("Error in DataLoader: ", repr(e))
        return

    valid_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logging.info(f'n_valid:{len(dataset)}')

    return valid_loader

class load_dataset_test(Dataset):
    def __init__(self, img_dir:str, img_fmt="png"):
        self.img_dir = img_dir
        self.fmt = img_fmt
   
        fname_lst = glob(osp.join(img_dir,f'*.{img_fmt}'))
        self.ftitle_lst = [osp.basename(fname).replace(f'.{img_fmt}','') for fname in fname_lst]


    def __len__(self):
        return len(self.ftitle_lst)


    def __getitem__(self, idx):
        ftitle = self.ftitle_lst[idx]
        img = cv2.imread(osp.join(self.img_dir,ftitle+f'.{self.fmt}'),0)
        img = (np.float32(img)/255)[np.newaxis,:,:]

        return {'img':img,'ftitle':ftitle}

def get_dataloader_test(img_dir,batch_size,img_fmt="png"):
    # Create dataset
    try:
        dataset = load_dataset_test(img_dir,img_fmt)
    except Exception as e:
        logging.error("Error in DataLoader: ", repr(e))
        return
    
    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return train_loader
