'''
Description: 
Author: Xiongjun Guan
Date: 2023-03-30 20:22:00
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-03-31 10:10:16

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description='Train parameters')
    parser.add_argument('--epochs', '-e', metavar='E',
                        type=int, default=20, help='number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size',
                        metavar='B', type=int, default=4, help='batch size')
    parser.add_argument('--learning-rate', '-l', dest='lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--scheduler-type', '-s', dest='scheduler_type', default="CosineAnnealingLR",
                        help='scheduler type')
    parser.add_argument('--optimizer', '-o', dest='optimizer', default="adamW",
                        help='optimizer')
    parser.add_argument(
        "--cuda-ids", "-c", dest="cuda_ids", default=[4,5,6,7], help="use cuda numbers"
    )
    parser.add_argument(
        "--model", "-m", dest="model", default='FingerNet', help="model name"
    )
    parser.add_argument(
        "--train-mode", "-t", dest="train_mode", default='ridge', help="train mode (ridge or full)"
    )
    return parser.parse_args()

def get_args_weight():
    parser = argparse.ArgumentParser(
        description='Loss parameters')
    parser.add_argument('--ori_1', default=0.1, help='orientation focal coherence loss')
    parser.add_argument('--ori_2', default=0.1, help='orientation focal loss with minutiae direction')
    parser.add_argument('--seg', default=10.0, help='segmentation focal loss')
    parser.add_argument('--mnt_w', default=0.5, help='minutiae weight focal loss')
    parser.add_argument('--mnt_h', default=0.5, help='minutiae heght focal loss')
    parser.add_argument('--mnt_o', default=0.5, help='minutiae orientation focal loss')
    parser.add_argument('--mnt_s', default=200.0, help='minutiae score focal loss')
    return parser.parse_args()
