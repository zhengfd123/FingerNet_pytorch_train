# FingerNet : An Universal Deep Convolutional Network for Extracting Fingerprint Representation
By Yao Tang, Fei Gao, JuFu Feng and YuHang Liu at Peking University


This is a `replicated version in pytorch`, for `training`.<br>
It should be noted that there are some differences in loss, such as replacing the original weighted BCE loss with focal loss.

## Links
* [TensorFlow version for deploying](https://github.com/592692070/FingerNet) This is the original version we have referred to.
* [Pytorch version for deploying](https://github.com/DasNachtlied/FingerNet_pytorch_deploy) This is our replicated version in pytorch for deploying.
* [Pytorch version for training](https://github.com/DasNachtlied/FingerNet_pytorch_train) This is our replicated version in pytorch for training.

## Introduction
FingerNet is an multi-task CNN for extracting fingerprint orientation field, segmentation, enhanced fingerprint and minutiae.

## Usage
* Run `main.py` for training.
* run `test.py` for deploying.


## Citing
```
@inproceedings{tang2017FingerNet,
    Author = {Tang, Yao and Gao, Fei and Feng, Jufu and Liu, Yuhang},
    Title = {FingerNet: An Unified Deep Network for Fingerprint Minutiae Extraction},
    booktitle = {Biometrics (IJCB), 2017 IEEE International Joint Conference on},
    Year = {2017}
    organization={IEEE}
}
```
