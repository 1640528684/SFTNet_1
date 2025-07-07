# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import cv2
import random
from cv2 import rotate
import numpy as np
import torch


def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img


def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path):
    """Paired random crop.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs


def paired_random_crop_hw(img_gts, img_lqs, gt_patch_size_h, gt_patch_size_w, scale, gt_path):
    """Paired random crop.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    lq_patch_size_h = gt_patch_size_h // scale
    lq_patch_size_w = gt_patch_size_w // scale

    top = random.randint(0, h_lq - lq_patch_size_h)
    left = random.randint(0, w_lq - lq_patch_size_w)

    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size_h, left:left + lq_patch_size_w, ...]
        for v in img_lqs
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size_h, left_gt:left_gt + gt_patch_size_w, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs

def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False, vflip=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    if vflip or rotation:
        vflip = random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
            if img.shape[2] == 6:
                img = img[:,:,[3,4,5,0,1,2]].copy() # swap left/right
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img

def adjust_image_size(img, patch_size):
    """调整图像尺寸到patch_size的整数倍（保持长宽比）"""
    if isinstance(img, np.ndarray):  # 处理numpy数组输入
        h, w = img.shape[:2]
        new_h = (h // patch_size) * patch_size
        new_w = (w // patch_size) * patch_size
        return img[:new_h, :new_w]
    elif isinstance(img, torch.Tensor):  # 处理PyTorch Tensor输入
        h, w = img.shape[-2:]
        new_h = (h // patch_size) * patch_size
        new_w = (w // patch_size) * patch_size
        return img[..., :new_h, :new_w]
    else:
        raise TypeError(f"Unsupported input type: {type(img)}")

# 新增类包装器
class PairedRandomCrop:
    def __init__(self, gt_patch_size, scale):
        self.gt_patch_size = gt_patch_size
        self.scale = scale

    def __call__(self, results):
        # 适配BasicSR的数据流格式（results是包含'lq'和'gt'的dict）
        lq, gt = paired_random_crop(
            img_gts=[results['gt']],
            img_lqs=[results['lq']],
            gt_patch_size=self.gt_patch_size,
            scale=self.scale,
            gt_path=''  # 此处留空，因原始函数需要但实际不需要
        )
        results['lq'], results['gt'] = lq[0], gt[0]
        return results
class Augment:
    def __init__(self, hflip=True, rotation=True, vflip=False):
        self.hflip = hflip
        self.rotation = rotation
        self.vflip = vflip

    def __call__(self, results):
        # 处理成对数据
        imgs, _ = augment(
            imgs=[results['lq'], results['gt']],
            hflip=self.hflip,
            rotation=self.rotation,
            vflip=self.vflip,
            flows=None,
            return_status=False
        )
        results['lq'], results['gt'] = imgs[0], imgs[1]
        return results
    
class AdjustSize:
    """替代Lambda的尺寸调整类"""
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, results):
        if isinstance(results, dict):  # 处理字典输入
            results['lq'] = self._adjust(results['lq'])
            if 'gt' in results:
                results['gt'] = self._adjust(results['gt'])
        else:  # 兼容单图像输入
            results = self._adjust(results)
        return results

    def _adjust(self, img):
        """实际调整函数"""
        if isinstance(img, np.ndarray):
            h, w = img.shape[:2]
            new_h = (h // self.patch_size) * self.patch_size
            new_w = (w // self.patch_size) * self.patch_size
            return img[:new_h, :new_w]
        elif isinstance(img, torch.Tensor):
            h, w = img.shape[-2:]
            new_h = (h // self.patch_size) * self.patch_size
            new_w = (w // self.patch_size) * self.patch_size
            return img[..., :new_h, :new_w]
        raise TypeError(f"Unsupported type: {type(img)}")

class Compose:
    """BasicSR专用的transforms组合类（替代torchvision的Compose）"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            # 同时支持字典输入和图像直接输入
            if isinstance(data, dict):
                data = t(data)
            else:
                # 如果是单图像输入，临时包装为字典
                tmp = {'img': data}
                tmp = t(tmp)
                data = tmp['img']
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
