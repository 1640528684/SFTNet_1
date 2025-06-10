# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from torch import nn as nn
from torchvision.models import vgg19
from collections import namedtuple
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)
@weighted_loss
def perceptual_weighted_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class FFTLoss(nn.Module):
    """L1 loss in frequency domain with FFT.

    Args:
        loss_weight (float): Loss weight for FFT loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (..., C, H, W). Predicted tensor.
            target (Tensor): of shape (..., C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (..., C, H, W). Element-wise
                weights. Default: None.
        """

        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
        return self.loss_weight * l1_loss(pred_fft, target_fft, weight, reduction=self.reduction)

# class PerceptualLoss(nn.Module):
#     def __init__(self, feature_extract_layers=None, use_l1_loss=False):
#         super(PerceptualLoss, self).__init__()
#         if feature_extract_layers is None:
#             # 默认提取第3、8、15、22层的特征
#             feature_extract_layers = ['3', '8', '15', '22']
#         self.feature_extract_layers = feature_extract_layers
#         self.use_l1_loss = use_l1_loss

#         vgg = vgg19(pretrained=True).features
#         self.model = nn.Sequential()
#         for layer in range(max(map(int, feature_extract_layers))+1):
#             self.model.add_module(str(layer), vgg[layer])
        
#         # 冻结参数
#         for param in self.model.parameters():
#             param.requires_grad = False
        
#         if self.use_l1_loss:
#             self.loss = nn.L1Loss()
#         else:
#             self.loss = nn.MSELoss()

#     def forward(self, pred, target):
#         loss = 0.0
#         for name, module in self.model._modules.items():
#             pred = module(pred)
#             target = module(target)
#             if name in self.feature_extract_layers:
#                 loss += self.loss(pred, target)
#         return loss
class PerceptualLoss(nn.Module):
    def __init__(self, feature_extract_layers=None, loss_weight=1.0, reduction='mean', start_iter=1000, use_vgg='vgg11'):
        """
        优化版感知损失：
            - 可选择使用轻量VGG（如 vgg11）
            - 只在指定迭代之后才启用感知损失
            - 每次 forward 后清除中间特征以节省显存
        :param feature_extract_layers: 提取特征的层索引列表，例如 ['3', '8']
        :param loss_weight: 损失权重
        :param reduction: 损失归约方式
        :param start_iter: 开始应用感知损失的迭代数
        :param use_vgg: 使用的VGG类型 ('vgg11' 或 'vgg19')
        """
        super(PerceptualLoss, self).__init__()

        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.start_iter = start_iter

        # 选择模型
        if use_vgg == 'vgg11':
            vgg_model = vgg11(pretrained=True).features
        elif use_vgg == 'vgg19':
            from torchvision.models import vgg19
            vgg_model = vgg19(pretrained=True).features
        else:
            raise ValueError(f"Unsupported VGG model: {use_vgg}")

        # 设置要提取的层
        if feature_extract_layers is None:
            feature_extract_layers = ['3', '8']  # vgg11 默认较浅层
        self.feature_extract_layers = feature_extract_layers

        # 构建子模型
        max_layer_idx = max(map(int, feature_extract_layers))
        self.model = nn.Sequential()
        for idx in range(max_layer_idx + 1):
            self.model.add_module(str(idx), vgg_model[idx])

        # 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 损失函数
        self.loss_func = perceptual_weighted_loss

    def forward(self, pred, target, current_iter, weight=None, **kwargs):
        """
        :param pred: 模型输出图像 (B, C, H, W)
        :param target: 真实图像 (B, C, H, W)
        :param current_iter: 当前训练迭代次数
        :param weight: 可选权重图
        """
        # 如果当前迭代未到指定次数，返回0
        if current_iter < self.start_iter:
            return torch.tensor(0.0, device=pred.device)

        loss = 0.0
        pred_features = []
        target_features = []

        with torch.no_grad():  # 不需要梯度，节省显存
            for name, module in self.model._modules.items():
                pred = module(pred)
                target = module(target)
                if name in self.feature_extract_layers:
                    pred_features.append(pred.clone())
                    target_features.append(target.clone())

        # 计算各层损失
        for p_feat, t_feat in zip(pred_features, target_features):
            loss += self.loss_func(p_feat, t_feat, weight, reduction=self.reduction)

        return self.loss_weight * loss

#新增SSIM损失
def _fspecial_gauss_1d(size, sigma):
    """Create Gaussian kernel for convolution"""
    coords = torch.arange(size).float()
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.view(1, 1, -1)
def ssim(img1, img2, window_size=11, channel=3, size_average=True):
    """Differentiable SSIM function"""
    # img1, img2: [B, C, H, W]
    window = _fspecial_gauss_1d(window_size, 1.5).to(img1.device)
    window = window.repeat(channel, 1, 1)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    numerator = (2*mu1_mu2 + C1)*(2*sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)

    ssim_index = numerator / denominator
    if size_average:
        return ssim_index.mean()
    else:
        return ssim_index.mean([1,2,3]).mean()
class SSIMLoss(nn.Module):
    """Differentiable SSIM loss.

    Args:
        loss_weight (float): Weight of this loss item.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(SSIMLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): Not used here since SSIM is a structural loss.
        """
        # Ignore weight for SSIM loss
        loss = 1.0 - ssim(pred, target)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        # 'none' returns scalar per batch
        return self.loss_weight * loss