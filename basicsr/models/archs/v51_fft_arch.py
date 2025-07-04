# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base
from einops import rearrange
from .layers import *
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class SimpleGate(nn.Module):
    def __init__(self, dim):
        super(SimpleGate, self).__init__()
        self.norm = nn.InstanceNorm2d(dim, affine=True)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return self.norm(x1) * x2


class SimpleGate2(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


# class DFFN(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor=2, bias=False, patch_size=8):
#         super(DFFN, self).__init__()
#         self.patch_size = patch_size
#         self.dim = dim
#         self.ffn_expansion_factor = ffn_expansion_factor
#         self.bias = bias
#         #self.hidden_features = int(dim * ffn_expansion_factor)  # 计算隐藏层通道数
#         self.hidden_features = int(dim * ffn_expansion_factor)
#         self.half_hidden_features = self.hidden_features // 2
        
#         self.fft = nn.Parameter(torch.randn(
#             self.hidden_features,  # 通道数与隐藏层一致
#             patch_size,
#             patch_size // 2 + 1
#         ))
        
#         self.before_dwconv = nn.Conv2d(self.half_hidden_features, dim, kernel_size=1, bias=bias)
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias)
#         self.project_in = nn.Conv2d(dim, self.hidden_features, kernel_size=1, bias=bias)
#         # 修改 project_out 输入通道数以匹配 half_hidden_features (256)，而非 hidden_features (512)
#         self.project_out = nn.Conv2d(self.half_hidden_features, dim, kernel_size=1, bias=bias)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         # === 自动 padding 到 patch_size 的整数倍 ===
#         pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
#         pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
#         if pad_h > 0 or pad_w > 0:
#             x = F.pad(x, (0, pad_w, 0, pad_h))
            
#         Hp, Wp = H + pad_h, W + pad_w
#         x = self.project_in(x)
#         print("After project_in:", x.shape)  # 应该是 [B, 512, H, W]
#         x_patch = rearrange(x, 'b c (h p1) (w p2) -> b h w c p1 p2', p1=self.patch_size, p2=self.patch_size)
        
#         h_blocks = H // self.patch_size
#         w_blocks = W // self.patch_size
#         x_fft_input = rearrange(x_patch, 'b h w c p1 p2 -> (b h w) c p1 p2')
#         x_fft = torch.fft.rfft2(x_fft_input)
#         expanded_fft = self.fft.unsqueeze(0).expand(x_fft.shape[0], -1, -1, -1)
#         x_fft = x_fft * expanded_fft
#         x = torch.fft.irfft2(x_fft, s=(self.patch_size, self.patch_size))
        
#         # 恢复原始结构
#         x = rearrange(x, '(b h w) c p1 p2 -> b h w c p1 p2',
#                       b=B, h=Hp // self.patch_size, w=Wp // self.patch_size)
#         x = rearrange(x, 'b h w c p1 p2 -> b c (h p1) (w p2)',
#                       p1=self.patch_size, p2=self.patch_size)
        

#         #x = self.dwconv(x)  # [B, hidden_features=512, H, W]
#         print("Before chunk:", x.shape)  # 应该是 [B, 512, H, W]
#         x1, x2 = x.chunk(2, dim=1)  # 拆分为两个 256 通道
#         print("After chunk:", x1.shape, x2.shape)  # 都应为 [B, 256, H, W]
#         x = F.gelu(x1) * x2  # 通道数保持 256
#         print("After non-linearity:", x.shape)  # 应该是 [B, 256, H, W]
#         x = self.before_dwconv(x)
#         print("After before_dwconv:", x.shape)  # 确保通道数仍为 256
#         x = self.dwconv(x)
#         print("After dwconv:", x.shape)  # 确保通道数仍为 256 或根据dwconv定义调整
#         x = self.project_out(x)  # 正确输入通道：256 → 输出通道：dim（如 256）
#         print("After project_out:", x.shape)  # 确保通道数符合预期
#         # === 恢复原始图像大小 ===
#         x = x[:, :, :H, :W]  # 剪裁回原始尺寸
#         return x

class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=1.0, bias=False, patch_size=8):
        super(DFFN, self).__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.ffn_expansion_factor = ffn_expansion_factor
        self.bias = bias
        
        # 确保不扩展通道
        self.hidden_features = dim
        self.half_hidden_features = dim // 2

        # FFT参数
        self.fft = nn.Parameter(torch.randn(
            self.hidden_features,
            patch_size,
            patch_size // 2 + 1
        ))

        # 网络结构
        self.project_in = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.before_dwconv = nn.Conv2d(self.half_hidden_features, self.half_hidden_features, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(self.half_hidden_features, self.half_hidden_features, kernel_size=3, padding=1, 
                               groups=self.half_hidden_features, bias=bias)
        self.project_out = nn.Conv2d(self.half_hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        identity = x
        
        B, C, H, W = x.shape
        
        # 填充处理
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        # FFT处理
        Hp, Wp = H + pad_h, W + pad_w
        x = self.project_in(x)
        
        x_patch = rearrange(x, 'b c (h p1) (w p2) -> b h w c p1 p2', 
                          p1=self.patch_size, p2=self.patch_size)
        x_fft_input = rearrange(x_patch, 'b h w c p1 p2 -> (b h w) c p1 p2')
        x_fft = torch.fft.rfft2(x_fft_input)
        expanded_fft = self.fft.unsqueeze(0).expand(x_fft.shape[0], -1, -1, -1)
        x_fft = x_fft * expanded_fft
        x = torch.fft.irfft2(x_fft, s=(self.patch_size, self.patch_size))
        
        # 重组张量
        x = rearrange(x, '(b h w) c p1 p2 -> b h w c p1 p2', 
                     b=B, h=Hp//self.patch_size, w=Wp//self.patch_size)
        x = rearrange(x, 'b h w c p1 p2 -> b c (h p1) (w p2)',
                     p1=self.patch_size, p2=self.patch_size)
        
        # 通道处理
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        
        # 深度可分离卷积
        x = self.before_dwconv(x)
        x = self.dwconv(x)
        x = self.project_out(x)
        
        # 裁剪并添加残差
        x = x[:, :, :H, :W]
        x = x + identity
        
        return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv2d(dim, dim, kernel_size=1)
        self.key = nn.Conv2d(dim, dim, kernel_size=1)
        self.value = nn.Conv2d(dim, dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(B, -1, H * W)
        value = self.value(x).view(B, -1, H * W)
        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(B, C, H, W)
        return out


# class TransformerBlock(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor=1, bias=False, LayerNorm_type='WithBias', att=False):
#         super(TransformerBlock, self).__init__()
#         self.attention = AttentionBlock(dim)
#         self.norm1 = LayerNorm(dim, LayerNorm_type)
#         self.norm2 = LayerNorm(dim, LayerNorm_type)
#         self.ffn = DFFN(dim, ffn_expansion_factor, bias)


#     def forward(self, x):
#         x = x + checkpoint(self.attention, self.norm1(x))  # 启用梯度检查点
#         x = x + checkpoint(self.ffn, self.norm2(x))        # 启用梯度检查点
#         return x
class TransformerBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=1.0):
        super().__init__()
        self.attention = AttentionBlock(dim)
        self.ffn = DFFN(dim, ffn_expansion_factor)
        
    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.ffn(x)
        return x


class FPNBlock(nn.Module):
    """特征金字塔模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lateral_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.smooth_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
    def forward(self, x, top_down=None):
        lateral = self.lateral_conv(x)
        if top_down is not None:
            lateral += F.interpolate(top_down, size=lateral.shape[2:], mode='bilinear')
        return self.smooth_conv(lateral)

    
class NAFBlock(nn.Module):
    def __init__(self, img_channel=3, width=48, enc_blk_nums=[2,2,3,6], 
                middle_blk_num=2, dec_blk_nums=[4,2,1,1], patch_size=8,
                denoising_config={'enable_layers': [1,2,3], 'type': 'adaptive'}):
        super().__init__()
        self.patch_size = patch_size
        self.width = width
        
        # ---------------------------- 编码器部分 ----------------------------
        self.encoders = nn.ModuleList()
        self.denoising_modules = nn.ModuleList()
        enc_channels = []
        
        for i in range(len(enc_blk_nums)):
            in_ch = img_channel if i == 0 else width * (2 ** (i-1))
            out_ch = width * (2 ** i)
            enc_channels.append(out_ch)
            
            # 增强的编码器块（带残差连接）
            layers = []
            for _ in range(enc_blk_nums[i]):
                layers += [
                    nn.Conv2d(in_ch if _ == 0 else out_ch, out_ch, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    ChannelAttention(out_ch)  # 通道注意力
                ]
                if _ > 0:  # 残差连接
                    layers.append(nn.Conv2d(out_ch, out_ch, 1))
            self.encoders.append(nn.Sequential(*layers))
            
            # 动态创建去噪模块
            if i in denoising_config['enable_layers']:
                if denoising_config['type'] == 'adaptive':
                    self.denoising_modules.append(AdaptiveDenoiser(out_ch, i))
                else:
                    self.denoising_modules.append(DenoisingModule(out_ch))
            else:
                self.denoising_modules.append(nn.Identity())

        # ---------------------------- 中间部分 ----------------------------
        self.middle_blocks = nn.ModuleList()
        for _ in range(middle_blk_num):
            self.middle_blocks.append(
                TransformerBlock(enc_channels[-1], ffn_expansion_factor=1.0)
            )
        
        self.middle_proj = nn.Sequential(
            nn.Conv2d(enc_channels[-1], width, 1),
            LayerNorm2d(width)
        )

        # ---------------------------- 解码器部分 ----------------------------
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.fpn = nn.ModuleList()
        
        # 通道适配器（统一为width通道）
        self.channel_adapters = nn.ModuleList([
            nn.Conv2d(c, width, 1) for c in enc_channels
        ])
        
        # 特征融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(width * len(enc_blk_nums), width, 1),
            ChannelAttention(width)  # 融合后加注意力
        )
        
        for i in range(len(dec_blk_nums)):
            layers = []
            for _ in range(dec_blk_nums[i]):
                layers += [
                    nn.Conv2d(width, width, 3, padding=1),
                    nn.GELU(),
                    ChannelAttention(width)
                ]
            self.decoders.append(nn.Sequential(*layers))
            self.ups.append(nn.Upsample(scale_factor=2, mode='bilinear'))
            self.fpn.append(FPNBlock(width, width))

        # ---------------------------- 输出部分 ----------------------------
        self.final_conv = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, img_channel, 1)
        )

    def forward(self, x):
        # 输入尺寸检查
        x, original_h, original_w = self._check_image_size(x)
        
        # ---------------------------- 编码 ----------------------------
        enc_features = []
        for i, (encoder, denoise) in enumerate(zip(self.encoders, self.denoising_modules)):
            x = encoder(x)
            if isinstance(denoise, (AdaptiveDenoiser, DenoisingModule)):
                x = denoise(x)  # 传递层索引
            enc_features.append(x)
            x = F.max_pool2d(x, 2)

        # ---------------------------- 中间转换 ----------------------------
        for blk in self.middle_blocks:
            x = blk(x)
        x = self.middle_proj(x)

        # ---------------------------- 解码 ----------------------------
        fpn_features = []
        for i, (decoder, up, fpn_block) in enumerate(zip(
                self.decoders, self.ups, self.fpn
        )):
            # 跳跃连接处理
            skip = self.channel_adapters[-i-1](enc_features[-i-1])
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear')
            
            x = up(x)
            x = x + skip  # 残差连接
            
            # 解码器处理
            x = decoder(x)
            fpn_features.append(fpn_block(x, skip))

        # ---------------------------- 特征融合 ----------------------------
        # 多尺度特征融合
        max_size = (max(f.size(2) for f in fpn_features), 
                   max(f.size(3) for f in fpn_features))
        fused = torch.cat([
            F.interpolate(f, size=max_size, mode='bilinear') 
            for f in fpn_features
        ], dim=1)
        fused = self.fusion_conv(fused)
        
        # 最终输出
        x = x + fused  # 全局残差连接
        x = self.final_conv(x)
        return x[:, :, :original_h, :original_w]  # 裁剪回原始尺寸

    def _check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x, h, w
    



class v51fftLocal(NAFBlock, Local_Base):  # 修改继承顺序
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        NAFBlock.__init__(self, *args, **kwargs)
        Local_Base.__init__(self)
        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))
        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

    def _check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x,h,w

# class DenoisingModule(nn.Module):
#     def __init__(self, in_channels=64, out_channels=None, num_blocks=1):
#         super(DenoisingModule, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels if out_channels is not None else in_channels
#         # 第一层卷积保持通道数不变
#         self.conv_first = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#         # Transformer块
#         blocks = []
#         for _ in range(num_blocks):
#             blocks.append(TransformerBlock(dim=in_channels))
#         self.transformer_blocks = nn.Sequential(*blocks)
#         # DFFN模块
#         self.dffn = DFFN(dim=in_channels, ffn_expansion_factor=1.0)  # 设置为1.0确保不扩展通道
#         # 最后一层卷积处理输出通道
#         self.conv_last = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1)
#         # 残差连接适配器
#         if in_channels != self.out_channels:
#             self.residual_adapter = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
#         else:
#             self.residual_adapter = None
#     def forward(self, x):
#         identity = x
#         # 第一层处理
#         x = F.relu(self.conv_first(x))
#         # Transformer处理
#         x = self.transformer_blocks(x)
#         # DFFN处理
#         x = self.dffn(x)
#         # 最后一层处理
#         x = self.conv_last(x)  
#         # 处理残差连接
#         if identity.shape[1] != self.out_channels:
#             if self.residual_adapter is not None:
#                 identity = self.residual_adapter(identity)
#             else:
#                 # 如果没定义适配器但需要调整通道，使用1x1卷积
#                 identity = F.conv2d(identity, 
#                                    torch.eye(self.out_channels, identity.shape[1])
#                                    .unsqueeze(-1).unsqueeze(-1)
#                                    .to(identity.device))    
#         # 残差连接
#         x = x + identity  
#         return x
class ChannelAttention(nn.Module):
    """轻量通道注意力"""
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class AdaptiveDenoiser(nn.Module):
    def __init__(self, channels, layer_idx):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.2 + 0.2 * layer_idx))  # 深度加权
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels//4, 1),
            nn.GELU(),
            nn.Conv2d(channels//4, channels//4, 3, padding=1, groups=channels//4),
            nn.GELU(),
            nn.Conv2d(channels//4, channels, 1)
        )
        
    def forward(self, x):
        return x + self.weight * self.conv(x)  # 加权残差
    
class DenoisingModule(nn.Module):
    """标准去噪模块（兼容旧配置）"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels//2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels//2, channels, 3, padding=1)
        )
        
    def forward(self, x, layer_idx=None):  # 修改1：添加layer_idx参数
        """
        支持可选层索引
        Args:
            x: 输入张量
            layer_idx: (可选) 当前处理的层索引
        """
        if layer_idx is not None:
            # 使用layer_idx的特定逻辑
            x = self.denoise(x, layer_idx)
        else:
            # 默认处理逻辑
            x = self.denoise(x)
        return x


if __name__ == '__main__':
    img_channel = 3
    width = 32
    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]

    net = NAFBlock(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                   enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
    inp_shape = (3, 256, 256)
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)
    print(f"MACs: {macs:.2f} G, Params: {params:.2f} M")



