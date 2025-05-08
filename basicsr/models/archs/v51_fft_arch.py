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


class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False, patch_size=8):
        super(DFFN, self).__init__()
        self.patch_size = patch_size
        self.hidden_features = int(dim * ffn_expansion_factor)  # 计算隐藏层通道数（如 512）
        self.fft = nn.Parameter(torch.randn(
            self.hidden_features,  # 通道数与隐藏层一致
            patch_size,
            patch_size // 2 + 1
        ))
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=bias)
        self.project_in = nn.Conv2d(dim, self.hidden_features, 1, bias=bias)  # 输入→隐藏层通道
        # 关键修正：project_out 的输入通道应为 hidden_features // 2（拆分后通道数）
        self.project_out = nn.Conv2d(self.hidden_features // 2, dim, 1, bias=bias)  

    def forward(self, x):
        B, C, H, W = x.shape
        h_blocks = H // self.patch_size
        w_blocks = W // self.patch_size

        x = self.project_in(x)  # [B, hidden_features=512, H, W]
        x_patch = rearrange(x, 'b c (h p1) (w p2) -> b h w c p1 p2', 
                            p1=self.patch_size, p2=self.patch_size, h=h_blocks, w=w_blocks)
        x_fft_input = rearrange(x_patch, 'b h w c p1 p2 -> (b h w) c p1 p2')
        x_fft = torch.fft.rfft2(x_fft_input)
        expanded_fft = self.fft.unsqueeze(0)
        x_fft = x_fft * expanded_fft
        x = torch.fft.irfft2(x_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x, '(b h w) c p1 p2 -> b h w c p1 p2', b=B, h=h_blocks, w=w_blocks)
        x = rearrange(x, 'b h w c p1 p2 -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size)

        x = self.dwconv(x)  # [B, hidden_features=512, H, W]
        x1, x2 = x.chunk(2, dim=1)  # 拆分为两个 256 通道
        x = F.gelu(x1) * x2  # 通道数保持 256
        x = self.project_out(x)  # 正确输入通道：256 → 输出通道：dim（如 256）
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

class TransformerBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=1, bias=False, LayerNorm_type='WithBias', att=False):
        super(TransformerBlock, self).__init__()
        self.attention = AttentionBlock(dim)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class FPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPNBlock, self).__init__()
        # Lateral connection: 1×1 Conv to reduce channels
        self.lateral_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # Smooth layer: 3×3 Conv for refinement
        self.smooth_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, top_down=None):
        """
        Args:
            x (Tensor): Low-level feature from encoder (e.g., C4).
            top_down (Tensor): High-level feature from previous FPN block (e.g., upsampled C5).
        Returns:
            Tensor: Processed feature after fusion.
        """
        # Step 1: Lateral connection (1×1 Conv)
        lateral = self.lateral_conv(x)
        
        # Step 2: Upsample top-down feature to match spatial dimensions of lateral
        if top_down is not None:
            if top_down.shape != lateral.shape:
                top_down = F.interpolate(top_down, size=lateral.shape[2:], mode='bilinear', align_corners=True)
            lateral += top_down  # Element-wise addition

        # Step 3: Smooth and refine the fused features
        out = self.smooth_conv(lateral)
        out = self.relu(out)
        return out


class NAFBlock(nn.Module):
    def __init__(self, img_channel=3, width=64, enc_blk_nums=[1, 1, 1, 28],
                 middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1], patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.width = width
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.fpn = nn.ModuleList()
        self.channel_adapters = nn.ModuleList()
        self.middle_blocks = nn.ModuleList()
        self.padder_size = 16

        enc_channels = []

        # 定义编码器
        for i in range(len(enc_blk_nums)):
            if i == 0:
                in_channels = img_channel
                out_channels = width
            else:
                in_channels = width * (2 ** (i - 1))
                out_channels = width * (2 ** i)
            layers = [
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU()
            ]
            for _ in range(enc_blk_nums[i] - 1):
                layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
                layers.append(nn.ReLU())
            self.encoders.append(nn.Sequential(*layers))
            enc_channels.append(out_channels)

        # 定义中间的Transformer块
        in_channels_middle = enc_channels[-1]
        for _ in range(middle_blk_num):
            self.middle_blocks.append(TransformerBlock(in_channels_middle))

        # 定义解码器
        decoder_in_channels = []
        for i in range(len(dec_blk_nums)):
            in_channels = width * (2 ** (len(enc_blk_nums) - i - 1))
            out_channels = width * (2 ** (len(enc_blk_nums) - i - 2)) if i < len(dec_blk_nums) - 1 else width
            layers = [
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU()
            ]
            for _ in range(dec_blk_nums[i] - 1):
                layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
                layers.append(nn.ReLU())
            self.decoders.append(nn.Sequential(*layers))
            self.ups.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            decoder_in_channels.append(in_channels)

        # 定义channel_adapters，确保输入通道数与编码器的输出通道数一致
        for ec in enc_channels:
            self.channel_adapters.append(nn.Conv2d(ec, width, 1, bias=False))

        # 定义FPNBlock
        fpn_channels = [width * (2 ** i) for i in reversed(range(len(enc_blk_nums)))]
        for c in fpn_channels:
            print(f"Initializing FPNBlock with in_channels={c}, out_channels={width}")
            self.fpn.append(FPNBlock(c, width))

        self.final_conv = nn.Conv2d(width, img_channel, kernel_size=1)

    def forward(self, x):
        # 编码器部分
        encs = []
        for encoder in self.encoders:
            x = encoder(x)
            encs.append(x)
            x = F.max_pool2d(x, 2)

        # 中间Transformer块
        for blk in self.middle_blocks:
            x = blk(x)

        # 解码器和FPN部分
        fpn_features = []
        for i, (decoder, up, fpn_block) in enumerate(zip(
                self.decoders, self.ups, self.fpn
        )):
            enc_skip = encs[-i - 1]
            enc_skip = self.channel_adapters[i](enc_skip)  # 将编码器输出通道数转换为 width=64
            target_size = x.size()[2:]
            enc_skip = F.interpolate(enc_skip, size=target_size, mode='bilinear')

            x = up(x)

            if x.shape[2:] != enc_skip.shape[2:]:
                enc_skip = F.interpolate(enc_skip, size=x.shape[2:], mode='bilinear')

            x = x + enc_skip
            x = decoder(x)
            fpn_out = fpn_block(x, enc_skip)
            fpn_features.append(fpn_out)

        # 特征融合和最终卷积
        max_size = (max(f.size(2) for f in fpn_features),
                    max(f.size(3) for f in fpn_features))
        fused = sum(F.interpolate(f, size=max_size, mode='bilinear') for f in fpn_features)
        x = x + fused
        x = self.final_conv(x)
        return x

    def _check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x


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