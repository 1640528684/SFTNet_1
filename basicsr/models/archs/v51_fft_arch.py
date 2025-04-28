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
        # return x1 * x2

class SimpleGate2(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(DFFN, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.patch_size = 8
        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size, patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size, patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
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
        self.normalized_shape = normalized_shape

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
        self.normalized_shape = normalized_shape

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
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
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

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        c2wh = dict([(16, 224), (32, 112), (64, 56), (128, 28), (256, 14), (512, 7), (1024, 4)])
        self.sca = MultiSpectralAttentionLayer(dw_channel // 2, c2wh[dw_channel // 2], c2wh[dw_channel // 2], freq_sel_method='top16')
        self.sg = SimpleGate(dim=dw_channel//2)
        self.sg2 = SimpleGate2()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        return y

class FPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPNBlock, self).__init__()
        # Debugging: Print the input and output channels
        print(f"Initializing FPNBlock with in_channels={in_channels}, out_channels={out_channels}")
        self.lateral = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        print(f"FPNBlock input shape: {x.shape}, lateral in_channels: {self.lateral.in_channels}")
        # 检查输入通道数是否匹配
        if x.shape[1] != self.lateral.in_channels:
            print(f"Channel mismatch detected in FPNBlock. Expected {self.lateral.in_channels} channels, but got {x.shape[1]} channels. Adjusting input channels.")
            # 动态调整输入通道数
            self.lateral = nn.Conv2d(x.shape[1], self.lateral.out_channels, kernel_size=1).to(x.device)
        x = self.lateral(x)
        x = self.smooth(x)
        return x

class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=64, enc_blk_nums=[1,1,1,28], middle_blk_num=1, dec_blk_nums=[1,1,1,1]):
        super().__init__()
        self.width = width
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.fpn = nn.ModuleList()
        self.channel_adapters = nn.ModuleList()  # 新增：通道适配器
        #self.middle_blk_num = middle_blk_num  # 新增属性
        self.middle_blocks = nn.ModuleList()  # 新增中间块列表
        
        # 初始化编码器
        for i in range(len(enc_blk_nums)):
            in_channels = img_channel if i == 0 else width * (2 ** (i-1))
            out_channels = width * (2 ** i)
            print(f"Encoder {i}: in_channels={in_channels}, out_channels={out_channels}")
            encoder_layers = [
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU()
            ]
            for _ in range(enc_blk_nums[i] - 1):
                encoder_layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            encoder_layers.append(nn.ReLU())
            self.encoders.append(nn.Sequential(*encoder_layers))
        
        # 初始化中间块（根据编码器最后一层输出通道计算）
        in_channels_middle = width * (2 ** (len(enc_blk_nums) - 1))
        for _ in range(middle_blk_num):
            self.middle_blocks.append(NAFBlock(c=in_channels_middle))  # 使用实际通道数
        
        # 初始化解码器
        for i in range(len(dec_blk_nums)):
            in_channels = width * (2 ** (len(enc_blk_nums)-i-1))
            #out_channels = width * (2 ** (len(enc_blk_nums)-i-2))
            # 避免 out_channels 为负数或零
            if len(enc_blk_nums) - i - 2 < 0:
                out_channels = width
            else:
                out_channels = width * (2 ** (len(enc_blk_nums)-i-2))
            # 检查 in_channels 和 out_channels 是否为有效的整数
            #assert isinstance(in_channels, int) and isinstance(out_channels, int), f"Invalid channel numbers: in_channels={in_channels}, out_channels={out_channels}"
            print(f"Decoder {i}: in_channels={in_channels}, out_channels={out_channels}")
            decoder_layers = [
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU()
            ]
            for _ in range(dec_blk_nums[i] - 1):
                decoder_layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            decoder_layers.append(nn.ReLU())
            self.decoders.append(nn.Sequential(*decoder_layers))
            self.ups.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        
        # 初始化FPN模块
        fpn_channels = [width * (2**i) for i in reversed(range(len(enc_blk_nums)))]
        for c in fpn_channels:
            print(f"Initializing FPNBlock with in_channels={c}, out_channels={self.width}")
            self.fpn.append(FPNBlock(c, self.width))
        
        # 初始化通道适配器
        enc_channels = [width * (2**i) for i in range(len(enc_blk_nums))]
        for i in range(len(enc_channels)):
            target_channels = width * (2**(len(enc_blk_nums)-i-1))
            # 检查 enc_channels[i] 和 target_channels 是否为有效的整数
            #assert isinstance(enc_channels[i], int) and isinstance(target_channels, int), f"Invalid channel numbers: enc_channels[i]={enc_channels[i]}, target_channels={target_channels}"
            print(f"Channel Adapter {i}: enc_channels[i]={enc_channels[i]}, target_channels={target_channels}")
            self.channel_adapters.append(
                nn.Conv2d(enc_channels[i], target_channels, kernel_size=1)
            )

    def forward(self, x):
        encs = []
        # 编码器前向传播
        for encoder in self.encoders:
            x = encoder(x)
            encs.append(x)
            # 下采样（假设每个编码器后接下采样）
            x = F.max_pool2d(x, 2)
        
        # 中间块处理
        for blk in self.middle_blocks:
            x = blk(x)
        
        # 解码器和FPN融合
        fpn_features = []
        for i, (decoder, up, fpn_block) in enumerate(zip(
            self.decoders, self.ups, self.fpn
        )):
            # 取对应的编码器输出（逆序）
            enc_skip = encs[-i-1]
            
            # 调整enc_skip的通道数（关键修改点）
            enc_skip = self.channel_adapters[len(encs) - i - 1](enc_skip)
            
            # 打印调试信息
            print(f"Before interpolation: x shape = {x.shape}, enc_skip shape = {enc_skip.shape}")
            
            # 空间尺寸对齐
            target_size = x.size()[2:]
            enc_skip = F.interpolate(enc_skip, size=target_size, mode='bilinear', align_corners=False)
            
            # 再次打印调试信息
            print(f"After interpolation: x shape = {x.shape}, enc_skip shape = {enc_skip.shape}")
            # 检查尺寸是否匹配
            assert x.shape[2:] == enc_skip.shape[2:], f"Size mismatch: x shape = {x.shape}, enc_skip shape = {enc_skip.shape}"
            
            # 确保通道数匹配
            if x.shape[1] != enc_skip.shape[1]:
                print(f"Channel mismatch detected. x channels: {x.shape[1]}, enc_skip channels: {enc_skip.shape[1]}. Adjusting enc_skip channels.")
                enc_skip = nn.Conv2d(enc_skip.shape[1], x.shape[1], kernel_size=1).to(enc_skip.device)(enc_skip)
            
            # 跳跃连接
            x = up(x)
            # 再次检查空间尺寸是否匹配
            if x.shape[2:] != enc_skip.shape[2:]:
                print(f"Spatial size mismatch detected. x size: {x.shape[2:]}, enc_skip size: {enc_skip.shape[2:]}. Adjusting enc_skip size.")
                enc_skip = F.interpolate(enc_skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = x + enc_skip  # 通道数已匹配
            x = decoder(x)
            # 应用FPNBlock并收集特征
            fpn_out = fpn_block(x)
            fpn_features.append(fpn_out)
        
        # 融合FPN特征
        fused = sum(fpn_features)
        x = x + fused
        return x
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class v51fftLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)
        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))
        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)
            
if __name__ == '__main__':
    img_channel = 3
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]

    net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                 enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)