"""
Provides a U-Net-like architecture.
"""
import torch
from torch import nn
import numpy as np


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, channels, skip_channels,
                 use_sigmoid=True, use_norm=True, sigmoid_saturation_thresh=9):
        super().__init__()
        assert (len(channels) == len(skip_channels))
        self.scales = len(channels)
        self.use_sigmoid = use_sigmoid
        self.sigmoid_saturation_thresh = sigmoid_saturation_thresh
        self.inc = InBlock(in_ch, channels[0], use_norm=use_norm)
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        for i in range(1, self.scales):
            self.down.append(DownBlock(in_ch=channels[i - 1],
                                       out_ch=channels[i],
                                       use_norm=use_norm))
        for i in range(1, self.scales):
            self.up.append(UpBlock(in_ch=channels[-i],
                                   out_ch=channels[-i - 1],
                                   skip_ch=skip_channels[-i],
                                   use_norm=use_norm))
        self.outc = OutBlock(in_ch=channels[0],
                             out_ch=out_ch)

    def forward(self, x0, saturation_safety=True, return_pre_sigmoid=False):
        xs = [self.inc(x0)]
        for i in range(self.scales - 1):
            xs.append(self.down[i](xs[-1]))
        x = xs[-1]
        for i in range(self.scales - 1):
            x = self.up[i](x, xs[-2 - i])
        out = self.outc(x)
        if self.use_sigmoid:
            if saturation_safety:
                out = out.clamp(
                        min=-self.sigmoid_saturation_thresh,
                        max=self.sigmoid_saturation_thresh)
            if not return_pre_sigmoid: 
                out = torch.sigmoid(out)
        return out 


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, num_groups=4, use_norm=True):
        super().__init__()
        to_pad = int((kernel_size - 1) / 2)
        if use_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=2, padding=to_pad),
                nn.GroupNorm(num_channels=out_ch, num_groups=num_groups),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.GroupNorm(num_channels=out_ch, num_groups=num_groups),
                nn.LeakyReLU(0.2))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=2, padding=to_pad),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.LeakyReLU(0.2))

    def forward(self, x):
        x = self.conv(x)
        return x


class InBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, num_groups=2, use_norm=True):
        super().__init__()
        to_pad = int((kernel_size - 1) / 2)
        if use_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.GroupNorm(num_channels=out_ch, num_groups=num_groups),
                nn.LeakyReLU(0.2))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.LeakyReLU(0.2))

    def forward(self, x):
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=4, kernel_size=3, num_groups=2, use_norm=True):
        super().__init__()
        to_pad = int((kernel_size - 1) / 2)
        self.skip = skip_ch > 0
        if skip_ch == 0:
            skip_ch = 1
        if use_norm:
            self.conv = nn.Sequential(
                nn.GroupNorm(num_channels=in_ch + skip_ch, num_groups=1),
                nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size, stride=1,
                          padding=to_pad),
                nn.GroupNorm(num_channels=out_ch, num_groups=num_groups),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.GroupNorm(num_channels=out_ch, num_groups=num_groups),
                nn.LeakyReLU(0.2))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size, stride=1,
                          padding=to_pad),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.LeakyReLU(0.2))

        if use_norm:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(out_ch, skip_ch, kernel_size=1, stride=1),
                nn.GroupNorm(num_channels=skip_ch, num_groups=1),
                nn.LeakyReLU(0.2))
        else:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(out_ch, skip_ch, kernel_size=1, stride=1),
                nn.LeakyReLU(0.2))

        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True)
        self.concat = Concat()

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.skip:
            x2 = self.skip_conv(x2)
        else:
            # leading batch dims like x1, skip_ch=1, image shape like x2
            x2 = torch.zeros(*x1.shape[:-3], 1, *x2.shape[-2:], dtype=x1.dtype, device=x1.device)
        x = self.concat(x1, x2)
        x = self.conv(x)
        return x


class Concat(nn.Module):
    def forward(self, *inputs):
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if (np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and
                np.all(np.array(inputs_shapes3) == min(inputs_shapes3))):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2,
                                   diff3:diff3 + target_shape3])
        return torch.cat(inputs_, dim=1)


class OutBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv(x)
        return x

    def __len__(self):
        return len(self._modules)
