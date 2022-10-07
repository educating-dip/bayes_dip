"""
Provides a U-Net-like architecture.
"""
from typing import Sequence
import torch
from torch import nn, Tensor
import numpy as np


class UNet(nn.Module):
    """U-Net model"""
    def __init__(self,
            in_ch: int,
            out_ch: int,
            channels: Sequence[int],
            skip_channels: Sequence[int],
            use_sigmoid: bool = True,
            use_norm: bool = True,
            sigmoid_saturation_thresh: float = 9.):
        """
        Parameters
        ----------
        in_ch : int
            Number of input channels.
        out_ch : int
            Number of output channels.
        channels : sequence of int
            Numbers of channels, specified individually for each scale.
        skip_channels : sequence of int
            Numbers of skip channels, specified individually for each scale.
            Note that then length must match ``channels``, but ``channels[-1]`` is unused,
            since there is no skip connection at the coarsest scale.
        use_sigmoid : bool, optional
            Whether to include a sigmoid activation at the network output.
            The default is ``True``.
        use_norm : bool, optional
            Whether to include group norm layers after each convolutional layer.
            The default is ``True``.
        sigmoid_saturation_thresh : float, optional
            Threshold for clamping pre-sigmoid activations
            if ``saturation_safety`` (is ``True`` by default) in :meth:`forward`.
            Has no effect if not ``use_sigmoid``.
        """
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

    def forward(self,
            x0: Tensor,
            saturation_safety: bool = True,
            return_pre_sigmoid: bool = False
            ) -> Tensor:
        """
        Parameters
        ----------
        x0 : Tensor
            Network input.
        saturation_safety : bool, optional
            If ``self.use_sigmoid``, this option controls whether the pre-sigmoid activations are
            clamped to ``(-self.sigmoid_saturation_thresh, self.sigmoid_saturation_thresh)``.
            The default is ``True``.
        return_pre_sigmoid : bool, optional
            If ``self.use_sigmoid``, this option can be used to return the pre-sigmoid activations
            instead of the final network output. This may save from numerically unstable inversion
            of the sigmoid, while the user can still manually apply :func:`torch.sigmoid` to obtain
            the final network output. If ``saturation_safety`` is active, the returned pre-sigmoid
            activations are clamped.
            The default is ``False``.

        Returns
        -------
        Tensor
            Network output (or pre-sigmoid activations if ``return_pre_sigmoid``).
        """
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


class UNetReturnPreSigmoid(nn.Module):
    """
    Same as UNet, but always passing ``return_pre_sigmoid=True`` and defaulting to
    ``saturation_safety=False`` in ``UNet.forward``.
    """
    def __init__(self, unet: UNet):
        super().__init__()
        self.unet = unet

    def forward(self,
            x: Tensor,
            saturation_safety: bool = False) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Network input.
        saturation_safety : bool, optional
            If ``self.unet.use_sigmoid``, this option controls whether the pre-sigmoid activations
            are clamped to ``(-self.sigmoid_saturation_thresh, self.sigmoid_saturation_thresh)``.
            The default is ``False``.

        Returns
        -------
        Tensor
            Pre-sigmoid activations of the network.
        """
        return self.unet(x, saturation_safety=saturation_safety, return_pre_sigmoid=True)


class DownBlock(nn.Module):
    """
    Down-sampling block.

    Includes two convolutional layers. The down-sampling is obtained by ``stride=2``
    in the first convolutional layer.
    """
    def __init__(self,
            in_ch: int,
            out_ch: int,
            kernel_size: int = 3,
            num_groups: int = 4,
            use_norm: bool = True):
        """
        Parameters
        ----------
        in_ch : int
            Number of input channels.
        out_ch : int
            Number of output channels.
        kernel_size : int, optional
            Kernel size. The default is ``3``.
        num_groups : int, optional
            Number of groups for group norm. The default is ``4``.
        use_norm : bool, optional
            Whether to include group norm layers after each convolutional layer.
            The default is ``True``.
        """
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Block input.

        Returns
        -------
        Tensor
            Block output.
        """
        x = self.conv(x)
        return x


class InBlock(nn.Module):
    """Input block"""
    def __init__(self,
            in_ch: int,
            out_ch: int,
            kernel_size: int = 3,
            num_groups: int = 2,
            use_norm: bool = True):
        """
        Parameters
        ----------
        in_ch : int
            Number of input channels.
        out_ch : int
            Number of output channels.
        kernel_size : int, optional
            Kernel size. The default is ``3``.
        num_groups : int, optional
            Number of groups for group norm. The default is ``2``.
        use_norm : bool, optional
            Whether to include a group norm layer after the convolutional layer.
            The default is ``True``.
        """
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Block input.

        Returns
        -------
        Tensor
            Block output.
        """
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    """Up-sampling block"""
    def __init__(self,
            in_ch: int,
            out_ch: int,
            skip_ch: int = 4,
            kernel_size: int = 3,
            num_groups: int = 2,
            use_norm: bool = True):
        """
        Parameters
        ----------
        in_ch : int
            Number of input channels.
        out_ch : int
            Number of output channels.
        skip_ch : int, optional
            Number of skip channels. The default is ``4``.
        kernel_size : int, optional
            Kernel size. The default is ``3``.
        num_groups : int, optional
            Number of groups for group norm. The default is ``2``.
        use_norm : bool, optional
            Whether to include a group norm layer after the convolutional layer.
            The default is ``True``.
        """
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

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x1 : Tensor
            Block input.
        x2 : Tensor
            Input from skip connection.

        Returns
        -------
        Tensor
            Block output.
        """
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
    """Layer concatenating channels.

    Crops to the central image parts if inputs have different image shapes.
    """
    def forward(self, *inputs: Tensor) -> Tensor:
        """
        Parameters
        ----------
        inputs : list of Tensor
            Inputs to concatenate.

        Returns
        -------
        Tensor
            Concatenated tensor.
        """
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
                inputs_.append(inp[:, :, diff2:diff2 + target_shape2,
                                   diff3:diff3 + target_shape3])
        return torch.cat(inputs_, dim=1)


class OutBlock(nn.Module):
    """Output block"""
    def __init__(self, in_ch: int, out_ch: int):
        """
        Parameters
        ----------
        in_ch : int
            Number of input channels.
        out_ch : int
            Number of output channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Block input.

        Returns
        -------
        Tensor
            Block output.
        """
        x = self.conv(x)
        return x

    def __len__(self):
        return len(self._modules)
