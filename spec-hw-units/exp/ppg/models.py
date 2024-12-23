# *----------------------------------------------------------------------------*
# * Copyright (C) 2023 Politecnico di Torino, Italy                            *
# * SPDX-License-Identifier: Apache-2.0                                        *
# *                                                                            *
# * Licensed under the Apache License, Version 2.0 (the "License");            *
# * you may not use this file except in compliance with the License.           *
# * You may obtain a copy of the License at                                    *
# *                                                                            *
# * http://www.apache.org/licenses/LICENSE-2.0                                 *
# *                                                                            *
# * Unless required by applicable law or agreed to in writing, software        *
# * distributed under the License is distributed on an "AS IS" BASIS,          *
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
# * See the License for the specific language governing permissions and        *
# * limitations under the License.                                             *
# *                                                                            *
# * Author:  Matteo Risso <matteo.risso@polito.it>                             *
# *----------------------------------------------------------------------------*

from math import ceil
from typing import Tuple, List
import torch
import torch.nn as nn

from odimo.method import ThermometricModule

__all__ = [
    "temponet_dw",
    "temponet_conv",
    "temponet_search",
]


class StdConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        groups=1,
        **kwargs,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
            groups=groups,
        )
        self.pool = kwargs.get("pool", None)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        if self.pool is not None:
            x = self.pool(x)
        return self.relu(self.bn(x))


class DWBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        **kwargs,
    ):
        super().__init__()
        self.depthwise = StdConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            **kwargs,
        )

    def forward(self, input):
        x = self.depthwise(input)
        return x


class SearchableBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        thermometric: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.block = ThermometricModule(
            [
                StdConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    **kwargs,
                ),
                DWBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    **kwargs,
                ),
            ],
            out_channels=out_channels,
            thermometric=thermometric,
        )

    def forward(self, input):
        x = self.block(input)
        return x


class TempConvBlock(nn.Module):
    """
    Temporal Convolutional Block composed of one temporal convolutional layer.
    The block is composed of :
    - Conv1d layer
    - ReLU layer
    - BatchNorm1d layer
    :param ch_in: Number of input channels
    :param ch_out: Number of output channels
    :param k_size: Kernel size
    :param dil: Amount of dilation
    :param pad: Amount of padding
    """

    def __init__(self, ch_in, ch_out, k_size, dil, pad, conv_block, **kwargs):
        super(TempConvBlock, self).__init__()
        self.conv_bn_relu = conv_block(
            ch_in,
            ch_out,
            (k_size, 1),
            (1, 1),
            (pad, 0),
            (dil, 1),
            **kwargs,
        )

    def forward(self, x):
        return self.conv_bn_relu(x)


class ConvBlock(nn.Module):
    """
    Convolutional Block composed of:
    - Conv1d layer
    - AvgPool1d layer
    - ReLU layer
    - BatchNorm1d layer
    :param ch_in: Number of input channels
    :param ch_out: Number of output channels
    :param k_size: Kernel size
    :param s: Amount of stride
    :param pad: Amount of padding
    """

    def __init__(self, ch_in, ch_out, k_size, s, dilation, pad, conv_block, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv_pool_bn_relu = conv_block(
            ch_in,
            ch_out,
            (k_size, 1),
            (s, 1),
            (pad, 0),
            (dilation, 1),
            pool=nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),
            **kwargs,
        )

    def forward(self, x):
        return self.conv_pool_bn_relu(x)


class Regressor(nn.Module):
    """
    Regressor block composed of:
    - Linear layer
    - ReLU layer
    - BatchNorm1d layer
    :param ft_in: Number of input channels
    :param ft_out: Number of output channels
    """

    def __init__(self, ft_in, ft_out):
        super(Regressor, self).__init__()
        self.fc = nn.Linear(in_features=ft_in, out_features=ft_out)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(num_features=ft_out)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TEMPONet(nn.Module):
    """
    TEMPONet architecture:
    Three repeated instances of TemporalConvBlock and ConvBlock organized as follows:
    - TemporalConvBlock
    - ConvBlock
    Two instances of Regressor followed by a final Linear layer with a single neuron.
    """

    def __init__(
        self,
        conv_block: nn.Module = StdConvBlock,
    ):
        super().__init__()

        # Parameters
        self.input_shape = (4, 256)  # default for PPG-DALIA dataset
        self.dil = [2, 2, 1, 4, 4, 8, 8]
        self.rf = [5, 5, 5, 9, 9, 17, 17]
        self.ch = [32, 32, 64, 64, 64, 128, 128, 128, 128, 256, 128]

        # 1st instance of two TempConvBlocks and ConvBlock
        k_tcb00 = ceil(self.rf[0] / self.dil[0])
        self.tcb00 = TempConvBlock(
            ch_in=4,
            ch_out=self.ch[0],
            k_size=k_tcb00,
            dil=self.dil[0],
            pad=((k_tcb00 - 1) * self.dil[0] + 1) // 2,
            conv_block=StdConvBlock,
        )
        k_tcb01 = ceil(self.rf[1] / self.dil[1])
        self.tcb01 = TempConvBlock(
            ch_in=self.ch[0],
            ch_out=self.ch[1],
            k_size=k_tcb01,
            dil=self.dil[1],
            pad=((k_tcb01 - 1) * self.dil[1] + 1) // 2,
            conv_block=conv_block,
        )
        k_cb0 = ceil(self.rf[2] / self.dil[2])
        self.cb0 = ConvBlock(
            ch_in=self.ch[1],
            ch_out=self.ch[2],
            k_size=k_cb0,
            s=1,
            dilation=self.dil[2],
            pad=((k_cb0 - 1) * self.dil[2] + 1) // 2,
            conv_block=StdConvBlock,
        )

        # 2nd instance of two TempConvBlocks and ConvBlock
        k_tcb10 = ceil(self.rf[3] / self.dil[3])
        self.tcb10 = TempConvBlock(
            ch_in=self.ch[2],
            ch_out=self.ch[3],
            k_size=k_tcb10,
            dil=self.dil[3],
            pad=((k_tcb10 - 1) * self.dil[3] + 1) // 2,
            conv_block=conv_block,
        )
        k_tcb11 = ceil(self.rf[4] / self.dil[4])
        self.tcb11 = TempConvBlock(
            ch_in=self.ch[3],
            ch_out=self.ch[4],
            k_size=k_tcb11,
            dil=self.dil[4],
            pad=((k_tcb11 - 1) * self.dil[4] + 1) // 2,
            conv_block=conv_block,
        )
        self.cb1 = ConvBlock(
            ch_in=self.ch[4],
            ch_out=self.ch[5],
            k_size=5,
            s=2,
            pad=2,
            dilation=1,
            conv_block=StdConvBlock,
        )

        # 3td instance of TempConvBlock and ConvBlock
        k_tcb20 = ceil(self.rf[5] / self.dil[5])
        self.tcb20 = TempConvBlock(
            ch_in=self.ch[5],
            ch_out=self.ch[6],
            k_size=k_tcb20,
            dil=self.dil[5],
            pad=((k_tcb20 - 1) * self.dil[5] + 1) // 2,
            conv_block=conv_block,
        )
        k_tcb21 = ceil(self.rf[6] / self.dil[6])
        self.tcb21 = TempConvBlock(
            ch_in=self.ch[6],
            ch_out=self.ch[7],
            k_size=k_tcb21,
            dil=self.dil[6],
            pad=((k_tcb21 - 1) * self.dil[6] + 1) // 2,
            conv_block=conv_block,
        )
        self.cb2 = ConvBlock(
            ch_in=self.ch[7],
            ch_out=self.ch[8],
            k_size=5,
            s=4,
            dilation=1,
            pad=4,
            conv_block=conv_block,
        )

        # 1st instance of regressor
        self.regr0 = Regressor(ft_in=self.ch[8] * 4, ft_out=self.ch[9])

        # 2nd instance of regressor
        self.regr1 = Regressor(ft_in=self.ch[9], ft_out=self.ch[10])

        # Output layer
        self.out_neuron = nn.Linear(in_features=self.ch[10], out_features=1)

    def forward(self, input):
        # 1st instance of two TempConvBlocks and ConvBlock
        x = self.tcb00(input.unsqueeze(-1))
        x = self.tcb01(x)
        x = self.cb0(x)
        # 2nd instance of two TempConvBlocks and ConvBlock
        x = self.tcb10(x)
        x = self.tcb11(x)
        x = self.cb1(x)
        # 3td instance of TempConvBlock and ConvBlock
        x = self.tcb20(x)
        x = self.tcb21(x)
        x = self.cb2(x)
        # Flatten
        x = x.flatten(1)
        # 1st instance of regressor
        x = self.regr0(x)
        # 2nd instance of regressor
        x = self.regr1(x)
        # Output layer
        x = self.out_neuron(x)
        return x


def temponet_dw():
    return TEMPONet(conv_block=DWBlock)


def temponet_conv():
    return TEMPONet(conv_block=StdConvBlock)


def temponet_search():
    return TEMPONet(conv_block=SearchableBlock)
