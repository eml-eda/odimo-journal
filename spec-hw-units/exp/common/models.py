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

from typing import Tuple, List
import torch
import torch.nn as nn

from odimo.method import ThermometricModule

__all__ = [
    'mbv1_dw_8', 'mbv1_dws_8', 'mbv1_conv_8', 'mbv1_search_8',
    'mbv1_dw_16', 'mbv1_dws_16', 'mbv1_conv_16', 'mbv1_search_16', 'mbv1_search_notherm_16',
    'mbv1_dw_32', 'mbv1_dws_32', 'mbv1_conv_32', 'mbv1_search_32',
    'mbv1_search_32_dw_dws', 'mbv1_search_32_dws_conv',
]


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 groups=1, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=False,
                               groups=groups)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))


class DWBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 **kwargs):
        super().__init__()
        self.depthwise = ConvBlock(in_channels=in_channels,
                                   out_channels=in_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   groups=in_channels)

    def forward(self, input):
        x = self.depthwise(input)
        return x


class DWSBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 **kwargs):
        super().__init__()
        self.depthwise = ConvBlock(in_channels=in_channels,
                                   out_channels=in_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   groups=in_channels)
        self.pointwise = ConvBlock(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)

    def forward(self, input):
        x = self.depthwise(input)
        x = self.pointwise(x)
        return x


class SearchableBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 thermometric: bool = True):
        super().__init__()
        self.block = ThermometricModule([
            ConvBlock(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            DWBlock(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding)
        ], out_channels=out_channels, thermometric=thermometric)

    def forward(self, input):
        x = self.block(input)
        return x


class SearchableBlockDwDws(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 thermometric: bool = True):
        super().__init__()
        self.block = ThermometricModule([
            DWSBlock(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding),
            DWBlock(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding)
        ], out_channels=out_channels, thermometric=thermometric)

    def forward(self, input):
        x = self.block(input)
        return x


class SearchableBlockDwsConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 thermometric: bool = True):
        super().__init__()
        self.block = ThermometricModule([
            ConvBlock(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            DWSBlock(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding)
        ], out_channels=out_channels, thermometric=thermometric)

    def forward(self, input):
        x = self.block(input)
        return x


class MobileNet(torch.nn.Module):
    def __init__(self, input_shape: Tuple, num_classes: int,
                 thermometric: List[bool] = [True] * 8,
                 conv_block: nn.Module = DWSBlock,
                 features: int = 8,
                 initial_stride: int = 2
                 ):
        super().__init__()

        # Parameters #
        self.input_shape = input_shape
        self.inp_filters = input_shape[0]
        self.num_classes = num_classes
        self.features = features

        # Layers #
        # Oth layer
        self.layer0 = ConvBlock(in_channels=self.inp_filters,
                                out_channels=features,
                                kernel_size=3, stride=initial_stride, padding=1)
        # 1st layer
        self.dw1 = ConvBlock(in_channels=features, out_channels=features,
                             kernel_size=3, stride=1, padding=1,
                             groups=features)
        self.pw1 = ConvBlock(in_channels=features, out_channels=2*features,
                             kernel_size=1, stride=1, padding=0)
        # 2nd layer
        self.dw2 = ConvBlock(in_channels=2*features, out_channels=2*features,
                             kernel_size=3, stride=2, padding=1,
                             groups=2*features)
        self.pw2 = ConvBlock(in_channels=2*features, out_channels=4*features,
                             kernel_size=1, stride=1, padding=0)
        # 3rd layer
        self.layer3 = conv_block(in_channels=4*features, out_channels=4*features,
                                 kernel_size=3, stride=1, padding=1,
                                 thermometric=thermometric[0])
        # 4th layer
        self.dw4 = ConvBlock(in_channels=4*features, out_channels=4*features,
                             kernel_size=3, stride=2, padding=1,
                             groups=4*features)
        self.pw4 = ConvBlock(in_channels=4*features, out_channels=8*features,
                             kernel_size=1, stride=1, padding=0)
        # 5th layer
        self.layer5 = conv_block(in_channels=8*features, out_channels=8*features,
                                 kernel_size=3, stride=1, padding=1,
                                 thermometric=thermometric[1])
        # 6th layer
        self.dw6 = ConvBlock(in_channels=8*features, out_channels=8*features,
                             kernel_size=3, stride=2, padding=1,
                             groups=8*features)
        self.pw6 = ConvBlock(in_channels=8*features, out_channels=16*features,
                             kernel_size=1, stride=1, padding=0)
        # 7th layer
        self.layer7 = conv_block(in_channels=16*features, out_channels=16*features,
                                 kernel_size=3, stride=1, padding=1,
                                 thermometric=thermometric[2])
        # 8th layer
        self.layer8 = conv_block(in_channels=16*features, out_channels=16*features,
                                 kernel_size=3, stride=1, padding=1,
                                 thermometric=thermometric[3])
        # 9th layer
        self.layer9 = conv_block(in_channels=16*features, out_channels=16*features,
                                 kernel_size=3, stride=1, padding=1,
                                 thermometric=thermometric[4])
        # 10th layer
        self.layer10 = conv_block(in_channels=16*features, out_channels=16*features,
                                  kernel_size=3, stride=1, padding=1,
                                  thermometric=thermometric[5])
        # 11th layer
        self.layer11 = conv_block(in_channels=16*features, out_channels=16*features,
                                  kernel_size=3, stride=1, padding=1,
                                  thermometric=thermometric[6])
        # 12th layer
        self.dw12 = ConvBlock(in_channels=16*features, out_channels=16*features,
                              kernel_size=3, stride=2, padding=1,
                              groups=16*features)
        self.pw12 = ConvBlock(in_channels=16*features, out_channels=32*features,
                              kernel_size=1, stride=1, padding=0)
        # 13th layer
        self.layer13 = conv_block(in_channels=32*features, out_channels=32*features,
                                  kernel_size=3, stride=1, padding=1,
                                  thermometric=thermometric[7])  # previously -> stride=2?
        # Classifier
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.out = nn.Linear(32*features, num_classes)
        # nn.init.kaiming_normal_(self.out.weight)
        self._initialize_weights()

    def forward(self, input):
        # 0th layer
        x = self.layer0(input)
        # 1st layer
        x = self.dw1(x)
        x = self.pw1(x)
        # 2nd layer
        x = self.dw2(x)
        x = self.pw2(x)
        # 3rd layer
        x = self.layer3(x)
        # 4th layer
        x = self.dw4(x)
        x = self.pw4(x)
        # 5th layer
        x = self.layer5(x)
        # 6th layer
        x = self.dw6(x)
        x = self.pw6(x)
        # 7th layer
        x = self.layer7(x)
        # 8th layer
        x = self.layer8(x)
        # 9th layer
        x = self.layer9(x)
        # 10th layer
        x = self.layer10(x)
        # 11th layer
        x = self.layer11(x)
        # 12th layer
        x = self.dw12(x)
        x = self.pw12(x)
        # 13th layer
        x = self.layer13(x)
        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.out(x)
        return x

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)


def mbv1_dw_8(input_shape: Tuple, num_classes: int):
    return MobileNet(input_shape=input_shape, num_classes=num_classes,
                     conv_block=DWBlock, features=8)


def mbv1_dws_8(input_shape: Tuple, num_classes: int):
    return MobileNet(input_shape=input_shape, num_classes=num_classes,
                     conv_block=DWSBlock, features=8)


def mbv1_conv_8(input_shape: Tuple, num_classes: int):
    return MobileNet(input_shape=input_shape, num_classes=num_classes,
                     conv_block=ConvBlock, features=8)


def mbv1_search_8(input_shape: Tuple, num_classes: int):
    return MobileNet(input_shape=input_shape, num_classes=num_classes,
                     conv_block=SearchableBlock, features=8)


def mbv1_dw_16(input_shape: Tuple, num_classes: int):
    return MobileNet(input_shape=input_shape, num_classes=num_classes,
                     conv_block=DWBlock, features=16)


def mbv1_dws_16(input_shape: Tuple, num_classes: int):
    return MobileNet(input_shape=input_shape, num_classes=num_classes,
                     conv_block=DWSBlock, features=16)


def mbv1_conv_16(input_shape: Tuple, num_classes: int):
    return MobileNet(input_shape=input_shape, num_classes=num_classes,
                     conv_block=ConvBlock, features=16)


def mbv1_search_16(input_shape: Tuple, num_classes: int):
    return MobileNet(input_shape=input_shape, num_classes=num_classes,
                     conv_block=SearchableBlock, features=16)


def mbv1_search_notherm_16(input_shape: Tuple, num_classes: int):
    return MobileNet(input_shape=input_shape, num_classes=num_classes,
                     thermometric=[False] * 8,
                     conv_block=SearchableBlock,
                     features=16)


def mbv1_dw_32(input_shape: Tuple, num_classes: int, initial_stride: int = 2):
    return MobileNet(input_shape=input_shape, num_classes=num_classes,
                     conv_block=DWBlock, features=32, initial_stride=initial_stride)


def mbv1_dws_32(input_shape: Tuple, num_classes: int, initial_stride: int = 2):
    return MobileNet(input_shape=input_shape, num_classes=num_classes,
                     conv_block=DWSBlock, features=32, initial_stride=initial_stride)


def mbv1_conv_32(input_shape: Tuple, num_classes: int, initial_stride: int = 2):
    return MobileNet(input_shape=input_shape, num_classes=num_classes,
                     conv_block=ConvBlock, features=32, initial_stride=initial_stride)


def mbv1_search_32(input_shape: Tuple, num_classes: int, initial_stride: int = 2):
    return MobileNet(input_shape=input_shape, num_classes=num_classes,
                     conv_block=SearchableBlock, features=32, initial_stride=initial_stride)


def mbv1_search_32_dw_dws(input_shape: Tuple, num_classes: int):
    return MobileNet(input_shape=input_shape, num_classes=num_classes,
                     conv_block=SearchableBlockDwDws, features=32)


def mbv1_search_32_dws_conv(input_shape: Tuple, num_classes: int):
    return MobileNet(input_shape=input_shape, num_classes=num_classes,
                     conv_block=SearchableBlockDwsConv, features=32)
