# *----------------------------------------------------------------------------*
# * Copyright (C) 2022 Politecnico di Torino, Italy                            *
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

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils
from . import quant_module as qm
from . import quant_module_pow2 as qm2
from . import hw_models as hw
from .quant_mobilenetv1 import quantmobilenetv1_fp, quantmobilenetv1_fp_foldbn

# MR
__all__ = [
    'mixmobilenetv1_diana_naive5', 'mixmobilenetv1_diana_naive10',
    'mixmobilenetv1_diana_full', 'mixmobilenetv1_pow2_diana_full',
    'mixmobilenetv1_diana_reduced',
]


def conv3x3(conv_func, hw_model, is_searchable, in_planes, out_planes,
            bias=False, stride=1, groups=1, fix_qtz=False,
            target='latency', **kwargs):
    "3x3 convolution with padding"
    if conv_func != nn.Conv2d:
        if not is_searchable:
            kwargs['wbits'] = [8]
        return conv_func(hw_model, in_planes, out_planes,
                         kernel_size=3, groups=groups, stride=stride,
                         padding=1, bias=bias, fix_qtz=fix_qtz,
                         target=target, **kwargs)
    else:
        return conv_func(in_planes, out_planes,
                         kernel_size=3, groups=groups, stride=stride,
                         padding=1, bias=bias, **kwargs)


def conv_depth(conv_func, hw_model, is_searchable, in_planes,
               bias=False, stride=1, target='latency', **kwargs):
    if conv_func != nn.Conv2d:
        if not is_searchable:
            kwargs['wbits'] = [8]
        return conv_func(hw_model, in_planes, in_planes,
                         kernel_size=3, groups=in_planes, stride=stride,
                         padding=1, bias=bias,
                         target=target, **kwargs)
    else:
        return conv_func(in_planes, in_planes,
                         kernel_size=3, groups=in_planes, stride=stride,
                         padding=1, bias=bias, **kwargs)


def conv_point(conv_func, hw_model, is_searchable, in_planes, out_planes,
               bias=False, stride=1, target='latency', **kwargs):
    if conv_func != nn.Conv2d:
        if not is_searchable:
            kwargs['wbits'] = [8]
        return conv_func(hw_model, in_planes, out_planes,
                         kernel_size=1, groups=1, stride=stride,
                         padding=0, bias=bias,
                         target=target, **kwargs)
    else:
        return conv_func(in_planes, out_planes,
                         kernel_size=1, groups=1, stride=stride,
                         padding=0, bias=bias, **kwargs)


def fc(conv_func, hw_model, is_searchable, in_planes, out_planes,
       stride=1, groups=1, search_fc=None, target='latency', **kwargs):
    "fc mapped to conv"
    if not is_searchable:
        kwargs['wbits'] = [8]
    return conv_func(hw_model, in_planes, out_planes,
                     kernel_size=1, groups=groups, stride=stride,
                     padding=0, bias=True, fc=search_fc,
                     target=target, **kwargs)


def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class BasicBlockGumbel(nn.Module):

    def __init__(self, conv_func, hw_model, is_searchable,
                 inp, oup, stride=1, bn=True,
                 target='latency', **kwargs):
        self.bn = bn
        self.use_bias = not bn
        super().__init__()
        # For depthwise archws is always [8]
        self.depth = conv_depth(conv_func, hw_model, False, inp,
                                bias=self.use_bias, stride=stride,
                                target=target, **kwargs)
        if bn:
            self.bn_depth = nn.BatchNorm2d(inp)
        self.point = conv_point(conv_func, hw_model, is_searchable[1],
                                inp, oup,
                                bias=self.use_bias, stride=1,
                                target=target, **kwargs)
        if bn:
            self.bn_point = nn.BatchNorm2d(oup)

    def forward(self, x, temp, is_hard):
        x = self.depth(x, temp, is_hard)
        if self.bn:
            x = self.bn_depth(x)
        x = self.point(x, temp, is_hard)
        if self.bn:
            x = self.bn_point(x)

        return x


class Backbone(nn.Module):

    def __init__(self, conv_func, hw_model, is_searchable,
                 input_size, bn, width_mult,
                 target='latency', **kwargs):
        self.fp = conv_func is qm.FpConv2d
        super().__init__()
        self.bb_1 = BasicBlockGumbel(
            conv_func, hw_model, is_searchable[:2],
            make_divisible(32*width_mult), make_divisible(64*width_mult),
            stride=1, bn=bn, target=target, **kwargs)
        self.bb_2 = BasicBlockGumbel(
            conv_func, hw_model, is_searchable[2:4],
            make_divisible(64*width_mult), make_divisible(128*width_mult),
            stride=2, bn=bn, target=target, **kwargs)
        self.bb_3 = BasicBlockGumbel(
            conv_func, hw_model, is_searchable[4:6],
            make_divisible(128*width_mult), make_divisible(128*width_mult),
            stride=1, bn=bn, target=target, **kwargs)
        self.bb_4 = BasicBlockGumbel(
            conv_func, hw_model, is_searchable[6:8],
            make_divisible(128*width_mult), make_divisible(256*width_mult),
            stride=2, bn=bn, target=target, **kwargs)
        self.bb_5 = BasicBlockGumbel(
            conv_func, hw_model, is_searchable[8:10],
            make_divisible(256*width_mult), make_divisible(256*width_mult),
            stride=1, bn=bn, target=target, **kwargs)
        self.bb_6 = BasicBlockGumbel(
            conv_func, hw_model, is_searchable[10:12],
            make_divisible(256*width_mult), make_divisible(512*width_mult),
            stride=2, bn=bn, target=target, **kwargs)
        self.bb_7 = BasicBlockGumbel(
            conv_func, hw_model, is_searchable[12:14],
            make_divisible(512*width_mult), make_divisible(512*width_mult),
            stride=1, bn=bn, target=target, **kwargs)
        self.bb_8 = BasicBlockGumbel(
            conv_func, hw_model, is_searchable[14:16],
            make_divisible(512*width_mult), make_divisible(512*width_mult),
            stride=1, bn=bn, target=target, **kwargs)
        self.bb_9 = BasicBlockGumbel(
            conv_func, hw_model, is_searchable[16:18],
            make_divisible(512*width_mult), make_divisible(512*width_mult),
            stride=1, bn=bn, target=target, **kwargs)
        self.bb_10 = BasicBlockGumbel(
            conv_func, hw_model, is_searchable[18:20],
            make_divisible(512*width_mult), make_divisible(512*width_mult),
            stride=1, bn=bn, target=target, **kwargs)
        self.bb_11 = BasicBlockGumbel(
            conv_func, hw_model, is_searchable[20:22],
            make_divisible(512*width_mult), make_divisible(512*width_mult),
            stride=1, bn=bn, target=target, **kwargs)
        self.bb_12 = BasicBlockGumbel(
            conv_func, hw_model, is_searchable[22:24],
            make_divisible(512*width_mult), make_divisible(1024*width_mult),
            stride=2, bn=bn, target=target, **kwargs)
        self.bb_13 = BasicBlockGumbel(
            conv_func, hw_model, is_searchable[24:26],
            make_divisible(1024*width_mult), make_divisible(1024*width_mult),
            stride=1, bn=bn, target=target, **kwargs)
        if not self.fp:
            # If not fp we use quantized pooling
            self.pool = qm2.QuantAvgPool2d(kwargs['abits'],
                                           int(input_size / (2**5)))
        else:
            self.pool = nn.AvgPool2d(int(input_size / (2**5)))

    def forward(self, x, temp, is_hard):
        out = self.bb_1(x, temp, is_hard)
        out = self.bb_2(out, temp, is_hard)
        out = self.bb_3(out, temp, is_hard)
        out = self.bb_4(out, temp, is_hard)
        out = self.bb_5(out, temp, is_hard)
        out = self.bb_6(out, temp, is_hard)
        out = self.bb_7(out, temp, is_hard)
        out = self.bb_8(out, temp, is_hard)
        out = self.bb_9(out, temp, is_hard)
        out = self.bb_10(out, temp, is_hard)
        out = self.bb_11(out, temp, is_hard)
        out = self.bb_12(out, temp, is_hard)
        out = self.bb_13(out, temp, is_hard)
        if self.fp:
            out = F.relu(out)
        out = self.pool(out)
        return out


class MobileNetV1(nn.Module):

    def __init__(self, conv_func, hw_model, is_searchable,
                 search_fc=None, width_mult=.25,
                 input_size=96, num_classes=2, bn=True,
                 target='latency', **kwargs):
        if 'abits' in kwargs:
            print('abits: {}'.format(kwargs['abits']))
        if 'wbits' in kwargs:
            print('wbits: {}'.format(kwargs['wbits']))

        self.conv_func = conv_func
        self.hw_model = hw_model
        self.search_types = ['fixed', 'mixed', 'multi']
        if search_fc in self.search_types:
            self.search_fc = search_fc
        else:
            self.search_fc = False
        self.bn = bn
        self.use_bias = not bn
        super().__init__()
        self.gumbel = kwargs.get('gumbel', False)
        self.target = target

        # Model
        self.input_layer = conv3x3(conv_func, hw_model, False,
                                   3, make_divisible(32*width_mult),
                                   stride=2, groups=1,
                                   bias=False, max_inp_val=1.0,
                                   target=target, **kwargs)
        if bn:
            self.bn = nn.BatchNorm2d(make_divisible(32*width_mult))
        self.backbone = Backbone(conv_func, hw_model, is_searchable[1:-1],
                                 input_size, bn, width_mult,
                                 target=target, **kwargs)

        # Initialize bn and conv weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

        # Final classifier
        self.fc = fc(conv_func, hw_model, False,
                     make_divisible(1024*width_mult), num_classes,
                     search_fc=self.search_fc,
                     target=target, **kwargs)

    def forward(self, x, temp, is_hard):
        x = self.input_layer(x, temp, is_hard)
        if self.bn:
            x = self.bn(x)
        x = self.backbone(x, temp, is_hard)
        x = self.fc(x, temp, is_hard)[:, :, 0, 0]
        return x

    def complexity_loss(self):
        loss = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                loss = loss + m.complexity_loss()
        return loss

    def fetch_best_arch(self):
        sum_cycles, sum_bita, sum_bitw = 0, 0, 0
        sum_mixcycles, sum_mixbita, sum_mixbitw = 0, 0, 0
        layer_idx = 0
        best_arch = None
        for m in self.modules():
            if isinstance(m, self.conv_func):
                outs = m.fetch_best_arch(layer_idx)  # Return tuple
                layer_arch, cycles, bita, bitw, mixcycles, mixbita, mixbitw = outs
                if best_arch is None:
                    best_arch = layer_arch
                else:
                    for key in layer_arch.keys():
                        if key not in best_arch:
                            best_arch[key] = layer_arch[key]
                        else:
                            best_arch[key].append(layer_arch[key][0])
                sum_cycles += cycles
                sum_bita += bita
                sum_bitw += bitw
                sum_mixcycles += mixcycles
                sum_mixbita += mixbita
                sum_mixbitw += mixbitw
                layer_idx += 1
        return best_arch, sum_cycles, sum_bita, sum_bitw, sum_mixcycles, sum_mixbita, sum_mixbitw


def mixmobilenetv1_diana_naive5(arch_cfg_path, **kwargs):
    search_model = MobileNetV1(
        qm.MultiPrecActivConv2d, hw.diana_naive(5.), [True]*28,
        search_fc='multi', wbits=[8, 2], abits=[7], bn=False,
        share_weight=True, **kwargs)
    return _mixmobilenetv1_diana(arch_cfg_path, search_model)


def mixmobilenetv1_diana_naive10(arch_cfg_path, **kwargs):
    search_model = MobileNetV1(
        qm.MultiPrecActivConv2d, hw.diana_naive(10.), [True]*28,
        search_fc='multi', wbits=[8, 2], abits=[7], bn=False,
        share_weight=True, **kwargs)
    return _mixmobilenetv1_diana(arch_cfg_path, search_model)


def mixmobilenetv1_diana_full(arch_cfg_path, **kwargs):
    search_model = MobileNetV1(
        qm.MultiPrecActivConv2d, hw.diana(), [True]*28,
        search_fc='multi', wbits=[8, 2], abits=[7], bn=False,
        share_weight=True, **kwargs)
    return _mixmobilenetv1_diana(arch_cfg_path, search_model)


def mixmobilenetv1_pow2_diana_full(arch_cfg_path, target='latency', **kwargs):
    search_model = MobileNetV1(
        qm2.MultiPrecActivConv2d, hw.diana(), [True]*28,
        search_fc='multi', wbits=[8, 2], abits=[7], bn=False,
        share_weight=True, target=target, **kwargs)
    return _mixmobilenetv1_diana(arch_cfg_path, search_model)


def mixmobilenetv1_diana_reduced(arch_cfg_path, **kwargs):
    is_searchable = utils.detect_ad_tradeoff(  # TODO: to be tested!!
        quantmobilenetv1_fp(None, pretrained=False),
        torch.rand((1, 3, 96, 96)))
    search_model = MobileNetV1(
        qm.MultiPrecActivConv2d, hw.diana(), is_searchable,
        search_fc='multi', wbits=[8, 2], abits=[7], bn=False,
        share_weight=True, **kwargs
    )
    return _mixmobilenetv1_diana(arch_cfg_path, search_model)


def _mixmobilenetv1_diana(arch_cfg_path, search_model):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    # Get folded pretrained model
    folded_fp_model = quantmobilenetv1_fp_foldbn(arch_cfg_path)
    folded_state_dict = folded_fp_model.state_dict()

    # Delete folded model
    del folded_fp_model

    # Translate folded state dict in a format compatible with searchable layers
    search_state_dict = utils.fpfold_to_q(folded_state_dict)
    search_model.load_state_dict(search_state_dict, strict=False)

    # Init quantization scale param
    utils.init_scale_param(search_model)

    return search_model
