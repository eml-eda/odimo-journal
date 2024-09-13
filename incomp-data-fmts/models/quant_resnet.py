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

import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from . import utils
from . import quant_module as qm
from . import quant_module_pow2 as qm2
from . import hw_models as hw

# MR
__all__ = [
    'quantres8_fp', 'quantres8_fp_foldbn',
    'quantres20_fp', 'quantres20_fp_foldbn',
    'quantres8_w8a8', 'quantres8_w8a8_nobn',
    'quantres8_w8a8_foldbn',
    'quantres8_w8a8_pretrained', 'quantres8_w8a8_nobn_pretrained',
    'quantres20_w8a8', 'quantres20_w8a8_foldbn',
    'quantres20_w8a7_foldbn',
    'quantres20_w8a7_pow2_foldbn',
    'quantres8_w8a7_foldbn',
    'quantres8_w5a8',
    'quantres8_w2a8', 'quantres8_w2a8_nobn',
    'quantres8_w2a8_pretrained', 'quantres8_w2a8_nobn_pretrained',
    'quantres20_w2a8', 'quantres20_w2a8_foldbn',
    'quantres20_w2a7_foldbn', 'quantres20_w2a7_pow2_foldbn',
    'quantres8_w2a8_true', 'quantres8_w2a8_true_nobn',
    'quantres8_w2a8_foldbn', 'quantres8_w2a8_foldbn_test',
    'quantres8_w2a7_foldbn',
    'quantres8_w2a8_true_pretrained', 'quantres8_w2a8_true_nobn_pretrained',
    'quantres8_w2a8_true_foldbn',
    'quantres8_w2a7_true_foldbn',
    'quantres20_w2a8_true',
    'quantres20_w2a7_true_foldbn', 'quantres20_w2a7_true_pow2_foldbn',
    'quantres20_minlat_foldbn',
    'quantres20_minlat_max8_foldbn', 'quantres20_minlat_max8_pow2_foldbn',
    'quantres8_diana',
    'quantres8_diana_naive5', 'quantres8_diana_naive10', 'quantres8_diana_naive100',
    'quantres20_minlat_naive_foldbn',
    'quantres20_diana_naive5', 'quantres20_diana_naive10',
    'quantres20_diana_reduced', 'quantres20_diana_full', 'quantres20_pow2_diana_full',
    'quantres18_fp', 'quantres18_fp_reduced', 'quantres18_fp_prtrext', 'quantres18_fp_foldbn',
    'quantres18_w8a7_foldbn', 'quantres18_w8a7_pow2_foldbn',
    'quantres18_w2a7_foldbn', 'quantres18_w2a7_pow2_foldbn',
    'quantres18_w2a7_true_foldbn', 'quantres18_w2a7_true_pow2_foldbn',
    'quantres18_minlat64_foldbn',
    'quantres18_minlat64_max8_foldbn', 'quantres18_minlat64_max8_pow2_foldbn',
    'quantres18_minlat64_naive5_foldbn', 'quantres18_minlat64_naive10_foldbn',
    'quantres18_diana_naive5', 'quantres18_diana_naive10', 'quantres18_pow2_diana_naive10',
    'quantres18_diana_reduced', 'quantres18_diana_full', 'quantres18_pow2_diana_full',
    'quantres18_fp_c100', 'quantres18_w8a7_pow2_foldbn_c100', 'quantres18_w2a7_pow2_foldbn_c100',
    'quantres18_w2a7_true_pow2_foldbn_c100', 'quantres18_pow2_diana_full_c100', 'quantres18_pow2_diana_full_c100_no1st',
    'quantres18_minlat_max8_pow2_foldbn_c100',
    'quantres18_minlat_max8_foldbn',
]


# MR
class Backbone18(nn.Module):
    def __init__(self, conv_func, input_size, bn, abits, wbits, std_head=True, **kwargs):
        self.fp = conv_func is qm.FpConv2d
        super().__init__()
        self.std_head = std_head
        if std_head:
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bb_1_0 = BasicBlock(conv_func, 64, 64, wbits[:2], abits[:2], stride=1,
                                 bn=bn, **kwargs)
        self.bb_1_1 = BasicBlock(conv_func, 64, 64, wbits[2:4], abits[2:4], stride=1,
                                 bn=bn, **kwargs)
        self.bb_2_0 = BasicBlock(conv_func, 64, 128, wbits[4:7], abits[4:7], stride=2,
                                 bn=bn, **kwargs)
        self.bb_2_1 = BasicBlock(conv_func, 128, 128, wbits[7:9], abits[7:9], stride=1,
                                 bn=bn, **kwargs)
        self.bb_3_0 = BasicBlock(conv_func, 128, 256, wbits[9:12], abits[9:12], stride=2,
                                 bn=bn, **kwargs)
        self.bb_3_1 = BasicBlock(conv_func, 256, 256, wbits[12:14], abits[12:14], stride=1,
                                 bn=bn, **kwargs)
        self.bb_4_0 = BasicBlock(conv_func, 256, 512, wbits[12:15], abits[12:15], stride=2,
                                 bn=bn, **kwargs)
        self.bb_4_1 = BasicBlock(conv_func, 512, 512, wbits[15:17], abits[15:17], stride=1,
                                 bn=bn, **kwargs)
        if not self.fp:
            # Use quantized pooling
            # self.avg_pool = qm2.QuantAvgPool2d(abits[0], 7)
            self.avg_pool = qm2.QuantAvgPool2d(abits[0], 4)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        if self.std_head:
            x = self.max_pool(F.relu(x))
        x = self.bb_1_0(x)
        x = self.bb_1_1(x)
        x = self.bb_2_0(x)
        x = self.bb_2_1(x)
        x = self.bb_3_0(x)
        x = self.bb_3_1(x)
        x = self.bb_4_0(x)
        out = self.bb_4_1(x)
        if self.fp:
            out = F.relu(out)
        out = self.avg_pool(out)
        return out


# MR
class Backbone20(nn.Module):
    def __init__(self, conv_func, input_size, bn, abits, wbits, **kwargs):
        self.fp = conv_func is qm.FpConv2d
        super().__init__()
        self.bb_1_0 = BasicBlock(conv_func, 16, 16, wbits[:2], abits[:2], stride=1,
                                 bn=bn, **kwargs)
        self.bb_1_1 = BasicBlock(conv_func, 16, 16, wbits[2:4], abits[2:4], stride=1,
                                 bn=bn, **kwargs)
        self.bb_1_2 = BasicBlock(conv_func, 16, 16, wbits[4:6], abits[4:6], stride=1,
                                 bn=bn, **kwargs)
        self.bb_2_0 = BasicBlock(conv_func, 16, 32, wbits[6:9], abits[6:9], stride=2,
                                 bn=bn, **kwargs)
        self.bb_2_1 = BasicBlock(conv_func, 32, 32, wbits[9:11], abits[9:11], stride=1,
                                 bn=bn, **kwargs)
        self.bb_2_2 = BasicBlock(conv_func, 32, 32, wbits[11:13], abits[11:13], stride=1,
                                 bn=bn, **kwargs)
        self.bb_3_0 = BasicBlock(conv_func, 32, 64, wbits[13:16], abits[13:16], stride=2,
                                 bn=bn, **kwargs)
        self.bb_3_1 = BasicBlock(conv_func, 64, 64, wbits[16:18], abits[16:18], stride=1,
                                 bn=bn, **kwargs)
        self.bb_3_2 = BasicBlock(conv_func, 64, 64, wbits[18:20], abits[18:20], stride=1,
                                 bn=bn, **kwargs)
        if not self.fp:
            # If not fp we use quantized pooling
            self.pool = qm2.QuantAvgPool2d(abits[0], 8)
        else:
            self.pool = nn.AvgPool2d(8)

    def forward(self, x_inp):
        x = self.bb_1_0(x_inp)
        x = self.bb_1_1(x)
        x = self.bb_1_2(x)
        x = self.bb_2_0(x)
        x = self.bb_2_1(x)
        x = self.bb_2_2(x)
        x = self.bb_3_0(x)
        x = self.bb_3_1(x)
        out = self.bb_3_2(x)
        if self.fp:
            out = F.relu(out)
        out = self.pool(out)
        return out


# MR
class BackboneTiny(nn.Module):
    def __init__(self, conv_func, input_size, bn, abits, wbits, **kwargs):
        super().__init__()
        self.bb_1 = BasicBlock(conv_func, 16, 16, wbits[:2], abits[:2], stride=1,
                               bn=bn, **kwargs)
        self.bb_2 = BasicBlock(conv_func, 16, 32, wbits[2:5], abits[2:5], stride=2,
                               bn=bn, **kwargs)
        self.bb_3 = BasicBlock(conv_func, 32, 64, wbits[5:7], abits[5:7], stride=2,
                               bn=bn, **kwargs)
        self.pool = nn.AvgPool2d(kernel_size=8)

    def forward(self, x):
        x = self.bb_1(x)
        x = self.bb_2(x)
        x = self.bb_3(x)
        x = self.pool(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, conv_func, inplanes, planes, archws, archas, stride=1,
                 downsample=None, bn=True, **kwargs):
        self.bn = bn
        self.use_bias = not bn
        self.fp = conv_func is qm.FpConv2d
        super().__init__()
        self.conv1 = conv_func(inplanes, planes, archws[0], archas[0], kernel_size=3, stride=stride,
                               groups=1, padding=1, bias=self.use_bias, **kwargs)
        if bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_func(planes, planes, archws[1], archas[1], kernel_size=3, stride=1,
                               groups=1, padding=1, bias=self.use_bias, **kwargs)
        if bn:
            self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or inplanes != planes:
            self.downsample = conv_func(
                inplanes, planes, archws[-1], archas[-1], kernel_size=1,
                groups=1, stride=stride, bias=self.use_bias, **kwargs)
            if bn:
                self.bn_ds = nn.BatchNorm2d(planes)
            if not self.fp:
                # Couple input clip_val
                inp_clip_val = self.conv1.mix_activ.mix_activ[0].clip_val
                self.downsample.mix_activ.mix_activ[0].clip_val = inp_clip_val

                # Quantized Sum node
                self.qadd = qm2.QuantAdd(archas[0])
        else:
            self.downsample = None
            if not self.fp:
                # If not fp and no downsample op we need to quantize the residual branch
                self.inp_q = qm2.QuantPaCTActiv(archas[0], round_pow2=True)
                # Couple input clip_val
                inp_clip_val = self.conv1.mix_activ.mix_activ[0].clip_val
                self.inp_q.mix_activ[0].clip_val = inp_clip_val

                # Quantized Sum node
                self.qadd = qm2.QuantAdd(archas[0], clip_val=inp_clip_val)

    def forward(self, x):
        if self.downsample is not None:
            residual = x
        else:
            if self.fp:
                residual = F.relu(x)
            else:
                # Here I need to quantize
                residual, _ = self.inp_q(x)

        out = self.conv1(x)
        if self.bn:
            out = self.bn1(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)
            if self.bn:
                residual = self.bn_ds(residual)

        if self.fp:
            out += residual
        else:
            out = self.qadd(out, residual)

        return out


class ResNet18(nn.Module):

    def __init__(self, conv_func, hw_model, archws, archas, qtz_fc=None,
                 input_size=224, num_classes=1000, bn=True, std_head=True,
                 target='latency', **kwargs):
        print('archas: {}'.format(archas))
        print('archws: {}'.format(archws))

        self.inplanes = 64
        self.conv_func = conv_func
        self.hw_model = hw_model
        self.search_types = ['fixed', 'mixed', 'multi']
        if qtz_fc in self.search_types:
            self.qtz_fc = qtz_fc
        else:
            self.qtz_fc = False
        self.bn = bn
        self.use_bias = not bn
        self.target = target
        if target == 'latency':
            self.fetch_arch_info = self._fetch_arch_latency
        elif target == 'power':
            self.power = hw.DianaPower()
            self.fetch_arch_info = self._fetch_arch_power
        else:
            raise ValueError('Use "latency" or "power" as target.')
        super().__init__()

        # Model
        if std_head:
            self.conv1 = conv_func(3, 64, abits=archas[0], wbits=archws[0],
                                   kernel_size=7, stride=2, bias=self.use_bias, padding=3,
                                   groups=1, first_layer=False,
                                   max_inp_val=1.0, **kwargs)
            # N.B., I modified first_layer to False
        else:
            self.conv1 = conv_func(3, 64, abits=archas[0], wbits=archws[0],
                                   kernel_size=3, stride=1, bias=self.use_bias, padding=1,
                                   groups=1, first_layer=False,
                                   max_inp_val=1.0, **kwargs)
        if bn:
            self.bn1 = nn.BatchNorm2d(64)
        self.backbone = Backbone18(
            conv_func, input_size, bn, abits=archas[1:-1], wbits=archws[1:-1],
            std_head=std_head, **kwargs)

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
        self.fc = conv_func(
            512, num_classes, abits=archas[-1], wbits=archws[-1],
            kernel_size=1, stride=1, groups=1, bias=True, fc=self.qtz_fc, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        if self.bn:
            x = self.bn1(x)

        x = self.backbone(x)

        x = x if self.qtz_fc else x.view(x.size(0), -1)
        x = self.fc(x)[:, :, 0, 0] if self.qtz_fc else self.fc(x)

        return x

    def harden_weights(self, dequantize=False):
        for _, module in self.named_modules():
            if isinstance(module, self.conv_func):
                module.harden_weights(dequantize=dequantize)

    def store_hardened_weights(self):
        with torch.no_grad():
            for _, module in self.named_modules():
                if isinstance(module, self.conv_func):
                    module.store_hardened_weights()

    def _fetch_arch_latency(self):
        sum_cycles, sum_bita, sum_bitw = 0, 0, 0
        layer_idx = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                size_product = m.size_product.item()
                memory_size = m.memory_size.item()
                wbit = m.wbits[0]
                abit = m.abits[0]
                sum_bitw += size_product * wbit

                cycles_analog, cycles_digital = 0, 0
                for idx, wb in enumerate(m.wbits):
                    if len(m.wbits) > 1:
                        ch_out = m.mix_weight.alpha_weight[idx].sum()
                    else:
                        ch_out = torch.tensor(m.ch_out)
                    # Define dict whit shape infos used to model accelerators perf
                    conv_shape = {
                        'ch_in': m.ch_in,
                        'ch_out': ch_out,
                        'groups': m.mix_weight.conv.groups,
                        'k_x': m.k_x,
                        'k_y': m.k_y,
                        'out_x': m.out_x,
                        'out_y': m.out_y,
                        }
                    if wb == 2:
                        cycles_analog = self.hw_model('analog', **conv_shape)
                    else:
                        cycles_digital = self.hw_model('digital', **conv_shape)
                if m.mix_weight.conv.groups == 1:
                    cycles = max(cycles_analog, cycles_digital)
                else:
                    cycles = cycles_digital

                bita = memory_size * abit
                bitw = m.param_size * wbit
                sum_cycles += cycles
                sum_bita += bita
                sum_bitw += bitw
                layer_idx += 1
        return sum_cycles, sum_bita, sum_bitw

    def _fetch_arch_power(self):
        sum_power, sum_bita, sum_bitw = 0, 0, 0
        layer_idx = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                size_product = m.size_product.item()
                memory_size = m.memory_size.item()
                wbit = m.wbits[0]
                abit = m.abits[0]
                sum_bitw += size_product * wbit

                cycles_analog, cycles_digital = 0, 0
                for idx, wb in enumerate(m.wbits):
                    if len(m.wbits) > 1:
                        ch_out = m.mix_weight.alpha_weight[idx].sum()
                    else:
                        ch_out = torch.tensor(m.ch_out)
                    # Define dict whit shape infos used to model accelerators perf
                    conv_shape = {
                        'ch_in': m.ch_in,
                        'ch_out': ch_out,
                        'groups': m.mix_weight.conv.groups,
                        'k_x': m.k_x,
                        'k_y': m.k_y,
                        'out_x': m.out_x,
                        'out_y': m.out_y,
                        }
                    if wb == 2:
                        cycles_analog = self.hw_model('analog', **conv_shape)
                    else:
                        cycles_digital = self.hw_model('digital', **conv_shape)
                if m.mix_weight.conv.groups == 1:
                    min_cycles = min(cycles_analog, cycles_digital)
                    power = (self.power.p_hyb * min_cycles) + \
                            (self.power.p_ana * (cycles_analog - min_cycles)) + \
                            (self.power.p_dig * (cycles_digital - min_cycles))
                else:
                    power = self.power.p_dig * cycles_digital

                bita = memory_size * abit
                bitw = m.param_size * wbit
                sum_power += power
                sum_bita += bita
                sum_bitw += bitw
                layer_idx += 1
        return sum_power, sum_bita, sum_bitw


class ResNet20(nn.Module):

    def __init__(self, conv_func, hw_model, archws, archas, qtz_fc=None,
                 input_size=32, num_classes=10, bn=True,
                 target='latency', **kwargs):
        print('archas: {}'.format(archas))
        print('archws: {}'.format(archws))

        self.inplanes = 16
        self.conv_func = conv_func
        self.hw_model = hw_model
        self.search_types = ['fixed', 'mixed', 'multi']
        if qtz_fc in self.search_types:
            self.qtz_fc = qtz_fc
        else:
            self.qtz_fc = False
        self.bn = bn
        self.use_bias = not bn
        self.target = target
        if target == 'latency':
            self.fetch_arch_info = self._fetch_arch_latency
        elif target == 'power':
            self.power = hw.DianaPower()
            self.fetch_arch_info = self._fetch_arch_power
        else:
            raise ValueError('Use "latency" or "power" as target.')
        super().__init__()

        # Model
        self.conv1 = conv_func(3, 16, abits=archas[0], wbits=archws[0],
                               kernel_size=3, stride=1, bias=self.use_bias, padding=1,
                               groups=1, first_layer=False,
                               max_inp_val=1.0, **kwargs)
        if bn:
            self.bn1 = nn.BatchNorm2d(16)
        self.backbone = Backbone20(
            conv_func, input_size, bn, abits=archas[1:-1], wbits=archws[1:-1], **kwargs)

        # Initialize weights
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
        self.fc = conv_func(
            64, num_classes, abits=archas[-1], wbits=archws[-1],
            kernel_size=1, stride=1, groups=1, bias=True, fc=self.qtz_fc, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        if self.bn:
            x = self.bn1(x)

        x = self.backbone(x)

        x = x if self.qtz_fc else x.view(x.size(0), -1)
        x = self.fc(x)[:, :, 0, 0] if self.qtz_fc else self.fc(x)

        return x

    def harden_weights(self, dequantize=False):
        for _, module in self.named_modules():
            if isinstance(module, self.conv_func):
                module.harden_weights(dequantize=dequantize)

    def store_hardened_weights(self):
        with torch.no_grad():
            for _, module in self.named_modules():
                if isinstance(module, self.conv_func):
                    module.store_hardened_weights()

    def _fetch_arch_latency(self):
        sum_cycles, sum_bita, sum_bitw = 0, 0, 0
        layer_idx = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                size_product = m.size_product.item()
                memory_size = m.memory_size.item()
                wbit = m.wbits[0]
                abit = m.abits[0]
                sum_bitw += size_product * wbit

                cycles_analog, cycles_digital = 0, 0
                for idx, wb in enumerate(m.wbits):
                    if len(m.wbits) > 1:
                        ch_out = m.mix_weight.alpha_weight[idx].sum()
                    else:
                        ch_out = torch.tensor(m.ch_out)
                    # Define dict whit shape infos used to model accelerators perf
                    conv_shape = {
                        'ch_in': m.ch_in,
                        'ch_out': ch_out,
                        'groups': m.mix_weight.conv.groups,
                        'k_x': m.k_x,
                        'k_y': m.k_y,
                        'out_x': m.out_x,
                        'out_y': m.out_y,
                        }
                    if wb == 2:
                        cycles_analog = self.hw_model('analog', **conv_shape)
                    else:
                        cycles_digital = self.hw_model('digital', **conv_shape)
                if m.mix_weight.conv.groups == 1:
                    cycles = max(cycles_analog, cycles_digital)
                else:
                    cycles = cycles_digital

                bita = memory_size * abit
                bitw = m.param_size * wbit
                sum_cycles += cycles
                sum_bita += bita
                sum_bitw += bitw
                layer_idx += 1
        return sum_cycles, sum_bita, sum_bitw

    def _fetch_arch_power(self):
        sum_power, sum_bita, sum_bitw = 0, 0, 0
        layer_idx = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                size_product = m.size_product.item()
                memory_size = m.memory_size.item()
                wbit = m.wbits[0]
                abit = m.abits[0]
                sum_bitw += size_product * wbit

                cycles_analog, cycles_digital = 0, 0
                for idx, wb in enumerate(m.wbits):
                    if len(m.wbits) > 1:
                        ch_out = m.mix_weight.alpha_weight[idx].sum()
                    else:
                        ch_out = torch.tensor(m.ch_out)
                    # Define dict whit shape infos used to model accelerators perf
                    conv_shape = {
                        'ch_in': m.ch_in,
                        'ch_out': ch_out,
                        'groups': m.mix_weight.conv.groups,
                        'k_x': m.k_x,
                        'k_y': m.k_y,
                        'out_x': m.out_x,
                        'out_y': m.out_y,
                        }
                    if wb == 2:
                        cycles_analog = self.hw_model('analog', **conv_shape)
                    else:
                        cycles_digital = self.hw_model('digital', **conv_shape)
                if m.mix_weight.conv.groups == 1:
                    min_cycles = min(cycles_analog, cycles_digital)
                    power = (self.power.p_hyb * min_cycles) + \
                            (self.power.p_ana * (cycles_analog - min_cycles)) + \
                            (self.power.p_dig * (cycles_digital - min_cycles))
                else:
                    power = self.power.p_dig * cycles_digital

                bita = memory_size * abit
                bitw = m.param_size * wbit
                sum_power += power
                sum_bita += bita
                sum_bitw += bitw
                layer_idx += 1
        return sum_power, sum_bita, sum_bitw


class TinyMLResNet(nn.Module):

    def __init__(self, conv_func, hw_model, archws, archas, qtz_fc=None,
                 input_size=32, num_classes=10, bn=True, **kwargs):
        print('archas: {}'.format(archas))
        print('archws: {}'.format(archws))

        self.inplanes = 16
        self.conv_func = conv_func
        self.hw_model = hw_model
        self.search_types = ['fixed', 'mixed', 'multi']
        if qtz_fc in self.search_types:
            self.qtz_fc = qtz_fc
        else:
            self.qtz_fc = False
        self.bn = bn
        self.use_bias = not bn
        super().__init__()

        # Model
        self.conv1 = conv_func(3, 16, abits=archas[0], wbits=archws[0],
                               kernel_size=3, stride=1, bias=self.use_bias, padding=1,
                               groups=1, first_layer=False, **kwargs)
        if bn:
            self.bn1 = nn.BatchNorm2d(16)
        self.backbone = BackboneTiny(
            conv_func, input_size, bn, abits=archas[1:-1], wbits=archws[1:-1], **kwargs)
        self.fc = conv_func(
            64, num_classes, abits=archas[-1], wbits=archws[-1],
            kernel_size=1, stride=1, groups=1, bias=True, fc=self.qtz_fc, **kwargs)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        if self.bn:
            x = self.bn1(x)

        x = self.backbone(x)

        x = x if self.qtz_fc else x.view(x.size(0), -1)
        x = self.fc(x)[:, :, 0, 0] if self.qtz_fc else self.fc(x)

        return x

    def fetch_arch_info(self):
        sum_cycles, sum_bita, sum_bitw = 0, 0, 0
        layer_idx = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                size_product = m.size_product.item()
                memory_size = m.memory_size.item()
                wbit = m.wbits[0]
                abit = m.abits[0]
                sum_bitw += size_product * wbit

                # Define dict whit shape infos used to model accelerators perf
                conv_shape = {
                    'ch_in': m.ch_in,
                    'ch_out': torch.tensor(m.ch_out),
                    'k_x': m.k_x,
                    'k_y': m.k_y,
                    'out_x': m.out_x,
                    'out_y': m.out_y,
                    }
                if wbit == 2:
                    cycles = self.hw_model('analog', **conv_shape)
                else:
                    cycles = self.hw_model('digital', **conv_shape)

                bita = memory_size * abit
                bitw = m.param_size * wbit
                sum_cycles += cycles
                sum_bita += bita
                sum_bitw += bitw
                layer_idx += 1
        return sum_cycles, sum_bita, sum_bitw


def _load_arch(arch_path, names_nbits):
    checkpoint = torch.load(arch_path)
    state_dict = checkpoint['state_dict']
    best_arch, worst_arch = {}, {}
    for name in names_nbits.keys():
        best_arch[name], worst_arch[name] = [], []
    for name, params in state_dict.items():
        name = name.split('.')[-1]
        if name in names_nbits.keys():
            alpha = params.cpu().numpy()
            assert names_nbits[name] == alpha.shape[0]
            best_arch[name].append(alpha.argmax())
            worst_arch[name].append(alpha.argmin())

    return best_arch, worst_arch


# MR
def _load_arch_multi_prec(arch_path):
    checkpoint = torch.load(arch_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    best_arch, worst_arch = {}, {}
    best_arch['alpha_activ'], worst_arch['alpha_activ'] = [], []
    best_arch['alpha_weight'], worst_arch['alpha_weight'] = [], []
    for name, params in state_dict.items():
        name = name.split('.')[-1]
        if name == 'alpha_activ':
            alpha = params.cpu().numpy()
            best_arch[name].append(alpha.argmax())
            worst_arch[name].append(alpha.argmin())
        elif name == 'alpha_weight':
            alpha = params.cpu().numpy()
            best_arch[name].append(alpha.argmax(axis=0))
            worst_arch[name].append(alpha.argmin(axis=0))

    return best_arch, worst_arch


# MR
def _load_weights(arch_path):
    checkpoint = torch.load(arch_path)
    state_dict = checkpoint['state_dict']
    weights = {}
    for name, params in state_dict.items():
        type_ = name.split('.')[-1]
        if type_ == 'weight':
            weight = params.cpu().numpy()
            weights[name] = weight
        elif name == 'bias':
            bias = params.cpu().numpy()
            weights[name] = bias

    return weights


# MR
def _load_alpha_state_dict(arch_path):
    checkpoint = torch.load(arch_path)
    state_dict = checkpoint['state_dict']
    alpha_state_dict = dict()
    for name, params in state_dict.items():
        full_name = name
        name = name.split('.')[-1]
        if name == 'alpha_activ' or name == 'alpha_weight':
            alpha_state_dict[full_name] = params

    return alpha_state_dict


# MR
def _load_alpha_state_dict_as_mp(arch_path, model):
    checkpoint = torch.load(arch_path)
    state_dict = checkpoint['state_dict']
    alpha_state_dict = dict()
    for name, params in state_dict.items():
        full_name = name
        name = name.split('.')[-1]
        if name == 'alpha_activ':
            alpha_state_dict[full_name] = params
        elif name == 'alpha_weight':
            mp_params = torch.tensor(model.state_dict()[full_name])
            mp_params[0] = params[0]
            mp_params[1] = params[1]
            mp_params[2] = params[2]
            alpha_state_dict[full_name] = mp_params

    return alpha_state_dict


# MR
def _remove_alpha(state_dict):
    weight_state_dict = copy.deepcopy(state_dict)
    for name, params in state_dict.items():
        full_name = name
        name = name.split('.')[-1]
        if name == 'alpha_activ':
            weight_state_dict.pop(full_name)
        elif name == 'alpha_weight':
            weight_state_dict.pop(full_name)

    return weight_state_dict


def quantres8_fp(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 10, [[8]] * 10
    model = TinyMLResNet(qm.FpConv2d, hw.diana(analog_speedup=5.),
                         archws, archas, qtz_fc='multi', **kwargs)
    return model


def quantres20_fp(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 22, [[8]] * 22
    model = ResNet20(qm.FpConv2d, hw.diana(analog_speedup=5.),
                     archws, archas, qtz_fc='multi', **kwargs)
    return model


def quantres18_fp_c100(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 21, [[8]] * 21
    kwargs['std_head'] = False
    model = ResNet18(qm.FpConv2d, hw.diana(analog_speedup=5.),
                     archws, archas, qtz_fc='multi', num_classes=100,
                     **kwargs)
    return model


def quantres18_w8a7_pow2_foldbn_c100(arch_cfg_path, target='latency', **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 21, [[8]] * 21
    s_up = kwargs.pop('analog_speedup', 5.)
    std_head = kwargs.pop('std_head', False)
    fp_model = ResNet18(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, num_classes=100,
                        qtz_fc='multi', std_head=std_head, **kwargs)
    q_model = ResNet18(qm2.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                       archws, archas, num_classes=100,
                       qtz_fc='multi', bn=False, std_head=std_head,
                       target=target, **kwargs)
    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['model_state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres18_w2a7_pow2_foldbn_c100(arch_cfg_path, target='latency', **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 21, [[2]] * 21
    # Set first and last layer weights precision to 8bit
    archws[0] = [8]
    archws[-1] = [8]
    s_up = kwargs.pop('analog_speedup', 5.)
    std_head = kwargs.pop('std_head', False)
    fp_model = ResNet18(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, num_classes=100,
                        qtz_fc='multi', std_head=std_head, **kwargs)
    q_model = ResNet18(qm2.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                       archws, archas, num_classes=100,
                       qtz_fc='multi', bn=False, std_head=std_head,
                       target=target, **kwargs)
    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['model_state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres18_w2a7_true_pow2_foldbn_c100(arch_cfg_path, target='latency', **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 21, [[2]] * 21
    s_up = kwargs.pop('analog_speedup', 5.)
    std_head = kwargs.pop('std_head', False)
    fp_model = ResNet18(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, num_classes=100,
                        qtz_fc='multi', std_head=std_head, **kwargs)
    q_model = ResNet18(qm2.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                       archws, archas, num_classes=100,
                       qtz_fc='multi', bn=False, std_head=std_head,
                       target=target, **kwargs)
    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['model_state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres18_fp(arch_cfg_path, pretrained=True, **kwargs):
    archas, archws = [[8]] * 21, [[8]] * 21
    pretrained_model = torchvision.models.resnet18(pretrained=pretrained)
    model = ResNet18(qm.FpConv2d, hw.diana(analog_speedup=5.),
                     archws, archas, qtz_fc='multi', **kwargs)
    if kwargs.get('std_head', True):
        state_dict = utils.adapt_resnet18_statedict(
            pretrained_model.state_dict(), model.state_dict())
    else:
        state_dict = utils.adapt_resnet18_statedict(
            pretrained_model.state_dict(), model.state_dict(), skip_inp=True)
    model.load_state_dict(state_dict, strict=False)

    # model_1 = torchvision.models.resnet18(pretrained=pretrained)
    # num_ftrs = model_1.fc.in_features
    # model_1.fc = nn.Linear(num_ftrs, 200)

    # m = model
    # mr = model_1

    return model


def quantres18_fp_reduced(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 21, [[8]] * 21
    std_head = kwargs.pop('std_head', False)
    model = ResNet18(qm.FpConv2d, hw.diana(analog_speedup=5.),
                     archws, archas, qtz_fc='multi', std_head=std_head, **kwargs)
    # Check `arch_cfg_path` existence
    if arch_cfg_path is not None:
        if Path(arch_cfg_path).exists():
            # print(f"The file {arch_cfg_path} does not exist.")
            # raise FileNotFoundError
            fp224_state_dict = torch.load(arch_cfg_path)['state_dict']
            state_dict = utils.adapt_resnet18_statedict(
                fp224_state_dict, model.state_dict(), skip_inp=True)
            model.load_state_dict(state_dict, strict=False)

    return model


def quantres18_fp_prtrext(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 21, [[8]] * 21
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError
    fp224_state_dict = torch.load(arch_cfg_path)['state_dict']
    model = ResNet18(qm.FpConv2d, hw.diana(analog_speedup=5.),
                     archws, archas, qtz_fc='multi', **kwargs)
    state_dict = utils.adapt_resnet18_statedict(
        fp224_state_dict, model.state_dict())
    model.load_state_dict(state_dict, strict=False)
    return model


def quantres8_fp_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[8]] * 10, [[8]] * 10
    model = TinyMLResNet(qm.FpConv2d, None,
                         archws, archas, qtz_fc='multi', **kwargs)
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    model.load_state_dict(fp_state_dict)

    model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(model)
    folded_model.train()  # Put folded model in train mode

    return folded_model


def quantres20_fp_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[8]] * 22, [[8]] * 22
    model = ResNet20(qm.FpConv2d, None,
                     archws, archas, qtz_fc='multi', **kwargs)
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    model.load_state_dict(fp_state_dict)

    model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(model)
    folded_model.train()  # Put folded model in train mode

    return folded_model


def quantres18_fp_foldbn(arch_cfg_path, std_head=True, **kwargs):
    pretrained_model = torchvision.models.resnet18(pretrained=True)
    archas, archws = [[8]] * 21, [[8]] * 21
    model = ResNet18(qm.FpConv2d, None,
                     archws, archas, qtz_fc='multi', std_head=std_head,
                     num_classes=1000,
                     **kwargs)
    fp_state_dict = utils.adapt_resnet18_statedict(
            pretrained_model.state_dict(), model.state_dict())
    model.load_state_dict(fp_state_dict, strict=False)

    model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(model)
    folded_model.train()  # Put folded model in train mode

    return folded_model


def quantres18_fp_foldbn_c100(arch_cfg_path, std_head=False, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[8]] * 21, [[8]] * 21
    model = ResNet18(qm.FpConv2d, None,
                     archws, archas, qtz_fc='multi', std_head=std_head,
                     num_classes=100,
                     **kwargs)
    fp_state_dict = torch.load(arch_cfg_path)['model_state_dict']
    model.load_state_dict(fp_state_dict)
    model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(model)
    folded_model.train()  # Put folded model in train mode

    return folded_model


def quantres8_w8a8(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 10, [[8]] * 10
    s_up = kwargs.pop('analog_speedup', 5.)
    model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                         archws, archas, qtz_fc='multi', **kwargs)
    return model


def quantres20_w8a8(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 22, [[8]] * 22
    s_up = kwargs.pop('analog_speedup', 5.)
    model = ResNet20(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                     archws, archas, qtz_fc='multi', **kwargs)
    return model


def quantres8_w8a8_pretrained(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[8]] * 10, [[8]] * 10
    s_up = kwargs.pop('analog_speedup', 5.)
    q_model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                           archws, archas, qtz_fc='multi', **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']

    # Load fp bn params in quantized model
    q_model.load_state_dict(fp_state_dict, strict=False)
    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fp_to_q(fp_state_dict)
    # Load fp weights in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres8_w8a8_nobn(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 10, [[8]] * 10
    s_up = kwargs.pop('analog_speedup', 5.)
    model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                         archws, archas, qtz_fc='multi', bn=False, **kwargs)
    return model


def quantres8_w8a8_nobn_pretrained(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[8]] * 10, [[8]] * 10
    s_up = kwargs.pop('analog_speedup', 5.)
    q_model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                           archws, archas, qtz_fc='multi', bn=False, **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fp_to_q(fp_state_dict)
    # Load fp weights in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres8_w8a8_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[8]] * 10, [[8]] * 10
    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = TinyMLResNet(qm.FpConv2d, hw.diana(analog_speedup=5.),
                            archws, archas, qtz_fc='multi', **kwargs)
    q_model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                           archws, archas, qtz_fc='multi', bn=False, **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres20_w8a8_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[8]] * 22, [[8]] * 22
    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = ResNet20(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, qtz_fc='multi', **kwargs)
    q_model = ResNet20(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                       archws, archas, qtz_fc='multi', bn=False, **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres8_w8a7_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 10, [[8]] * 10
    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = TinyMLResNet(qm.FpConv2d, hw.diana(analog_speedup=5.),
                            archws, archas, qtz_fc='multi', **kwargs)
    q_model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                           archws, archas, qtz_fc='multi', bn=False, **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres20_w8a7_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 22, [[8]] * 22
    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = ResNet20(qm.FpConv2d, hw.diana(analog_speedup=s_up),
                        archws, archas, qtz_fc='multi', **kwargs)
    q_model = ResNet20(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                       archws, archas, qtz_fc='multi', bn=False, **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres20_w8a7_pow2_foldbn(arch_cfg_path, target='latency', **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 22, [[8]] * 22
    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = ResNet20(qm.FpConv2d, hw.diana(analog_speedup=s_up),
                        archws, archas, qtz_fc='multi', **kwargs)
    q_model = ResNet20(qm2.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                       archws, archas, qtz_fc='multi', bn=False,
                       target=target, **kwargs)
    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres18_w8a7_foldbn(arch_cfg_path, target='latency', **kwargs):
    pretrained_model = torchvision.models.resnet18(pretrained=True)

    archas, archws = [[7]] * 21, [[8]] * 21
    s_up = kwargs.pop('analog_speedup', 5.)
    std_head = kwargs.pop('std_head', True)
    fp_model = ResNet18(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, qtz_fc='multi', std_head=std_head, **kwargs)
    q_model = ResNet18(qm2.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                       archws, archas, qtz_fc='multi', bn=False, std_head=std_head,
                       target=target, **kwargs)
    # Load pretrained fp state_dict
    fp_state_dict = utils.adapt_resnet18_statedict(
            pretrained_model.state_dict(), fp_model.state_dict())
    fp_model.load_state_dict(fp_state_dict, strict=False)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres18_w8a7_pow2_foldbn(arch_cfg_path, target='latency', **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 21, [[8]] * 21
    s_up = kwargs.pop('analog_speedup', 5.)
    std_head = kwargs.pop('std_head', True)
    fp_model = ResNet18(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, qtz_fc='multi', std_head=std_head, **kwargs)
    q_model = ResNet18(qm2.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                       archws, archas, qtz_fc='multi', bn=False, std_head=std_head,
                       target=target, **kwargs)
    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres8_w5a8(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 10, [[5]] * 10
    # Set first and last layer weights precision to 8bit
    archws[0] = [8]
    archws[-1] = [8]
    s_up = kwargs.pop('analog_speedup', 5.)

    # Build Model
    model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                         archws, archas, qtz_fc='multi', **kwargs)
    state_dict = torch.load(arch_cfg_path)['state_dict']
    model.load_state_dict(state_dict)
    return model


def quantres8_w2a8(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 10, [[2]] * 10
    # Set first and last layer weights precision to 8bit
    archws[0] = [8]
    archws[-1] = [8]
    s_up = kwargs.pop('analog_speedup', 5.)

    # Build Model
    model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                         archws, archas, qtz_fc='multi', **kwargs)
    # state_dict = torch.load(arch_cfg_path)['state_dict']
    # model.load_state_dict(state_dict)
    return model


def quantres20_w2a8(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 22, [[2]] * 22
    # Set first and last layer weights precision to 8bit
    archws[0] = [8]
    archws[-1] = [8]
    s_up = kwargs.pop('analog_speedup', 5.)

    # Build Model
    model = ResNet20(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                     archws, archas, qtz_fc='multi', **kwargs)
    # state_dict = torch.load(arch_cfg_path)['state_dict']
    # model.load_state_dict(state_dict)
    return model


def quantres8_w2a8_pretrained(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[8]] * 10, [[2]] * 10
    # Set first and last layer weights precision to 8bit
    archws[0] = [8]
    archws[-1] = [8]
    s_up = kwargs.pop('analog_speedup', 5.)
    q_model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                           archws, archas, qtz_fc='multi', **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']

    # Load fp bn params in quantized model
    q_model.load_state_dict(fp_state_dict, strict=False)
    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fp_to_q(fp_state_dict)
    # Load fp weights in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres8_w2a8_nobn(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 10, [[2]] * 10
    # Set first and last layer weights precision to 8bit
    archws[0] = [8]
    archws[-1] = [8]
    s_up = kwargs.pop('analog_speedup', 5.)

    # Build Model
    model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                         archws, archas, qtz_fc='multi', bn=False, **kwargs)

    return model


def quantres8_w2a8_nobn_pretrained(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[8]] * 10, [[2]] * 10
    # Set first and last layer weights precision to 8bit
    archws[0] = [8]
    archws[-1] = [8]
    s_up = kwargs.pop('analog_speedup', 5.)
    q_model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                           archws, archas, qtz_fc='multi', bn=False, **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fp_to_q(fp_state_dict)
    # Load fp weights in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres8_w2a8_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[8]] * 10, [[2]] * 10
    # Set first and last layer weights precision to 8bit
    archws[0] = [8]
    archws[-1] = [8]
    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = TinyMLResNet(qm.FpConv2d, hw.diana(analog_speedup=5.),
                            archws, archas, qtz_fc='multi', **kwargs)
    q_model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                           archws, archas, qtz_fc='multi', bn=False, **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres20_w2a8_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[8]] * 22, [[2]] * 22
    # Set first and last layer weights precision to 8bit
    archws[0] = [8]
    archws[-1] = [8]
    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = ResNet20(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, qtz_fc='multi', **kwargs)
    q_model = ResNet20(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                       archws, archas, qtz_fc='multi', bn=False, **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres8_w2a8_foldbn_test(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[8]] * 10, [[2]] * 10
    # Set first and last layer weights precision to 8bit
    archws[0] = [8]
    archws[-1] = [8]
    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = TinyMLResNet(qm.FpConv2d, hw.diana(analog_speedup=5.),
                            archws, archas, qtz_fc='multi', **kwargs)
    q_model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                           archws, archas, qtz_fc='multi', bn=False, **kwargs)

    # Load pretrained q state_dict with float bn
    qbn_state_dict = torch.load(arch_cfg_path)['state_dict']
    # Translate qbn state dict to fp state dict
    fp_state_dict = utils.q_to_fp(qbn_state_dict)
    fp_model.load_state_dict(fp_state_dict, strict=False)
    # Fold bn on fp model
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Adapt and Load scale params from qbn dict
    q_model.load_state_dict(
        utils.adapt_scale_params(qbn_state_dict, q_model),
        strict=False)

    # Init scale param
    # utils.init_scale_param(q_model)

    return q_model


def quantres8_w2a7_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 10, [[2]] * 10
    # Set first and last layer weights precision to 8bit
    archws[0] = [8]
    archws[-1] = [8]
    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = TinyMLResNet(qm.FpConv2d, hw.diana(analog_speedup=5.),
                            archws, archas, qtz_fc='multi', **kwargs)
    q_model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                           archws, archas, qtz_fc='multi', bn=False, **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres20_w2a7_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 22, [[2]] * 22
    # Set first and last layer weights precision to 8bit
    archws[0] = [8]
    archws[-1] = [8]
    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = ResNet20(qm.FpConv2d, hw.diana(analog_speedup=s_up),
                        archws, archas, qtz_fc='multi', **kwargs)
    q_model = ResNet20(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                       archws, archas, qtz_fc='multi', bn=False, **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres20_w2a7_pow2_foldbn(arch_cfg_path, target='latency', **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 22, [[2]] * 22
    # Set first and last layer weights precision to 8bit
    archws[0] = [8]
    archws[-1] = [8]
    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = ResNet20(qm.FpConv2d, hw.diana(analog_speedup=s_up),
                        archws, archas, qtz_fc='multi', **kwargs)
    q_model = ResNet20(qm2.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                       archws, archas, qtz_fc='multi', bn=False,
                       target=target, **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres18_w2a7_foldbn(arch_cfg_path, target='latency', **kwargs):
    pretrained_model = torchvision.models.resnet18(pretrained=True)

    archas, archws = [[7]] * 21, [[2]] * 21
    # Set first and last layer weights precision to 8bit
    archws[0] = [8]
    archws[-1] = [8]
    s_up = kwargs.pop('analog_speedup', 5.)
    std_head = kwargs.pop('std_head', True)
    fp_model = ResNet18(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, qtz_fc='multi', std_head=std_head, **kwargs)
    q_model = ResNet18(qm2.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                       archws, archas, qtz_fc='multi', bn=False, std_head=std_head,
                       target=target, **kwargs)
    # Load pretrained fp state_dict
    fp_state_dict = utils.adapt_resnet18_statedict(
            pretrained_model.state_dict(), fp_model.state_dict())
    fp_model.load_state_dict(fp_state_dict, strict=False)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres18_w2a7_pow2_foldbn(arch_cfg_path, target='latency', **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 21, [[2]] * 21
    # Set first and last layer weights precision to 8bit
    archws[0] = [8]
    archws[-1] = [8]
    s_up = kwargs.pop('analog_speedup', 5.)
    std_head = kwargs.pop('std_head', True)
    fp_model = ResNet18(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, qtz_fc='multi', std_head=std_head, **kwargs)
    q_model = ResNet18(qm2.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                       archws, archas, qtz_fc='multi', bn=False, std_head=std_head,
                       target=target, **kwargs)
    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres8_w2a8_true(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 10, [[2]] * 10
    s_up = kwargs.pop('analog_speedup', 5.)

    # Build Model
    model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                         archws, archas, qtz_fc='multi', **kwargs)
    # state_dict = torch.load(arch_cfg_path)['state_dict']
    # model.load_state_dict(state_dict)
    return model


def quantres8_w2a8_true_pretrained(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[8]] * 10, [[2]] * 10
    s_up = kwargs.pop('analog_speedup', 5.)
    q_model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                           archws, archas, qtz_fc='multi', **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']

    # Load fp bn params in quantized model
    q_model.load_state_dict(fp_state_dict, strict=False)
    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fp_to_q(fp_state_dict)
    # Load fp weights in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres8_w2a8_true_nobn(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 10, [[2]] * 10
    s_up = kwargs.pop('analog_speedup', 5.)

    # Build Model
    model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                         archws, archas, qtz_fc='multi', bn=False, **kwargs)
    # state_dict = torch.load(arch_cfg_path)['state_dict']
    # model.load_state_dict(state_dict)
    return model


def quantres8_w2a8_true_nobn_pretrained(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[8]] * 10, [[2]] * 10
    s_up = kwargs.pop('analog_speedup', 5.)
    q_model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                           archws, archas, qtz_fc='multi', bn=False, **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fp_to_q(fp_state_dict)
    # Load fp weights in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres8_w2a8_true_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[8]] * 10, [[2]] * 10
    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = TinyMLResNet(qm.FpConv2d, hw.diana(analog_speedup=5.),
                            archws, archas, qtz_fc='multi', **kwargs)
    q_model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                           archws, archas, qtz_fc='multi', bn=False, **kwargs)

    # Load pretrained state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres8_w2a7_true_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 10, [[2]] * 10
    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = TinyMLResNet(qm.FpConv2d, hw.diana(analog_speedup=5.),
                            archws, archas, qtz_fc='multi', **kwargs)
    q_model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                           archws, archas, qtz_fc='multi', bn=False, **kwargs)

    # Load pretrained state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres20_w2a8_true(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 22, [[2]] * 22
    s_up = kwargs.pop('analog_speedup', 5.)

    # Build Model
    model = ResNet20(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                     archws, archas, qtz_fc='multi', **kwargs)
    # state_dict = torch.load(arch_cfg_path)['state_dict']
    # model.load_state_dict(state_dict)
    return model


def quantres20_w2a7_true_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 22, [[2]] * 22
    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = ResNet20(qm.FpConv2d, hw.diana(analog_speedup=s_up),
                        archws, archas, qtz_fc='multi', **kwargs)
    q_model = ResNet20(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                       archws, archas, qtz_fc='multi', bn=False, **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres20_w2a7_true_pow2_foldbn(arch_cfg_path, target='latency', **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 22, [[2]] * 22
    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = ResNet20(qm.FpConv2d, hw.diana(analog_speedup=s_up),
                        archws, archas, qtz_fc='multi', **kwargs)
    q_model = ResNet20(qm2.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                       archws, archas, qtz_fc='multi', bn=False,
                       target=target, **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres18_w2a7_true_foldbn(arch_cfg_path, target='latency', **kwargs):
    pretrained_model = torchvision.models.resnet18(pretrained=True)

    archas, archws = [[7]] * 21, [[2]] * 21
    s_up = kwargs.pop('analog_speedup', 5.)
    std_head = kwargs.pop('std_head', True)
    fp_model = ResNet18(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, qtz_fc='multi', std_head=std_head, **kwargs)
    q_model = ResNet18(qm2.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                       archws, archas, qtz_fc='multi', bn=False, std_head=std_head,
                       target=target, **kwargs)
    # Load pretrained fp state_dict
    fp_state_dict = utils.adapt_resnet18_statedict(
            pretrained_model.state_dict(), fp_model.state_dict())
    fp_model.load_state_dict(fp_state_dict, strict=False)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres18_w2a7_true_pow2_foldbn(arch_cfg_path, target='latency', **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 21, [[2]] * 21
    s_up = kwargs.pop('analog_speedup', 5.)
    std_head = kwargs.pop('std_head', True)
    fp_model = ResNet18(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, qtz_fc='multi', std_head=std_head, **kwargs)
    q_model = ResNet18(qm2.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                       archws, archas, qtz_fc='multi', bn=False, std_head=std_head,
                       target=target, **kwargs)
    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres20_minlat_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    # # OLD MODEL # #
    # archas, archws = [[7]] * 22, [[8]] * 22
    # # Set weights precision to 2bit in layers where analog is faster
    # archws[8] = [2]
    # archws[10] = [2]
    # archws[11] = [2]
    # archws[12] = [2]
    # archws[13] = [2]
    # archws[14] = [2]
    # archws[15] = [2]
    # archws[17] = [2]
    # archws[18] = [2]
    # archws[19] = [2]
    # archws[20] = [2]

    archas, archws = [[7]] * 22, [[2]] * 22
    # Set weights precision to 8bit in layers where digital is faster
    archws[0] = [8]
    archws[-1] = [8]

    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = ResNet20(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, qtz_fc='multi', **kwargs)
    q_model = ResNet20(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                       archws, archas, qtz_fc='multi', bn=False, **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres18_minlat64_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    std_head = kwargs.pop('std_head', True)
    archas, archws = [[7]] * 21, [[2]] * 21
    # Set weights precision to 8bit in layers where digital is faster
    # if std_head:  # TODO: check with new analog model.
    #     archws[7] = [8]
    #     archws[12] = [8]
    # With no-std-head minlat64 == w2a7_true
    # archws[20] = [8]
    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = ResNet18(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, qtz_fc='multi', std_head=std_head, **kwargs)
    q_model = ResNet18(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                       archws, archas, qtz_fc='multi', bn=False, std_head=std_head, **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantres18_minlat64_naive5_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    std_head = kwargs.pop('std_head', True)
    archas, archws = [[7]] * 21, [[8, 2]] * 21
    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = ResNet18(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, qtz_fc='multi', std_head=std_head, **kwargs)
    q_model = ResNet18(qm.QuantMultiPrecActivConv2d, hw.diana_naive(analog_speedup=s_up),
                       archws, archas, qtz_fc='multi', bn=False, std_head=std_head, **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    # Fix 16 channels to 8bit prec in each layer to achieve min latency
    utils.fix_ch_prec_naive(q_model, speedup=s_up)

    return q_model


def quantres18_minlat64_naive10_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    std_head = kwargs.pop('std_head', True)
    archas, archws = [[7]] * 21, [[8, 2]] * 21
    s_up = kwargs.pop('analog_speedup', 10.)
    fp_model = ResNet18(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, qtz_fc='multi', std_head=std_head, **kwargs)
    q_model = ResNet18(qm.QuantMultiPrecActivConv2d, hw.diana_naive(analog_speedup=s_up),
                       archws, archas, qtz_fc='multi', bn=False, std_head=std_head, **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    # Fix 16 channels to 8bit prec in each layer to achieve min latency
    utils.fix_ch_prec_naive(q_model, speedup=s_up)

    return q_model


def quantres20_minlat_max8_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    # # OLD MODEL # #
    # archas, archws = [[7]] * 22, [[8]] * 22
    # # Set weights precision to 2bit in layers where analog is faster
    # archws[8] = [8, 2]
    # archws[10] = [8, 2]
    # archws[11] = [8, 2]
    # archws[12] = [8, 2]
    # archws[13] = [8, 2]
    # archws[14] = [8, 2]
    # archws[15] = [8, 2]
    # archws[17] = [8, 2]
    # archws[18] = [8, 2]
    # archws[19] = [8, 2]
    # archws[20] = [8, 2]

    archas, archws = [[7]] * 22, [[2]] * 22
    # Set weights precision to 2bit in layers where analog is faster
    archws[0] = [8]
    archws[14] = [8, 2]
    archws[-1] = [8]

    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = ResNet20(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, qtz_fc='multi', **kwargs)
    q_model = ResNet20(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                       archws, archas, qtz_fc='multi', bn=False, **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    # Fix 16 channels to 8bit prec in each layer to achieve min latency
    utils.fix_ch_prec(q_model, prec=8, ch=[4])

    return q_model


def quantres20_minlat_max8_pow2_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    # # OLD MODEL # #
    # archas, archws = [[7]] * 22, [[8]] * 22
    # # Set weights precision to 2bit in layers where analog is faster
    # archws[8] = [8, 2]
    # archws[10] = [8, 2]
    # archws[11] = [8, 2]
    # archws[12] = [8, 2]
    # archws[13] = [8, 2]
    # archws[14] = [8, 2]
    # archws[15] = [8, 2]
    # archws[17] = [8, 2]
    # archws[18] = [8, 2]
    # archws[19] = [8, 2]
    # archws[20] = [8, 2]

    archas, archws = [[7]] * 22, [[2]] * 22
    # Set weights precision to 2bit in layers where analog is faster
    archws[0] = [8]
    archws[14] = [8, 2]
    archws[-1] = [8]

    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = ResNet20(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, qtz_fc='multi', **kwargs)
    q_model = ResNet20(qm2.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                       archws, archas, qtz_fc='multi', bn=False, **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    # Fix 16 channels to 8bit prec in each layer to achieve min latency
    utils.fix_ch_prec(q_model, prec=8, ch=[4])

    return q_model


def quantres20_minlat_naive_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 22, [[8, 2]] * 22

    s_up = kwargs.pop('analog_speedup', 10.)
    fp_model = ResNet20(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, qtz_fc='multi', **kwargs)
    q_model = ResNet20(qm.QuantMultiPrecActivConv2d, hw.diana_naive(analog_speedup=s_up),
                       archws, archas, qtz_fc='multi', bn=False, **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    # Fix 16 channels to 8bit prec in each layer to achieve min latency
    utils.fix_ch_prec_naive(q_model, speedup=s_up)

    return q_model


def quantres18_minlat64_max8_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    std_head = kwargs.pop('std_head', True)
    # Set weights precision to 8bit in layers where digital is faster
    if std_head:  # TODO: check with new analog model
        archas, archws = [[7]] * 21, [[8, 2]] * 21
        archws[7] = [8]
        archws[12] = [8]
        archws[20] = [8]
    else:
        # Commented values are with older analog model
        archas, archws = [[7]] * 21, [[2]] * 21
        archws[0] = [8, 2]
        # archws[7] = [8, 2]
        # archws[12] = [8, 2]
        archws[15] = [8, 2]
        archws[16] = [8, 2]
        archws[17] = [8, 2]
        archws[18] = [8, 2]
        archws[19] = [8, 2]
        archws[20] = [8, 2]

    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = ResNet18(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, qtz_fc='multi', std_head=std_head, **kwargs)
    q_model = ResNet18(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                       archws, archas, qtz_fc='multi', bn=False, std_head=std_head, **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    # Fix 16 channels to 8bit prec in each layer to achieve min latency
    if std_head:
        utils.fix_ch_prec(q_model, prec=8, ch=16)
    else:
        utils.fix_ch_prec(q_model, prec=8, ch=[16]*6 + [127])

    return q_model


def quantres18_minlat_max8_foldbn(arch_cfg_path, target='latency', **kwargs):
    pretrained_model = torchvision.models.resnet18(pretrained=True)

    std_head = kwargs.pop('std_head', True)

    archas, archws = [[7]] * 21, [[2]] * 21
    archws[0] = [8, 2]
    archws[15] = [8, 2]
    archws[16] = [8, 2]
    archws[17] = [8, 2]
    archws[18] = [8, 2]
    archws[19] = [8, 2]
    archws[20] = [8, 2]

    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = ResNet18(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, qtz_fc='multi', std_head=std_head, **kwargs)
    q_model = ResNet18(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                       archws, archas, qtz_fc='multi', bn=False, std_head=std_head,
                       target=target, **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = utils.adapt_resnet18_statedict(
            pretrained_model.state_dict(), fp_model.state_dict())
    fp_model.load_state_dict(fp_state_dict, strict=False)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    utils.fix_ch_prec(q_model, prec=8, ch=[16]*6 + [127])

    return q_model


def quantres18_minlat64_max8_pow2_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    std_head = kwargs.pop('std_head', True)
    # Set weights precision to 8bit in layers where digital is faster
    if std_head:  # TODO: check with new analog model
        archas, archws = [[7]] * 21, [[8, 2]] * 21
        archws[7] = [8]
        archws[12] = [8]
        archws[20] = [8]
    else:
        # Commented values are with older analog model
        archas, archws = [[7]] * 21, [[2]] * 21
        archws[0] = [8, 2]
        # archws[7] = [8, 2]
        # archws[12] = [8, 2]
        archws[15] = [8, 2]
        archws[16] = [8, 2]
        archws[17] = [8, 2]
        archws[18] = [8, 2]
        archws[19] = [8, 2]
        archws[20] = [8, 2]

    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = ResNet18(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, qtz_fc='multi', std_head=std_head, **kwargs)
    q_model = ResNet18(qm2.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                       archws, archas, qtz_fc='multi', bn=False, std_head=std_head, **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    # Fix 16 channels to 8bit prec in each layer to achieve min latency
    if std_head:
        utils.fix_ch_prec(q_model, prec=8, ch=16)
    else:
        utils.fix_ch_prec(q_model, prec=8, ch=[16]*6 + [127])

    return q_model


def quantres18_minlat_max8_pow2_foldbn_c100(arch_cfg_path, target='latency', **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 21, [[2]] * 21
    archws[0] = [8, 2]
    archws[10] = [8, 2]
    archws[11] = [8, 2]
    archws[13] = [8, 2]
    archws[14] = [8, 2]
    archws[15] = [8, 2]
    archws[16] = [8, 2]
    archws[17] = [8, 2]
    archws[18] = [8, 2]
    archws[19] = [8, 2]
    archws[20] = [8]

    s_up = kwargs.pop('analog_speedup', 5.)
    std_head = kwargs.pop('std_head', False)
    fp_model = ResNet18(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, num_classes=100,
                        qtz_fc='multi', std_head=std_head, **kwargs)
    q_model = ResNet18(qm2.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                       archws, archas, num_classes=100,
                       qtz_fc='multi', bn=False, std_head=std_head,
                       target=target, **kwargs)

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['model_state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    utils.fix_ch_prec(q_model, prec=8, ch=[16]*5 + [32]*5)

    return q_model


def quantres8_diana_naive5(arch_cfg_path, **kwargs):
    return quantres8_diana_naive(arch_cfg_path, **kwargs)


def quantres8_diana_naive10(arch_cfg_path, **kwargs):
    return quantres8_diana_naive(arch_cfg_path, **kwargs)


def quantres8_diana_naive100(arch_cfg_path, **kwargs):
    return quantres8_diana_naive(arch_cfg_path, **kwargs)


# ToDO
# qtz_fc: None or 'fixed' or 'mixed' or 'multi'
def quantres8_diana_naive(arch_cfg_path, **kwargs):
    wbits, abits = [2, 8], [7]

    # ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    if len(archws) == 9:
        # Case of fixed-precision on last fc layer
        archws.append(8)
    assert len(archas) == 10  # 10 insead of 8 because conv1 and fc activations are also quantized
    assert len(archws) == 10  # 10 instead of 8 because conv1 and fc weights are also quantized
    ##

    model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=5.),
                         archws, archas, qtz_fc='multi', bn=False, **kwargs)

    if kwargs['fine_tune']:
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict)
    else:
        # Load only alphas weights
        alpha_state_dict = _load_alpha_state_dict(arch_cfg_path)
        model.load_state_dict(alpha_state_dict, strict=False)
    return model


def quantres8_diana(arch_cfg_path, **kwargs):
    wbits, abits = [8, 2], [7]

    # ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    if len(archws) == 9:
        # Case of fixed-precision on last fc layer
        archws.append(8)
    assert len(archas) == 10  # 10 insead of 8 because conv1 and fc activations are also quantized
    assert len(archws) == 10  # 10 instead of 8 because conv1 and fc weights are also quantized
    ##

    model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(),
                         archws, archas, qtz_fc='multi', bn=False, **kwargs)

    if kwargs['fine_tune']:
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict)
    else:
        # Load only alphas weights
        alpha_state_dict = _load_alpha_state_dict(arch_cfg_path)
        model.load_state_dict(alpha_state_dict, strict=False)
    return model


def quantres20_diana_naive5(arch_cfg_path, **kwargs):
    wbits, abits = [8, 2], [7]

    # ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    # if len(archws) == 21:
    #     # Case of fixed-precision on last fc layer
    #     archws.append(8)
    # assert len(archas) == 22  # 10 insead of 8 because conv1 and fc activations are also quantized
    # assert len(archws) == 22  # 10 instead of 8 because conv1 and fc weights are also quantized
    ##

    kwargs.pop('analog_speedup', 5.)
    model = ResNet20(qm.QuantMultiPrecActivConv2d, hw.diana_naive(5.),
                     archws, archas, qtz_fc='multi', bn=False, **kwargs)
    utils.init_scale_param(model)

    return _quantres20_diana(arch_cfg_path, model, **kwargs)


def quantres20_diana_naive10(arch_cfg_path, **kwargs):
    wbits, abits = [8, 2], [7]

    # ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    # if len(archws) == 21:
    #     # Case of fixed-precision on last fc layer
    #     archws.append(8)
    # assert len(archas) == 22  # 10 insead of 8 because conv1 and fc activations are also quantized
    # assert len(archws) == 22  # 10 instead of 8 because conv1 and fc weights are also quantized
    ##

    kwargs.pop('analog_speedup', 5.)
    model = ResNet20(qm.QuantMultiPrecActivConv2d, hw.diana_naive(10.),
                     archws, archas, qtz_fc='multi', bn=False, **kwargs)
    utils.init_scale_param(model)

    return _quantres20_diana(arch_cfg_path, model, **kwargs)


def quantres20_diana_full(arch_cfg_path, **kwargs):
    wbits, abits = [8, 2], [7]

    # ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    # if len(archws) == 21:
    #     # Case of fixed-precision on last fc layer
    #     archws.append(8)
    # assert len(archas) == 22  # 10 insead of 8 because conv1 and fc activations are also quantized
    # assert len(archws) == 22  # 10 instead of 8 because conv1 and fc weights are also quantized
    ##

    kwargs.pop('analog_speedup', 5.)
    model = ResNet20(qm.QuantMultiPrecActivConv2d, hw.diana(),
                     archws, archas, qtz_fc='multi', bn=False, **kwargs)
    utils.init_scale_param(model)

    return _quantres20_diana(arch_cfg_path, model, **kwargs)


def quantres20_pow2_diana_full(arch_cfg_path, **kwargs):
    wbits, abits = [8, 2], [7]

    # ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    # if len(archws) == 21:
    #     # Case of fixed-precision on last fc layer
    #     archws.append(8)
    # assert len(archas) == 22  # 10 insead of 8 because conv1 and fc activations are also quantized
    # assert len(archws) == 22  # 10 instead of 8 because conv1 and fc weights are also quantized
    ##

    kwargs.pop('analog_speedup', 5.)
    model = ResNet20(qm2.QuantMultiPrecActivConv2d, hw.diana(),
                     archws, archas, qtz_fc='multi', bn=False, **kwargs)
    utils.init_scale_param(model)

    return _quantres20_diana(arch_cfg_path, model, **kwargs)


def quantres20_diana_reduced(arch_cfg_path, **kwargs):
    is_searchable = utils.detect_ad_tradeoff(quantres20_fp(None), torch.rand((1, 3, 32, 32)))

    wbits, abits = [8, 2], [7]

    # ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits if is_searchable[idx] else [wbits[0]]
              for idx, w_ch in enumerate(best_arch['alpha_weight'])]
    if len(archws) == 21:
        # Case of fixed-precision on last fc layer
        archws.append(8)
    assert len(archas) == 22  # 10 insead of 8 because conv1 and fc activations are also quantized
    assert len(archws) == 22  # 10 instead of 8 because conv1 and fc weights are also quantized
    ##

    model = ResNet20(qm.QuantMultiPrecActivConv2d, hw.diana(),
                     archws, archas, qtz_fc='multi', bn=False, **kwargs)

    return _quantres20_diana(arch_cfg_path, model, **kwargs)


def quantres18_diana_naive5(arch_cfg_path, **kwargs):
    wbits, abits = [8, 2], [7]

    # ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    # if len(archws) == 21:
    #     # Case of fixed-precision on last fc layer
    #     archws.append(8)
    # assert len(archas) == 22  # 10 insead of 8 because conv1 and fc activations are also quantized
    # assert len(archws) == 22  # 10 instead of 8 because conv1 and fc weights are also quantized
    ##

    kwargs.pop('analog_speedup', 5.)
    model = ResNet18(qm.QuantMultiPrecActivConv2d, hw.diana_naive(5.),
                     archws, archas, qtz_fc='multi', bn=False, **kwargs)
    utils.init_scale_param(model)

    return _quantres18_diana(arch_cfg_path, model, **kwargs)


def quantres18_diana_naive10(arch_cfg_path, **kwargs):
    wbits, abits = [8, 2], [7]

    # ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    # if len(archws) == 21:
    #     # Case of fixed-precision on last fc layer
    #     archws.append(8)
    # assert len(archas) == 22  # 10 insead of 8 because conv1 and fc activations are also quantized
    # assert len(archws) == 22  # 10 instead of 8 because conv1 and fc weights are also quantized
    ##

    kwargs.pop('analog_speedup', 5.)
    model = ResNet18(qm.QuantMultiPrecActivConv2d, hw.diana_naive(10.),
                     archws, archas, qtz_fc='multi', bn=False, **kwargs)
    utils.init_scale_param(model)

    return _quantres18_diana(arch_cfg_path, model, **kwargs)


def quantres18_pow2_diana_naive10(arch_cfg_path, **kwargs):
    wbits, abits = [8, 2], [7]

    # ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    # if len(archws) == 21:
    #     # Case of fixed-precision on last fc layer
    #     archws.append(8)
    # assert len(archas) == 22  # 10 insead of 8 because conv1 and fc activations are also quantized
    # assert len(archws) == 22  # 10 instead of 8 because conv1 and fc weights are also quantized
    ##

    kwargs.pop('analog_speedup', 5.)
    model = ResNet18(qm2.QuantMultiPrecActivConv2d, hw.diana_naive(10.),
                     archws, archas, qtz_fc='multi', bn=False, **kwargs)
    utils.init_scale_param(model)

    return _quantres18_diana(arch_cfg_path, model, **kwargs)


def quantres18_diana_full(arch_cfg_path, **kwargs):
    wbits, abits = [8, 2], [7]

    # ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    # if len(archws) == 20:
    #     # Case of fixed-precision on last fc layer
    #     archws.append(8)
    # assert len(archas) == 21  # 10 insead of 8 because conv1 and fc activations are also quantized
    # assert len(archws) == 21  # 10 instead of 8 because conv1 and fc weights are also quantized
    ##

    kwargs.pop('analog_speedup', 5.)
    model = ResNet18(qm2.QuantMultiPrecActivConv2d, hw.diana(),
                     archws, archas, qtz_fc='multi', bn=False, **kwargs)
    utils.init_scale_param(model)

    return _quantres18_diana(arch_cfg_path, model, **kwargs)


def quantres18_pow2_diana_full(arch_cfg_path, **kwargs):
    wbits, abits = [8, 2], [7]

    # ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    # if len(archws) == 21:
    #     # Case of fixed-precision on last fc layer
    #     archws.append(8)
    # assert len(archas) == 22  # 10 insead of 8 because conv1 and fc activations are also quantized
    # assert len(archws) == 22  # 10 instead of 8 because conv1 and fc weights are also quantized
    ##

    kwargs.pop('analog_speedup', 5.)
    model = ResNet18(qm2.QuantMultiPrecActivConv2d, hw.diana(),
                     archws, archas, qtz_fc='multi', bn=False, **kwargs)
    utils.init_scale_param(model)

    return _quantres18_diana(arch_cfg_path, model, **kwargs)


def quantres18_pow2_diana_full_c100(arch_cfg_path, **kwargs):
    wbits, abits = [8, 2], [7]

    # ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    # if len(archws) == 21:
    #     # Case of fixed-precision on last fc layer
    #     archws.append(8)
    # assert len(archas) == 22  # 10 insead of 8 because conv1 and fc activations are also quantized
    # assert len(archws) == 22  # 10 instead of 8 because conv1 and fc weights are also quantized
    ##

    kwargs.pop('analog_speedup', 5.)
    std_head = kwargs.pop('std_head', False)
    model = ResNet18(qm2.QuantMultiPrecActivConv2d, hw.diana(),
                     archws, archas, num_classes=100,
                     qtz_fc='multi', bn=False, std_head=std_head, **kwargs)
    utils.init_scale_param(model)

    return _quantres18_diana(arch_cfg_path, model, **kwargs)


def quantres18_pow2_diana_full_c100_no1st(arch_cfg_path, **kwargs):
    wbits, abits = [8, 2], [7]

    # ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    # if len(archws) == 21:
    #     # Case of fixed-precision on last fc layer
    #     archws.append(8)
    # assert len(archas) == 22  # 10 insead of 8 because conv1 and fc activations are also quantized
    # assert len(archws) == 22  # 10 instead of 8 because conv1 and fc weights are also quantized
    ##
    archws[0] = 8

    kwargs.pop('analog_speedup', 5.)
    model = ResNet18(qm2.QuantMultiPrecActivConv2d, hw.diana(),
                     archws, archas, num_classes=100,
                     qtz_fc='multi', bn=False, **kwargs)
    utils.init_scale_param(model)

    return _quantres18_diana(arch_cfg_path, model, **kwargs)


def quantres18_diana_reduced(arch_cfg_path, **kwargs):
    res = kwargs['input_size']
    std_head = kwargs['std_head']
    is_searchable = utils.detect_ad_tradeoff(
        quantres18_fp(None, pretrained=False, std_head=std_head),
        torch.rand((1, 3, res, res)))

    wbits, abits = [8, 2], [7]

    # ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits if is_searchable[idx] else [wbits[0]]
              for idx, w_ch in enumerate(best_arch['alpha_weight'])]
    # if len(archws) == 20:
    #     # Case of fixed-precision on last fc layer
    #     archws.append(8)
    # assert len(archas) == 21  # 10 insead of 8 because conv1 and fc activations are also quantized
    # assert len(archws) == 21  # 10 instead of 8 because conv1 and fc weights are also quantized
    ##

    model = ResNet18(qm.QuantMultiPrecActivConv2d, hw.diana(),
                     archws, archas, qtz_fc='multi', bn=False, **kwargs)

    return _quantres18_diana(arch_cfg_path, model, **kwargs)


def _quantres20_diana(arch_cfg_path, model, **kwargs):
    if kwargs.get('fine_tune', True):
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict)
    else:
        # Load only alphas weights
        alpha_state_dict = _load_alpha_state_dict(arch_cfg_path)
        model.load_state_dict(alpha_state_dict, strict=False)
    return model


def _quantres18_diana(arch_cfg_path, model, **kwargs):
    if kwargs.get('fine_tune', True):
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['model_state_dict']
        model.load_state_dict(state_dict)
    else:
        # Load only alphas weights
        alpha_state_dict = _load_alpha_state_dict(arch_cfg_path)
        model.load_state_dict(alpha_state_dict, strict=False)
    return model
