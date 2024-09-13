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

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils
from . import quant_module as qm
from . import quant_module_pow2 as qm2
from . import hw_models as hw
from .quant_resnet import quantres8_fp_foldbn, quantres20_fp_foldbn, quantres20_fp, \
    quantres18_fp_foldbn, quantres18_fp, quantres18_fp_foldbn_c100


# DJP
__all__ = [
    'mixres8_diana_naive5', 'mixres8_diana_naive10', 'mixres8_diana_naive100',
    'mixres8_diana',
    'mixres20_diana_naive5', 'mixres20_diana_naive10',
    'mixres20_diana_reduced', 'mixres20_diana_full', 'mixres20_pow2_diana_full',
    'mixres18_diana_naive5', 'mixres18_diana_naive10', 'mixres18_pow2_diana_naive10',
    'mixres18_diana_reduced', 'mixres18_diana_full', 'mixres18_pow2_diana_full',
    'mixres18_pow2_diana_full_c100', 'mixres18_pow2_diana_full_c100_no1st',
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


def conv7x7(conv_func, hw_model, is_searchable, in_planes, out_planes, bias=False,
            stride=1, groups=1, fix_qtz=False,
            target='latency', **kwargs):
    "7x7 convolution with padding"
    if conv_func != nn.Conv2d:
        if not is_searchable:
            kwargs['wbits'] = [8]
        return conv_func(hw_model, in_planes, out_planes,
                         kernel_size=7, groups=groups, stride=stride,
                         padding=3, bias=bias, fix_qtz=fix_qtz,
                         target=target, **kwargs)
    else:
        return conv_func(in_planes, out_planes,
                         kernel_size=7, groups=groups, stride=stride,
                         padding=3, bias=bias, **kwargs)


# MR
def fc(conv_func, hw_model, is_searchable, in_planes, out_planes, stride=1, groups=1,
       search_fc=None, target='latency', **kwargs):
    "fc mapped to conv"
    if not is_searchable:
        kwargs['wbits'] = [8]
    return conv_func(hw_model, in_planes, out_planes, kernel_size=1, groups=groups, stride=stride,
                     padding=0, bias=True, fc=search_fc,
                     target=target, **kwargs)


# MR
class Backbone20(nn.Module):
    def __init__(self, conv_func, hw_model, is_searchable, input_size, bn,
                 target='latency', **kwargs):
        self.fp = conv_func is qm.FpConv2d
        super().__init__()
        self.bb_1_0 = BasicBlockGumbel(conv_func, hw_model, is_searchable[:2], 16, 16, stride=1,
                                       bn=bn, target=target, **kwargs)
        self.bb_1_1 = BasicBlockGumbel(conv_func, hw_model, is_searchable[2:4], 16, 16, stride=1,
                                       bn=bn, target=target, **kwargs)
        self.bb_1_2 = BasicBlockGumbel(conv_func, hw_model, is_searchable[4:6], 16, 16, stride=1,
                                       bn=bn, target=target, **kwargs)
        self.bb_2_0 = BasicBlockGumbel(conv_func, hw_model, is_searchable[6:9], 16, 32, stride=2,
                                       bn=bn, target=target, **kwargs)
        self.bb_2_1 = BasicBlockGumbel(conv_func, hw_model, is_searchable[9:11], 32, 32, stride=1,
                                       bn=bn, target=target, **kwargs)
        self.bb_2_2 = BasicBlockGumbel(conv_func, hw_model, is_searchable[11:13], 32, 32, stride=1,
                                       bn=bn, target=target, **kwargs)
        self.bb_3_0 = BasicBlockGumbel(conv_func, hw_model, is_searchable[13:16], 32, 64, stride=2,
                                       bn=bn, target=target, **kwargs)
        self.bb_3_1 = BasicBlockGumbel(conv_func, hw_model, is_searchable[16:18], 64, 64, stride=1,
                                       bn=bn, target=target, **kwargs)
        self.bb_3_2 = BasicBlockGumbel(conv_func, hw_model, is_searchable[18:20], 64, 64, stride=1,
                                       bn=bn, target=target, **kwargs)
        if not self.fp:
            # If not fp we use quantized pooling
            self.pool = qm2.QuantAvgPool2d(kwargs['abits'], kernel_size=8)
        else:
            self.pool = nn.AvgPool2d(kernel_size=8)

    def forward(self, x, temp, is_hard):
        x = self.bb_1_0(x, temp, is_hard)
        x = self.bb_1_1(x, temp, is_hard)
        x = self.bb_1_2(x, temp, is_hard)
        x = self.bb_2_0(x, temp, is_hard)
        x = self.bb_2_1(x, temp, is_hard)
        x = self.bb_2_2(x, temp, is_hard)
        x = self.bb_3_0(x, temp, is_hard)
        x = self.bb_3_1(x, temp, is_hard)
        out = self.bb_3_2(x, temp, is_hard)
        if self.fp:
            out = F.relu(out)
        out = self.pool(out)
        return out


# MR
class BackboneTiny(nn.Module):
    def __init__(self, conv_func, hw_model, input_size, bn, **kwargs):
        super().__init__()
        self.bb_1 = BasicBlockGumbel(conv_func, hw_model, 16, 16, stride=1, bn=bn, **kwargs)
        self.bb_2 = BasicBlockGumbel(conv_func, hw_model, 16, 32, stride=2, bn=bn, **kwargs)
        self.bb_3 = BasicBlockGumbel(conv_func, hw_model, 32, 64, stride=2, bn=bn, **kwargs)
        self.pool = nn.AvgPool2d(kernel_size=8)

    def forward(self, x, temp, is_hard):
        x = self.bb_1(x, temp, is_hard)
        x = self.bb_2(x, temp, is_hard)
        x = self.bb_3(x, temp, is_hard)
        x = self.pool(x)
        return x


# MR
class Backbone18(nn.Module):
    def __init__(self, conv_func, hw_model, is_searchable, input_size, bn,
                 std_head=True, target='latency', **kwargs):
        self.fp = conv_func is qm.FpConv2d
        super().__init__()
        self.std_head = std_head
        if std_head:
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bb_1_0 = BasicBlockGumbel(conv_func, hw_model, is_searchable[:2], 64, 64, stride=1,
                                       bn=bn, target=target, **kwargs)
        self.bb_1_1 = BasicBlockGumbel(conv_func, hw_model, is_searchable[2:4], 64, 64, stride=1,
                                       bn=bn, target=target, **kwargs)
        self.bb_2_0 = BasicBlockGumbel(conv_func, hw_model, is_searchable[4:7], 64, 128, stride=2,
                                       bn=bn, target=target, **kwargs)
        self.bb_2_1 = BasicBlockGumbel(conv_func, hw_model, is_searchable[7:9], 128, 128, stride=1,
                                       bn=bn, target=target, **kwargs)
        self.bb_3_0 = BasicBlockGumbel(conv_func, hw_model, is_searchable[9:12], 128, 256, stride=2,
                                       bn=bn, target=target, **kwargs)
        self.bb_3_1 = BasicBlockGumbel(conv_func, hw_model, is_searchable[12:14], 256, 256,
                                       stride=1, bn=bn, target=target, **kwargs)
        self.bb_4_0 = BasicBlockGumbel(conv_func, hw_model, is_searchable[12:15], 256, 512,
                                       stride=2, bn=bn, target=target, **kwargs)
        self.bb_4_1 = BasicBlockGumbel(conv_func, hw_model, is_searchable[15:17], 512, 512,
                                       stride=1, bn=bn, target=target, **kwargs)
        if not self.fp:
            # If not fp we use quantized pooling
            # self.avg_pool = qm2.QuantAvgPool2d(kwargs['abits'], kernel_size=7)
            self.avg_pool = qm2.QuantAvgPool2d(kwargs['abits'], kernel_size=4)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x, temp, is_hard):
        if self.std_head:
            x = self.max_pool(x)
        x = self.bb_1_0(x, temp, is_hard)
        x = self.bb_1_1(x, temp, is_hard)
        x = self.bb_2_0(x, temp, is_hard)
        x = self.bb_2_1(x, temp, is_hard)
        x = self.bb_3_0(x, temp, is_hard)
        x = self.bb_3_1(x, temp, is_hard)
        x = self.bb_4_0(x, temp, is_hard)
        out = self.bb_4_1(x, temp, is_hard)
        if self.fp:
            out = F.relu(out)
        out = self.avg_pool(out)
        return out


class BasicBlockGumbel(nn.Module):
    def __init__(self, conv_func, hw_model, is_searchable, inplanes, planes,
                 stride=1, downsample=None, bn=True, target='latency', **kwargs):
        self.bn = bn
        self.use_bias = not bn
        self.fp = conv_func is qm.FpConv2d
        super().__init__()
        self.conv1 = conv3x3(conv_func, hw_model, is_searchable[0], inplanes, planes,
                             stride=stride, bias=self.use_bias, target=target, **kwargs)
        if bn:
            self.bn1 = nn.BatchNorm2d(planes, affine=bn)
        self.conv2 = conv3x3(conv_func, hw_model, is_searchable[1], planes, planes,
                             bias=self.use_bias, target=target, **kwargs)
        if bn:
            self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or inplanes != planes:
            if not is_searchable[2]:
                kwargs['wbits'] = [8]
            self.downsample = conv_func(hw_model, inplanes, planes,
                                        kernel_size=1, stride=stride, groups=1, bias=self.use_bias,
                                        target=target, **kwargs)
            if bn:
                self.bn_ds = nn.BatchNorm2d(planes)
            if not self.fp:
                # Couple input clip_val
                inp_clip_val = self.conv1.mix_activ.mix_activ[0].clip_val
                self.downsample.mix_activ.mix_activ[0].clip_val = inp_clip_val

                # Quantized Sum node
                self.qadd = qm2.QuantAdd(kwargs['abits'])
        else:
            self.downsample = None
            if not self.fp:
                # If not fp and no downsample op we need to quantize the residual branch
                self.inp_q = qm2.QuantPaCTActiv(kwargs['abits'])
                # Couple input clip_val
                inp_clip_val = self.conv1.mix_activ.mix_activ[0].clip_val
                self.inp_q.mix_activ[0].clip_val = inp_clip_val

                # Quantized Sum node
                self.qadd = qm2.QuantAdd(kwargs['abits'], clip_val=inp_clip_val)

    def forward(self, x, temp, is_hard):
        if self.downsample is not None:
            residual = x
        else:
            if self.fp:
                residual = F.relu(x)
            else:
                # Here I need to quantize
                residual, _ = self.inp_q(x)

        out = self.conv1(x, temp, is_hard)
        if self.bn:
            out = self.bn1(out)
        out = self.conv2(out, temp, is_hard)
        if self.bn:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual, temp, is_hard)
            if self.bn:
                residual = self.bn_ds(residual)

        if self.fp:
            out += residual
        else:
            out = self.qadd(out, residual)

        return out


class ResNet20(nn.Module):
    def __init__(self, conv_func, hw_model, is_searchable,
                 search_fc=None, input_size=32, num_classes=10, bn=True,
                 target='latency', **kwargs):
        if 'abits' in kwargs:
            print('abits: {}'.format(kwargs['abits']))
        if 'wbits' in kwargs:
            print('wbits: {}'.format(kwargs['wbits']))

        self.inplanes = 16
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
        self.conv1 = conv3x3(conv_func, hw_model, is_searchable[0], 3, 16,
                             stride=1, groups=1,
                             bias=self.use_bias, max_inp_val=1.0,
                             target=target, **kwargs)
        if bn:
            self.bn1 = nn.BatchNorm2d(16)
        self.backbone = Backbone20(conv_func, hw_model, is_searchable[1:-1], input_size,
                                   bn, target=target, **kwargs)

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
        self.fc = fc(conv_func, hw_model, is_searchable[-1], 64, num_classes,
                     search_fc=self.search_fc, target=target, **kwargs)

    def forward(self, x, temp, is_hard):
        x = self.conv1(x, temp, is_hard)
        if self.bn:
            x = self.bn1(x)

        x = self.backbone(x, temp, is_hard)

        x = x if self.search_fc else x.view(x.size(0), -1)

        if self.search_fc:
            x = self.fc(x, temp, is_hard)
            return x[:, :, 0, 0]
        else:
            x = self.fc(x)
            return x

    def complexity_loss(self):
        loss = torch.tensor(0.)
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


class ResNet18(nn.Module):
    def __init__(self, conv_func, hw_model, is_searchable,
                 search_fc=None, input_size=64, num_classes=200, bn=True, std_head=True,
                 target='latency', **kwargs):
        if 'abits' in kwargs:
            print('abits: {}'.format(kwargs['abits']))
        if 'wbits' in kwargs:
            print('wbits: {}'.format(kwargs['wbits']))

        self.inplanes = 64
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
        if std_head:
            self.conv1 = conv7x7(conv_func, hw_model, is_searchable[0], 3, 64, stride=2, groups=1,
                                 bias=self.use_bias, target=target, **kwargs)
        else:
            self.conv1 = conv3x3(conv_func, hw_model, is_searchable[0], 3, 64, stride=1, groups=1,
                                 bias=self.use_bias, target=target, **kwargs)
        if bn:
            self.bn1 = nn.BatchNorm2d(64)
        self.backbone = Backbone18(conv_func, hw_model, is_searchable[1:-1], input_size,
                                   bn, std_head=std_head, max_inp_val=1.0,
                                   target=target, **kwargs)

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

        # Final classifier
        self.fc = fc(conv_func, hw_model, is_searchable[-1], 512, num_classes,
                     search_fc=self.search_fc, target=target, **kwargs)

    def forward(self, x, temp, is_hard):
        x = self.conv1(x, temp, is_hard)
        if self.bn:
            x = self.bn1(x)

        x = self.backbone(x, temp, is_hard)

        x = x if self.search_fc else x.view(x.size(0), -1)

        if self.search_fc:
            x = self.fc(x, temp, is_hard)
            return x[:, :, 0, 0]
        else:
            x = self.fc(x)
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


class TinyMLResNet(nn.Module):
    def __init__(self, conv_func, hw_model,
                 search_fc=None, input_size=32, num_classes=10, bn=True, **kwargs):
        if 'abits' in kwargs:
            print('abits: {}'.format(kwargs['abits']))
        if 'wbits' in kwargs:
            print('wbits: {}'.format(kwargs['wbits']))

        self.inplanes = 16
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

        # Model
        self.conv1 = conv3x3(conv_func, hw_model, 3, 16, stride=1, groups=1,
                             bias=self.use_bias, **kwargs)
        if bn:
            self.bn1 = nn.BatchNorm2d(16)
        self.backbone = BackboneTiny(conv_func, hw_model, input_size, bn, **kwargs)
        self.fc = fc(conv_func, hw_model, 64, num_classes, search_fc=self.search_fc, **kwargs)

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

    def forward(self, x, temp, is_hard):
        x = self.conv1(x, temp, is_hard)
        if self.bn:
            x = self.bn1(x)

        x = self.backbone(x, temp, is_hard)

        x = x if self.search_fc else x.view(x.size(0), -1)

        if self.search_fc:
            x = self.fc(x, temp, is_hard)
            return x[:, :, 0, 0]
        else:
            x = self.fc(x)
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


# MR
def mixres8_diana_naive5(arch_cfg_path, **kwargs):
    search_model = mixres8_diana_naive(arch_cfg_path, 5., **kwargs)
    return search_model


# MR
def mixres8_diana_naive10(arch_cfg_path, **kwargs):
    search_model = mixres8_diana_naive(arch_cfg_path, 10., **kwargs)
    return search_model


# MR
def mixres8_diana_naive100(arch_cfg_path, **kwargs):
    search_model = mixres8_diana_naive(arch_cfg_path, 100., **kwargs)
    return search_model


def mixres8_diana_naive(arch_cfg_path, s_up, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    # NB: 2 bits is equivalent for ternary weights!!
    search_model = TinyMLResNet(
        qm.MultiPrecActivConv2d, hw.diana_naive(analog_speedup=s_up),
        search_fc='multi', wbits=[2, 8], abits=[7], bn=False,
        share_weight=True, **kwargs)

    # Get folded pretrained model
    folded_fp_model = quantres8_fp_foldbn(arch_cfg_path)
    folded_state_dict = folded_fp_model.state_dict()

    # Delete folded model
    del folded_fp_model

    # Translate folded state dict in a format compatible with searchable layers
    search_state_dict = utils.fpfold_to_q(folded_state_dict)
    search_model.load_state_dict(search_state_dict, strict=False)

    # Init quantization scale param
    utils.init_scale_param(search_model)

    return search_model


def mixres8_diana(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    # NB: 2 bits is equivalent for ternary weights!!
    search_model = TinyMLResNet(
        qm.MultiPrecActivConv2d, hw.diana(),
        search_fc='multi', wbits=[8, 2], abits=[7], bn=False,
        share_weight=True, **kwargs)

    # Get folded pretrained model
    folded_fp_model = quantres8_fp_foldbn(arch_cfg_path)
    folded_state_dict = folded_fp_model.state_dict()

    # Delete folded model
    del folded_fp_model

    # Translate folded state dict in a format compatible with searchable layers
    search_state_dict = utils.fpfold_to_q(folded_state_dict)
    search_model.load_state_dict(search_state_dict, strict=False)

    # Init quantization scale param
    utils.init_scale_param(search_model)

    return search_model


def mixres20_diana_naive5(arch_cfg_path, **kwargs):
    # NB: 2 bits is equivalent for ternary weights!!
    search_model = ResNet20(
        qm.MultiPrecActivConv2d, hw.diana_naive(5.), [True]*22,
        search_fc='multi', wbits=[8, 2], abits=[7], bn=False,
        share_weight=True, **kwargs)
    return _mixres20_diana(arch_cfg_path, search_model)


def mixres20_diana_naive10(arch_cfg_path, **kwargs):
    # NB: 2 bits is equivalent for ternary weights!!
    search_model = ResNet20(
        qm.MultiPrecActivConv2d, hw.diana_naive(10.), [True]*22,
        search_fc='multi', wbits=[8, 2], abits=[7], bn=False,
        share_weight=True, **kwargs)
    return _mixres20_diana(arch_cfg_path, search_model)


def mixres20_diana_full(arch_cfg_path, **kwargs):
    # NB: 2 bits is equivalent for ternary weights!!
    search_model = ResNet20(
        qm.MultiPrecActivConv2d, hw.diana(), [True]*22,
        search_fc='multi', wbits=[8, 2], abits=[7], bn=False,
        share_weight=True, **kwargs)
    return _mixres20_diana(arch_cfg_path, search_model)


def mixres20_pow2_diana_full(arch_cfg_path, target='latency', **kwargs):
    # NB: 2 bits is equivalent for ternary weights!!
    search_model = ResNet20(
        qm2.MultiPrecActivConv2d, hw.diana(), [True]*22,
        search_fc='multi', wbits=[8, 2], abits=[7], bn=False,
        share_weight=True, target=target, **kwargs)
    return _mixres20_diana(arch_cfg_path, search_model)


def mixres20_diana_reduced(arch_cfg_path, **kwargs):
    is_searchable = utils.detect_ad_tradeoff(quantres20_fp(None), torch.rand((1, 3, 32, 32)))
    # NB: 2 bits is equivalent for ternary weights!!
    search_model = ResNet20(
        qm.MultiPrecActivConv2d, hw.diana(), is_searchable,
        search_fc='multi', wbits=[8, 2], abits=[7], bn=False,
        share_weight=True, **kwargs)
    return _mixres20_diana(arch_cfg_path, search_model)


def mixres18_diana_naive5(arch_cfg_path, **kwargs):
    std_head = kwargs.pop('std_head', True)
    # NB: 2 bits is equivalent for ternary weights!!
    search_model = ResNet18(
        qm.MultiPrecActivConv2d, hw.diana_naive(5.), [True]*22,
        search_fc='multi', wbits=[8, 2], abits=[7], bn=False,
        share_weight=True, std_head=std_head, **kwargs)
    return _mixres18_diana(arch_cfg_path, search_model, std_head)


def mixres18_diana_naive10(arch_cfg_path, **kwargs):
    std_head = kwargs.pop('std_head', True)
    # NB: 2 bits is equivalent for ternary weights!!
    search_model = ResNet18(
        qm.MultiPrecActivConv2d, hw.diana_naive(10.), [True]*22,
        search_fc='multi', wbits=[8, 2], abits=[7], bn=False,
        share_weight=True, std_head=std_head, **kwargs)
    return _mixres18_diana(arch_cfg_path, search_model, std_head)


def mixres18_pow2_diana_naive10(arch_cfg_path, target='latency', **kwargs):
    std_head = kwargs.pop('std_head', True)
    # NB: 2 bits is equivalent for ternary weights!!
    search_model = ResNet18(
        qm2.MultiPrecActivConv2d, hw.diana_naive(10.), [True]*22,
        search_fc='multi', wbits=[8, 2], abits=[7], bn=False,
        share_weight=True, std_head=std_head,
        target=target, **kwargs)
    return _mixres18_diana(arch_cfg_path, search_model, std_head)


def mixres18_diana_full(arch_cfg_path, target='latency', **kwargs):
    std_head = kwargs.pop('std_head', True)
    # NB: 2 bits is equivalent for ternary weights!!
    search_model = ResNet18(
        qm2.MultiPrecActivConv2d, hw.diana(), [True]*22,
        search_fc='multi', wbits=[8, 2], abits=[7], bn=False,
        share_weight=True, std_head=std_head,
        target=target, **kwargs)
    return _mixres18_diana(arch_cfg_path, search_model, std_head)


def mixres18_pow2_diana_full(arch_cfg_path, target='latency', **kwargs):
    std_head = kwargs.pop('std_head', True)
    # NB: 2 bits is equivalent for ternary weights!!
    search_model = ResNet18(
        qm2.MultiPrecActivConv2d, hw.diana(), [True]*22,
        search_fc='multi', wbits=[8, 2], abits=[7], bn=False,
        share_weight=True, std_head=std_head,
        target=target, **kwargs)
    return _mixres18_diana(arch_cfg_path, search_model, std_head)


def mixres18_pow2_diana_full_c100_no1st(arch_cfg_path, target='latency', **kwargs):
    std_head = kwargs.pop('std_head', False)
    # NB: 2 bits is equivalent for ternary weights!!
    search_model = ResNet18(
        qm2.MultiPrecActivConv2d, hw.diana(), [False] + [True]*21,
        search_fc='multi', wbits=[8, 2], abits=[7], bn=False,
        share_weight=True, std_head=std_head, num_classes=100,
        target=target, **kwargs)
    return _mixres18_diana_c100(arch_cfg_path, search_model, std_head)


def mixres18_pow2_diana_full_c100(arch_cfg_path, target='latency', **kwargs):
    std_head = kwargs.pop('std_head', False)
    # NB: 2 bits is equivalent for ternary weights!!
    search_model = ResNet18(
        qm2.MultiPrecActivConv2d, hw.diana(), [True]*22,
        search_fc='multi', wbits=[8, 2], abits=[7], bn=False,
        share_weight=True, std_head=std_head, num_classes=100,
        target=target, **kwargs)
    return _mixres18_diana_c100(arch_cfg_path, search_model, std_head)


def mixres18_diana_reduced(arch_cfg_path, **kwargs):
    res = kwargs['input_size']
    std_head = kwargs.pop('std_head', True)
    is_searchable = utils.detect_ad_tradeoff(
        quantres18_fp(None, pretrained=False, std_head=std_head),
        torch.rand((1, 3, res, res)))
    # NB: 2 bits is equivalent for ternary weights!!
    search_model = ResNet18(
        qm.MultiPrecActivConv2d, hw.diana(), is_searchable,
        search_fc='multi', wbits=[8, 2], abits=[7], bn=False,
        share_weight=True, std_head=std_head, **kwargs)
    return _mixres18_diana(arch_cfg_path, search_model, std_head)


def _mixres20_diana(arch_cfg_path, search_model):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    # # NB: 2 bits is equivalent for ternary weights!!
    # search_model = ResNet20(
    #     qm.MultiPrecActivConv2d, hw.diana(),
    #     search_fc='multi', wbits=[8, 2], abits=[7], bn=False,
    #     share_weight=True, **kwargs)

    # Get folded pretrained model
    folded_fp_model = quantres20_fp_foldbn(arch_cfg_path)
    folded_state_dict = folded_fp_model.state_dict()

    # Delete folded model
    del folded_fp_model

    # Translate folded state dict in a format compatible with searchable layers
    search_state_dict = utils.fpfold_to_q(folded_state_dict)
    search_model.load_state_dict(search_state_dict, strict=False)

    # Init quantization scale param
    utils.init_scale_param(search_model)

    return search_model


def _mixres18_diana(arch_cfg_path, search_model, std_head):
    # # NB: 2 bits is equivalent for ternary weights!!
    # search_model = ResNet20(
    #     qm.MultiPrecActivConv2d, hw.diana(),
    #     search_fc='multi', wbits=[8, 2], abits=[7], bn=False,
    #     share_weight=True, **kwargs)

    # Get folded pretrained model
    folded_fp_model = quantres18_fp_foldbn(arch_cfg_path, std_head=std_head)
    folded_state_dict = folded_fp_model.state_dict()

    # Delete folded model
    del folded_fp_model

    # Translate folded state dict in a format compatible with searchable layers
    search_state_dict = utils.fpfold_to_q(folded_state_dict)
    search_model.load_state_dict(search_state_dict, strict=False)

    # Init quantization scale param
    utils.init_scale_param(search_model)

    return search_model


def _mixres18_diana_c100(arch_cfg_path, search_model, std_head):
    # # NB: 2 bits is equivalent for ternary weights!!
    # search_model = ResNet20(
    #     qm.MultiPrecActivConv2d, hw.diana(),
    #     search_fc='multi', wbits=[8, 2], abits=[7], bn=False,
    #     share_weight=True, **kwargs)

    # Get folded pretrained model
    folded_fp_model = quantres18_fp_foldbn_c100(arch_cfg_path, std_head=std_head)
    folded_state_dict = folded_fp_model.state_dict()

    # Delete folded model
    del folded_fp_model

    # Translate folded state dict in a format compatible with searchable layers
    search_state_dict = utils.fpfold_to_q(folded_state_dict)
    search_model.load_state_dict(search_state_dict, strict=False)

    # Init quantization scale param
    utils.init_scale_param(search_model)

    return search_model
