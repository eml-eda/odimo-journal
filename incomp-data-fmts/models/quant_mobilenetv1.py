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

# import copy
# import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils
from . import quant_module as qm
from . import quant_module_pow2 as qm2
from . import hw_models as hw

# MR
__all__ = [
    'quantmobilenetv1_fp', 'quantmobilenetv1_fp_foldbn',
    'quantmobilenetv1_w8a7_foldbn', 'quantmobilenetv1_w8a7_pow2_foldbn',
    'quantmobilenetv1_w2a7_foldbn', 'quantmobilenetv1_w2a7_pow2_foldbn',
    'quantmobilenetv1_w2a7_true_foldbn',
    'quantmobilenetv1_minlat_foldbn',
    'quantmobilenetv1_minlat_max8_foldbn',
    'quantmobilenetv1_minlat_naive5_foldbn', 'quantmobilenetv1_minlat_naive10_foldbn',
    'quantmobilenetv1_diana_naive5', 'quantmobilenetv1_diana_naive10',
    'quantmobilenetv1_diana_full', 'quantmobilenetv1_pow2_diana_full',
    'quantmobilenetv1_diana_reduced',
    ]


def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class BasicBlock(nn.Module):

    def __init__(self, conv_func, inp, oup, archws, archas,
                 stride=1, bn=True, **kwargs):
        self.bn = bn
        self.use_bias = not bn
        super().__init__()
        # For depthwise archws is always [8]
        self.depth = conv_func(inp, inp, [8], archas[0],
                               kernel_size=3, stride=stride, padding=1,
                               bias=self.use_bias, groups=inp, **kwargs)
        if bn:
            self.bn_depth = nn.BatchNorm2d(inp)
        self.point = conv_func(inp, oup, archws[1], archas[1],
                               kernel_size=1, stride=1, padding=0,
                               bias=self.use_bias, groups=1, **kwargs)
        if bn:
            self.bn_point = nn.BatchNorm2d(oup)

    def forward(self, x):
        x = self.depth(x)
        if self.bn:
            x = self.bn_depth(x)
        x = self.point(x)
        if self.bn:
            x = self.bn_point(x)

        return x


class Backbone(nn.Module):

    def __init__(self, conv_func, input_size, bn, width_mult, abits, wbits,
                 **kwargs):
        self.fp = conv_func is qm.FpConv2d
        super().__init__()
        self.bb_1 = BasicBlock(
            conv_func, make_divisible(32*width_mult), make_divisible(64*width_mult),
            wbits[0:2], abits[0:2], stride=1, bn=bn, **kwargs)
        self.bb_2 = BasicBlock(
            conv_func, make_divisible(64*width_mult), make_divisible(128*width_mult),
            wbits[2:4], abits[2:4], stride=2, bn=bn, **kwargs)
        self.bb_3 = BasicBlock(
            conv_func, make_divisible(128*width_mult), make_divisible(128*width_mult),
            wbits[4:6], abits[4:6], stride=1, bn=bn, **kwargs)
        self.bb_4 = BasicBlock(
            conv_func, make_divisible(128*width_mult), make_divisible(256*width_mult),
            wbits[6:8], abits[6:8], stride=2, bn=bn, **kwargs)
        self.bb_5 = BasicBlock(
            conv_func, make_divisible(256*width_mult), make_divisible(256*width_mult),
            wbits[8:10], abits[8:10], stride=1, bn=bn, **kwargs)
        self.bb_6 = BasicBlock(
            conv_func, make_divisible(256*width_mult), make_divisible(512*width_mult),
            wbits[10:12], abits[10:12], stride=2, bn=bn, **kwargs)
        self.bb_7 = BasicBlock(
            conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult),
            wbits[12:14], abits[12:14], stride=1, bn=bn, **kwargs)
        self.bb_8 = BasicBlock(
            conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult),
            wbits[14:16], abits[14:16], stride=1, bn=bn, **kwargs)
        self.bb_9 = BasicBlock(
            conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult),
            wbits[16:18], abits[16:18], stride=1, bn=bn, **kwargs)
        self.bb_10 = BasicBlock(
            conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult),
            wbits[18:20], abits[18:20], stride=1, bn=bn, **kwargs)
        self.bb_11 = BasicBlock(
            conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult),
            wbits[20:22], abits[20:22], stride=1, bn=bn, **kwargs)
        self.bb_12 = BasicBlock(
            conv_func, make_divisible(512*width_mult), make_divisible(1024*width_mult),
            wbits[22:24], abits[22:24], stride=2, bn=bn, **kwargs)
        self.bb_13 = BasicBlock(
            conv_func, make_divisible(1024*width_mult), make_divisible(1024*width_mult),
            wbits[24:26], abits[24:26], stride=1, bn=bn, **kwargs)
        if not self.fp:
            # If not fp we use quantized pooling
            self.pool = qm2.QuantAvgPool2d(abits[0], int(input_size / (2**5)))
        else:
            self.pool = nn.AvgPool2d(int(input_size / (2**5)))

    def forward(self, x):
        out = self.bb_1(x)
        out = self.bb_2(out)
        out = self.bb_3(out)
        out = self.bb_4(out)
        out = self.bb_5(out)
        out = self.bb_6(out)
        out = self.bb_7(out)
        out = self.bb_8(out)
        out = self.bb_9(out)
        out = self.bb_10(out)
        out = self.bb_11(out)
        out = self.bb_12(out)
        out = self.bb_13(out)
        if self.fp:
            out = F.relu(out)
        out = self.pool(out)
        return out


class MobileNetV1(nn.Module):

    def __init__(self, conv_func, hw_model, archws, archas,
                 qtz_fc=None, width_mult=.25,
                 input_size=96, num_classes=2, bn=True,
                 target='latency', **kwargs):
        print('archas: {}'.format(archas))
        print('archws: {}'.format(archws))

        self.conv_func = conv_func
        self.hw_model = hw_model
        self.search_types = ['fixed', 'mixed', 'multi']
        if qtz_fc in self.search_types:
            self.qtz_fc = qtz_fc
        else:
            self.qtz_fc = False
        self.width_mult = width_mult
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
        self.input_layer = conv_func(3, make_divisible(32*width_mult),
                                     abits=archas[0], wbits=archws[0],
                                     kernel_size=3, stride=2, padding=1,
                                     bias=False, groups=1,
                                     max_inp_val=1.0, **kwargs)
        if bn:
            self.bn = nn.BatchNorm2d(make_divisible(32*width_mult))
        self.backbone = Backbone(conv_func, input_size, bn, width_mult,
                                 abits=archas[1:-1], wbits=archws[1:-1],
                                 **kwargs)

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
            make_divisible(1024*width_mult), num_classes,
            abits=archas[-1], wbits=archws[-1],
            kernel_size=1, stride=1, padding=0, bias=True, groups=1,
            fc=self.qtz_fc, **kwargs)

    def forward(self, x):
        x = self.input_layer(x)
        if self.bn:
            x = self.bn(x)
        x = self.backbone(x)
        x = self.fc(x)[:, :, 0, 0]

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


def quantmobilenetv1_fp(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 28, [[8]] * 28
    model = MobileNetV1(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, qtz_fc='multi', **kwargs)
    return model


def quantmobilenetv1_fp_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[8]] * 28, [[8]] * 28
    model = MobileNetV1(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, qtz_fc='multi', **kwargs)
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    model.load_state_dict(fp_state_dict)

    # Fold bn
    model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(model)
    folded_model.train()  # Put folded model in train mode

    return folded_model


def quantmobilenetv1_w8a7_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 28, [[8]] * 28
    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = MobileNetV1(qm.FpConv2d, hw.diana(analog_speedup=s_up),
                           archws, archas, qtz_fc='multi', **kwargs)
    q_model = MobileNetV1(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
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


def quantmobilenetv1_w8a7_pow2_foldbn(arch_cfg_path, target='latency', **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 28, [[8]] * 28
    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = MobileNetV1(qm.FpConv2d, hw.diana(analog_speedup=s_up),
                           archws, archas, qtz_fc='multi', **kwargs)
    q_model = MobileNetV1(qm2.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
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


def quantmobilenetv1_w2a7_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 28, [[2]] * 28
    # Set first and last layer weights precision to 8bit
    archws[0] = [8]
    archws[-1] = [8]
    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = MobileNetV1(qm.FpConv2d, hw.diana(analog_speedup=s_up),
                           archws, archas, qtz_fc='multi', **kwargs)
    q_model = MobileNetV1(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                          archws, archas, qtz_fc='multi', bn=False,
                          **kwargs)

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


def quantmobilenetv1_w2a7_pow2_foldbn(arch_cfg_path, target='latency', **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 28, [[2]] * 28
    # Set first and last layer weights precision to 8bit
    archws[0] = [8]
    archws[-1] = [8]
    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = MobileNetV1(qm.FpConv2d, hw.diana(analog_speedup=s_up),
                           archws, archas, qtz_fc='multi', **kwargs)
    q_model = MobileNetV1(qm2.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
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


def quantmobilenetv1_w2a7_true_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 28, [[2]] * 28
    s_up = kwargs.pop('analog_speedup', 5.)
    fp_model = MobileNetV1(qm.FpConv2d, hw.diana(analog_speedup=s_up),
                           archws, archas, qtz_fc='multi', **kwargs)
    q_model = MobileNetV1(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
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


def quantmobilenetv1_minlat_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 28, [[2]] * 28
    # Set weights precision to 8bit in layers where digital is faster
    archws[0] = [8]
    archws[-1] = [8]

    fp_model = MobileNetV1(qm.FpConv2d, hw.diana(analog_speedup=5.),
                           archws, archas, qtz_fc='multi', **kwargs)
    q_model = MobileNetV1(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=5.),
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


def quantmobilenetv1_minlat_max8_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 28, [[2]] * 28
    # Set weights precision to 8bit in layers where digital is faster
    archws[0] = [8]
    archws[2] = [8, 2]
    archws[12] = [8, 2]
    archws[24] = [8, 2]
    archws[26] = [8, 2]
    archws[27] = [8]

    fp_model = MobileNetV1(qm.FpConv2d, hw.diana(analog_speedup=5.),
                           archws, archas, qtz_fc='multi', **kwargs)
    q_model = MobileNetV1(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=5.),
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

    utils.fix_ch_prec(q_model, prec=8, ch=[6, 0, 31, 31])

    return q_model


def quantmobilenetv1_minlat_naive5_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 28, [[8, 2]] * 28
    archws[0] = [2]
    archws[-1] = [2]
    s_up = kwargs.pop('analog_speedup', 5.)

    fp_model = MobileNetV1(qm.FpConv2d, hw.diana(analog_speedup=5.),
                           archws, archas, qtz_fc='multi', **kwargs)
    q_model = MobileNetV1(qm.QuantMultiPrecActivConv2d, hw.diana_naive(analog_speedup=s_up),
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

    utils.fix_ch_prec_naive(q_model, speedup=s_up)

    return q_model


def quantmobilenetv1_minlat_naive10_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 28, [[8, 2]] * 28
    archws[0] = [2]
    archws[-1] = [2]
    s_up = kwargs.pop('analog_speedup', 10.)

    fp_model = MobileNetV1(qm.FpConv2d, hw.diana(analog_speedup=10.),
                           archws, archas, qtz_fc='multi', **kwargs)
    q_model = MobileNetV1(qm.QuantMultiPrecActivConv2d, hw.diana_naive(analog_speedup=s_up),
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

    utils.fix_ch_prec_naive(q_model, speedup=s_up)

    return q_model


def quantmobilenetv1_diana_naive5(arch_cfg_path, **kwargs):
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

    archws[0] = [8]
    archws[-1] = [8]

    kwargs.pop('analog_speedup', 5.)
    model = MobileNetV1(qm.QuantMultiPrecActivConv2d, hw.diana_naive(5.),
                        archws, archas, qtz_fc='multi', bn=False, **kwargs)
    utils.init_scale_param(model)

    return _quantmobilenetv1_diana(arch_cfg_path, model, **kwargs)


def quantmobilenetv1_diana_naive10(arch_cfg_path, **kwargs):
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

    archws[0] = [8]
    archws[-1] = [8]

    kwargs.pop('analog_speedup', 5.)
    model = MobileNetV1(qm.QuantMultiPrecActivConv2d, hw.diana_naive(10.),
                        archws, archas, qtz_fc='multi', bn=False, **kwargs)
    utils.init_scale_param(model)

    return _quantmobilenetv1_diana(arch_cfg_path, model, **kwargs)


def quantmobilenetv1_diana_full(arch_cfg_path, **kwargs):
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

    archws[0] = [8]
    archws[-1] = [8]

    kwargs.pop('analog_speedup', 5.)
    model = MobileNetV1(qm.QuantMultiPrecActivConv2d, hw.diana(),
                        archws, archas, qtz_fc='multi', bn=False, **kwargs)
    utils.init_scale_param(model)

    return _quantmobilenetv1_diana(arch_cfg_path, model, **kwargs)


def quantmobilenetv1_pow2_diana_full(arch_cfg_path, **kwargs):
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

    archws[0] = [8]
    archws[-1] = [8]

    kwargs.pop('analog_speedup', 5.)
    model = MobileNetV1(qm2.QuantMultiPrecActivConv2d, hw.diana(),
                        archws, archas, qtz_fc='multi', bn=False, **kwargs)
    utils.init_scale_param(model)

    return _quantmobilenetv1_diana(arch_cfg_path, model, **kwargs)


def quantmobilenetv1_diana_reduced(arch_cfg_path, **kwargs):
    is_searchable = utils.detect_ad_tradeoff(  # TODO: to be tested!!
        quantmobilenetv1_fp(None, pretrained=False),
        torch.rand((1, 3, 96, 96)))

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

    archws[0] = [8]
    archws[-1] = [8]

    kwargs.pop('analog_speedup', 5.)
    model = MobileNetV1(qm.QuantMultiPrecActivConv2d, hw.diana(),
                        archws, archas, qtz_fc='multi', bn=False, **kwargs)
    utils.init_scale_param(model)

    return _quantmobilenetv1_diana(arch_cfg_path, model, **kwargs)


def _quantmobilenetv1_diana(arch_cfg_path, model, **kwargs):
    if kwargs.get('fine_tune', True):
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict)
    else:
        # Load only alphas weights
        alpha_state_dict = _load_alpha_state_dict(arch_cfg_path)
        model.load_state_dict(alpha_state_dict, strict=False)
    return model


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


def _load_arch_multi_prec(arch_path):
    checkpoint = torch.load(arch_path)
    state_dict = checkpoint['state_dict']
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
