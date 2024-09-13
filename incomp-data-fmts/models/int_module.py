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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import models.quant_module as qm

__all__ = [
    'FakeIntMultiPrecActivConv2d',
    'IntMultiPrecActivConv2d',
]


class ClippedLinearQuantizeSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, num_bits, act_scale, dequantize=False):
        clip_val = (2**num_bits - 1) * act_scale
        out = torch.clamp(x, 0, clip_val.data[0])
        # out = torch.max(torch.min(x, clip_val), torch.tensor(0.))
        out_q = torch.floor(out / act_scale)
        if dequantize:
            out_q = out_q * act_scale
        return out_q

    @staticmethod
    def backward(ctx, grad_output):
        # Need to do something here??
        return grad_output, None, None, None


class ClippedLinearQuantization(nn.Module):

    def __init__(self, num_bits, act_scale, dequantize=False):
        super().__init__()
        self.num_bits = num_bits[0]
        self.act_scale = act_scale
        self.dequantize = dequantize

    def forward(self, x):
        x_q = ClippedLinearQuantizeSTE.apply(x, self.num_bits, self.act_scale, self.dequantize)
        return x_q


class IntPaCTActiv(nn.Module):

    def __init__(self, abits, act_scale, dequantize=False):
        super().__init__()
        self.abits = abits
        self.act_scale = act_scale
        self.dequantize = dequantize

        self.quantizer = ClippedLinearQuantization(num_bits=abits,
                                                   act_scale=act_scale,
                                                   dequantize=dequantize)

    def forward(self, x):
        x_q = self.quantizer(x)
        return x_q


class IntAdd(nn.Module):

    def __init__(self, int_params):
        super().__init__()
        self.alpha = int_params['alpha']
        self.n_sh = int_params['n_sh']

    def forward(self, act_x, act_y):
        act_qx = torch.min(  # ReLU127
                    torch.max(torch.tensor(0.), act_x),
                    torch.tensor(127.))
        act_qy = torch.min(  # ReLU127
                    torch.max(torch.tensor(0.), act_y),
                    torch.tensor(127.))
        act_qx = torch.floor(act_qx)
        act_qy = torch.floor(act_qy)

        out = act_qx + act_qy
        scale_out = self.alpha * out / 2**self.n_sh

        return scale_out


class IntAvgPool2d(nn.Module):

    def __init__(self, int_params, **kwargs):
        super().__init__()
        self.s_x = int_params['s_x']
        self.s_y = int_params['s_y']

        self.pool = nn.AvgPool2d(**kwargs)

    def forward(self, act_in):
        act_q = torch.min(  # ReLU127
                torch.max(torch.tensor(0.), act_in),
                torch.tensor(127.))
        act_q = torch.floor(act_q)
        out = self.pool(act_q)
        scaled_out = out * self.s_x / self.s_y
        return scaled_out


class IntMultiPrecActivConv2d(nn.Module):

    def __init__(self, int_params, abits, wbits, **kwargs):
        super().__init__()
        self.act_scale = int_params['s_x'][7]  # TODO: remove hard-code
        self.abits = abits
        self.wbits = wbits
        self.first_layer = int_params['first']

        if self.first_layer:
            self.mix_activ = IntPaCTActiv(abits, self.act_scale)
        self.mix_weight = IntMultiPrecConv2d(int_params, wbits, **kwargs)
        # self.conv = nn.Conv2d(**kwargs)

    def forward(self, act_in):
        if self.first_layer:
            act_q = self.mix_activ(act_in)
        else:
            act_q = torch.min(  # ReLU127
                    torch.max(torch.tensor(0.), act_in),
                    torch.tensor(127.))
            act_q = torch.floor(act_q)
        act_out = self.mix_weight(act_q)
        return act_out


class IntMultiPrecConv2d(nn.Module):

    def __init__(self, int_params, wbits, **kwargs):
        super().__init__()
        self.bits = wbits

        self.b_16 = int_params['b_16']
        self.n_sh = int_params['n_sh']
        self.alpha = int_params['alpha']
        self.b_8 = int_params['b_8']
        self.n_b = int_params['n_b']

        self.cout = kwargs['out_channels']

        self.alpha_weight = Parameter(torch.Tensor(len(self.bits), self.cout),
                                      requires_grad=False)
        self.alpha_weight.data.fill_(0.01)

        self.conv = nn.Conv2d(**kwargs)

    def forward(self, x):
        out = []
        sw = F.one_hot(torch.argmax(self.alpha_weight, dim=0),
                       num_classes=len(self.bits)).t()
        conv = self.conv
        weight = conv.weight
        for i, bit in enumerate(self.bits):
            eff_weight = weight * sw[i].view((self.cout, 1, 1, 1))
            if bit == 2:
                conv_out = F.conv2d(
                    x, eff_weight, None, conv.stride,
                    conv.padding, conv.dilation, conv.groups)
                alpha = self.alpha[bit].view(1, self.cout, 1, 1)
                b = (self.b_8[bit] * 2**self.n_b[bit]).view(1, self.cout, 1, 1)
                shift = (2**self.n_sh[bit]).view(1, self.cout, 1, 1)
                scale_out = (alpha * conv_out + b) / shift
                eff_out = scale_out * sw[i].view((1, self.cout, 1, 1))
                out.append(eff_out)
            elif bit == 8:
                if self.b_16[bit] is not None:
                    eff_bias = self.b_16[bit] * sw[i].view(self.cout)
                else:
                    eff_bias = None
                conv_out = F.conv2d(
                    x, eff_weight, eff_bias, conv.stride,
                    conv.padding, conv.dilation, conv.groups)
                scale_out = self.alpha[bit] * conv_out / 2**self.n_sh[bit]  # to be removed
                # out.append(conv_out / 2**self.n_sh)
                eff_out = scale_out * sw[i].view((1, self.cout, 1, 1))
                out.append(eff_out)  # to be removed
        out = sum(out)

        return out


class FakeIntAdd(nn.Module):

    def __init__(self, int_params, abits):
        super().__init__()
        self.abits = abits
        self.act_scale = int_params['s_x_fakeint']

        self.mix_activ = IntPaCTActiv(abits, self.act_scale, dequantize=True)

    def forward(self, act_x, act_y):
        act_qx = self.mix_activ(act_x)
        act_qy = self.mix_activ(act_y)

        out = act_qx + act_qy

        return out


class FakeIntAvgPool2d(nn.Module):

    def __init__(self, int_params, abits, **kwargs):
        super().__init__()
        self.abits = abits
        self.act_scale = int_params['s_x_fakeint']

        self.mix_activ = IntPaCTActiv(abits, self.act_scale, dequantize=True)
        self.pool = nn.AvgPool2d(**kwargs)

    def forward(self, act_in):
        act_q = self.mix_activ(act_in)
        out = self.pool(act_q)
        return out


class FakeIntMultiPrecActivConv2d(nn.Module):

    def __init__(self, int_params, abits, wbits, **kwargs):
        super().__init__()
        self.abits = abits
        self.wbits = wbits

        self.act_scale = int_params['s_x_fakeint']

        self.mix_activ = IntPaCTActiv(abits, self.act_scale, dequantize=True)
        self.mix_weight = FakeIntMultiPrecConv2d(int_params, wbits, **kwargs)

    def forward(self, act_in):
        act_q = self.mix_activ(act_in)
        act_out = self.mix_weight(act_q)
        return act_out


class FakeIntMultiPrecConv2d(nn.Module):

    def __init__(self, int_params, wbits, **kwargs):
        super().__init__()
        self.bits = wbits

        self.s_w = int_params['s_w']
        self.bias = int_params['bias']

        self.cout = kwargs['out_channels']

        if isinstance(kwargs['kernel_size'], tuple):
            k_size = kwargs['kernel_size'][0] * kwargs['kernel_size'][1]
        else:
            k_size = kwargs['kernel_size'] * kwargs['kernel_size']

        self.alpha_weight = Parameter(torch.Tensor(len(self.bits), self.cout),
                                      requires_grad=False)
        self.alpha_weight.data.fill_(0.01)

        # Quantizer
        self.mix_weight = nn.ModuleList()
        for idx, bit in enumerate(self.bits):
            self.mix_weight.append(
                qm.FQConvWeightQuantization(
                    self.cout, k_size,
                    num_bits=bit,
                    train_scale_param=False))
            # s_w = torch.exp(q_w.scale_param) / (2**(q_w.num_bits - 1) - 1)
            scale_param = torch.log(self.s_w[bit] * (2**(bit - 1) - 1))
            self.mix_weight[idx].scale_param.copy_(scale_param)

        self.conv = nn.Conv2d(**kwargs)

    def forward(self, x):
        mix_quant_weight = list()
        mix_quant_bias = list()
        sw = F.one_hot(torch.argmax(self.alpha_weight, dim=0),
                       num_classes=len(self.bits)).t()
        conv = self.conv
        weight = conv.weight
        for i, bit in enumerate(self.bits):
            quant_weight = self.mix_weight[i](weight)
            scaled_quant_weight = quant_weight * sw[i].view((self.cout, 1, 1, 1))
            mix_quant_weight.append(scaled_quant_weight)
            if self.bias is not None:
                eff_bias = self.bias * sw[i].view(self.cout)
                mix_quant_bias.append(eff_bias)
        if mix_quant_bias:
            mix_quant_bias = sum(mix_quant_bias)
        else:
            mix_quant_bias = None
        mix_quant_weight = sum(mix_quant_weight)
        out = F.conv2d(
            x, mix_quant_weight, mix_quant_bias, conv.stride,
            conv.padding, conv.dilation, conv.groups)

        return out


class ResidualBranch(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x, None
