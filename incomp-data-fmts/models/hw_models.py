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

# DISCLAIMER:
# The integration of different HW models is currently not impemented,
# the proposed MPIC model is only an example but the current implementation
# directly support only `diana`

# TODO: Understand how model changes for deptwhise conv.
#       At this time groups is not taken into account!

import math
import torch

MPIC = {
    2: {2: 6.5, 4: 4., 8: 2.2},
    4: {2: 3.9, 4: 3.5, 8: 2.1},
    8: {2: 2.5, 4: 2.3, 8: 2.1},
}

DIANA_NAIVE = {
    'digital': 1.0,
    'analog': 5.0,  # Default SpeedUp
}

F = 260000000  # Hz


class ComputeOxUnrollSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ch_eff, ch_in, k_x, k_y):
        device = ch_eff.device
        ox_unroll_list = [1, 2, 4, 8]
        ox_unroll = torch.as_tensor(ox_unroll_list, device=device)
        ch_in_unroll = max(64, ch_in)
        mask_out = ox_unroll * ch_eff <= 512
        mask_in = (ox_unroll + k_x - 1) * ch_in_unroll * k_y <= 1152
        mask = torch.logical_and(mask_out, mask_in)
        mask[0] = True
        return ox_unroll[mask][-1]

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class FloorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ch, N):
        return torch.floor((ch + N - 1) / N)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class GateSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ch, th):
        ctx.save_for_backward(ch, torch.tensor(th))
        return (ch >= th).float()

    @staticmethod
    def backward(ctx, grad_output):
        ch, th = ctx.saved_tensors
        grad = grad_output.clone()
        grad = 1 / (grad + 1)  # smooth step grad with log derivative
        grad.masked_fill_(ch.le(0), 0)
        grad.masked_fill_(ch.ge(th.data), 0)
        return grad, None


def _analog_cycles(**kwargs):
    ch_in = kwargs['ch_in']
    ch_eff = kwargs['ch_out']
    k_x = kwargs['k_x']
    k_y = kwargs['k_y']
    groups = kwargs['groups']
    if groups != 1:
        msg = f'groups={groups}. Analog accelerator supports only groups=1'
        raise ValueError(msg)
    out_x = kwargs['out_x']
    out_y = kwargs['out_y']
    ox_unroll_base = ComputeOxUnrollSTE.apply(ch_eff, ch_in, k_x, k_y)
    cycles_comp = FloorSTE.apply(ch_eff, 512) * _floor(ch_in, 128) * out_x * out_y / ox_unroll_base
    # cycles_weights = 4 * 2 * 1152
    cycles_weights = 4 * 2 * ch_in * k_x * k_y
    cycles_comp_norm = cycles_comp * 70 / (1000000000 / F)
    gate = GateSTE.apply(ch_eff, 1.)
    return (gate * cycles_weights) + cycles_comp_norm
    # return cycles_weights + cycles_comp_norm


def _digital_cycles(**kwargs):
    ch_in = kwargs['ch_in']
    ch_eff = kwargs['ch_out']
    groups = kwargs.get('groups', 1)
    k_x = kwargs['k_x']
    k_y = kwargs['k_y']
    out_x = kwargs['out_x']
    out_y = kwargs['out_y']

    # Original model (no depthwise):
    # cycles = FloorSTE.apply(ch_eff, 16) * ch_in * _floor(out_x, 16) * out_y * k_x * k_y
    # Depthwise support:
    # min(ch_eff, groups) * FloorSTE.apply(1, 16) * 1 * _floor(out_x, 16) * out_y * k_x * k_y
    cycles = FloorSTE.apply(ch_eff / groups, 16) * ch_in * _floor(out_x, 16) * out_y * k_x * k_y

    # Works with both depthwise and normal conv:
    cycles_load_store = out_x * out_y * (ch_eff + ch_in) / 8

    gate = GateSTE.apply(ch_eff, 1.)
    return (gate * cycles_load_store) + cycles
    # return cycles_load_store + cycles


def _floor(ch, N):
    return math.floor((ch + N - 1) / N)


def mpic_model(a_bit, w_bit):
    return MPIC[a_bit][w_bit]


def diana_naive(analog_speedup=5.):

    def diana_model(accelerator, **kwargs):
        ch_in = kwargs['ch_in']
        ch_eff = kwargs['ch_out']
        groups = kwargs.get('groups', 1)
        k_x = kwargs['k_x']
        k_y = kwargs['k_y']
        out_x = kwargs['out_x']
        out_y = kwargs['out_y']
        mac = ch_in * (ch_eff / groups) * k_x * k_y * out_x * out_y
        DIANA_NAIVE['analog'] = float(analog_speedup)  # Update SpeedUp
        if accelerator in DIANA_NAIVE.keys():
            return mac / DIANA_NAIVE[accelerator]
        else:
            raise ValueError(f'Unknown accelerator: {accelerator}')

    return diana_model


def diana(**kwargs):

    def diana_model(accelerator, **kwargs):
        if accelerator == 'analog':
            return _analog_cycles(**kwargs)
        elif accelerator == 'digital':
            return _digital_cycles(**kwargs)
        else:
            raise ValueError(f'Unknown accelerator: {accelerator}')

    return diana_model


class DianaPower:
    def __init__(self):
        self.p_dig = 24.96  # [mW]
        self.p_ana = 28.74  # [mW]
        self.p_hyb = 42.39  # [mW]
