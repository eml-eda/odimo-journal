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

import math

import torch


POWER_DARKSIDE = {
    'GAP8': 168,
    'DWE': 118,
    'IDLE': 1.32,
}


class FloorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ch, N):
        return torch.floor((ch + N - 1) / N)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def _floor(ch, N):
    return math.floor((ch + N - 1) / N)


def cost_naive(layer_shapes):
    # Unpack input dict
    o_x = layer_shapes['o_x']
    o_y = layer_shapes['o_y']
    c_in = layer_shapes['c_in']
    c_out = layer_shapes['c_out']
    k_x = layer_shapes['k_x']
    k_y = layer_shapes['k_y']
    groups = layer_shapes['groups']
    # Eval cost
    cost = o_x * o_y * c_in * c_out * k_x * k_y / groups
    return cost


def cost_darkside(layer_shapes):
    # Unpack input dict
    o_x = layer_shapes['o_x']
    o_y = layer_shapes['o_y']
    c_in = layer_shapes['c_in']
    c_out = layer_shapes['c_out']
    k_x = layer_shapes['k_x']
    k_y = layer_shapes['k_y']
    groups = layer_shapes['groups']
    # Eval cost
    if groups == 1:  # GAP8
        iterations = _floor(o_y, 2) * _floor(o_x, 8)
        im2col = k_x * k_y * c_in * 2
        matmul = FloorSTE.apply(c_out, 4) * (14 * _floor(k_x * k_y * c_in, 4) + 15)
        cost = iterations * (im2col + matmul)
    else:  # DWE
        iterations = FloorSTE.apply(c_out, 16)
        load_i_w = 9 + o_x * 9
        compute = o_x * o_y * 4
        cost = iterations * (load_i_w + compute)
    return cost
