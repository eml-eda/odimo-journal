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

from typing import Type, Tuple

import torch
import torch.nn as nn
import torch.fx as fx

import deployment.observer as obs
from deployment.utils import IntegerizationMode

import models.quant_module_pow2 as qm

__all__ = [
    'build_qgraph',
]


A_MIN = -128
A_MAX = 127
ASH_MAX = 16
DSH_MAX = 32


def _compute_analog_bparams(s_w, s_x, s_y, b_16, n_sh, lut):
    b_8 = []
    n_b = []
    n_ch = s_w.shape[0]
    device = s_w.device
    for idx in range(n_ch):
        diff = torch.abs(
            lut -
            (2**n_sh[idx] * b_16[idx] * s_w[idx] * s_x / s_y))
        argmin = (diff == diff.amin()).nonzero(as_tuple=True)
        n_b.append(argmin[0] if len(argmin[0]) == 1 else argmin[0][0])
        b_8.append(argmin[1] + A_MIN if len(argmin[1]) == 1 else argmin[1][0] + A_MIN)
    return torch.tensor(b_8, device=device), torch.tensor(n_b, device=device)


def _compute_analog_bparams_v2_old(alpha, b_16, lut):
    b_8 = []
    n_b = []
    n_ch = alpha.shape[0]
    device = alpha.device
    for idx in range(n_ch):
        diff = torch.abs(
            lut -
            (alpha[idx] * b_16[idx]))
        argmin = (diff == diff.amin()).nonzero(as_tuple=True)
        n_b.append(argmin[0] if len(argmin[0]) == 1 else argmin[0][0])
        b_8.append(argmin[1] + A_MIN if len(argmin[1]) == 1 else argmin[1][0] + A_MIN)
    return torch.tensor(b_8, device=device), torch.tensor(n_b, device=device)


# def _compute_analog_wparams(s_w, s_x, s_y, max_act, lut):
#     alpha = []
#     n_sh = []
#     n_ch = s_w.shape[0]
#     device = max_act.device
#     for idx in range(n_ch):
#         if max_act[idx]:
#             mask_lut = torch.cat((
#                 torch.arange(A_MIN, 0).to(device) > 2**15 / max_act[idx],
#                 torch.arange(1, A_MAX+1).to(device) < 2**15 / max_act[idx])
#                 )
#             diff = torch.abs(lut[:, mask_lut] - (s_w[idx] * s_x / s_y))
#         else:
#             diff = torch.abs(lut - (s_w[idx] * s_x / s_y))
#         argmin = (diff == diff.amin()).nonzero(as_tuple=True)
#         n_sh.append(argmin[0] if len(argmin[0]) == 1 else argmin[0][0])
#         alpha.append(argmin[1] + A_MIN if len(argmin[1]) == 1 else argmin[1][0] + A_MIN)
#         if alpha[-1] >= 0:
#             alpha[-1] += 1
#
#     return torch.tensor(alpha, device=device), torch.tensor(n_sh, device=device)


def _compute_analog_wparams_old(s_w, s_x, s_y, max_act, lut):
    alpha = []
    n_sh = []
    n_ch = s_w.shape[0]
    device = max_act.device
    for idx in range(n_ch):
        if 0:  # if max_act[idx]:
            mask_lut = torch.arange(1, A_MAX+1).to(device) < 2**15 / abs(max_act[idx])
            diff = torch.abs(lut[:, mask_lut] - (s_w[idx] * s_x / s_y))
        else:
            diff = torch.abs(lut - (s_w[idx] * s_x / s_y))
        argmin = (diff == diff.amin()).nonzero(as_tuple=True)
        # print(f'Approx Error: {100 * (diff.amin() / (s_w[idx] * s_x / s_y)).item()}%')
        n_sh.append(argmin[0] if len(argmin[0]) == 1 else argmin[0][0])
        alpha.append(argmin[1] + 1 if len(argmin[1]) == 1 else argmin[1][0] + 1)

    return torch.tensor(alpha, device=device), torch.tensor(n_sh, device=device)


def _compute_digital_wparams_old(s_w, s_x, s_y, lut):
    # n_sh = []
    # device = s_w.device
    diff = torch.abs(lut - (s_w * s_x / s_y))
    argmin = (diff == diff.amin()).nonzero(as_tuple=True)
    print(f'Approx Error: {100 * (diff.amin() / (s_w * s_x / s_y)).item()}%')
    # n_sh.append(argmin[0] if len(argmin[0]) == 1 else argmin[0][0])

    # return argmin[0]
    return argmin[0][0], argmin[1][0]+1  # to be removed


def _integer_approximation(target, sh_b, alpha_b=0):
    device = target.device
    target = target.clone().detach().cpu()

    params = []
    sh_list = []
    alpha_list = []
    upper_bound = 2**(alpha_b - 1) - 1 if alpha_b != 0 else 1

    for idx in range(len(target)):
        for sh in range(sh_b):
            params.append(
                [sh, _binary_search(2**-sh, 1, upper_bound, target[idx])]
                )
        min_diff = float('inf')
        min_sh, min_alpha = None, None
        for param in params:
            sh, alpha = param[0], param[1]
            diff = abs(alpha/2**sh - target[idx])
            if diff < min_diff:
                min_diff = diff
                min_sh, min_alpha = sh, alpha
        sh_list.append(min_sh)
        alpha_list.append(min_alpha)
        print(f'Approx Error: {(100 * min_diff / target[idx]).item()}%')
    if len(sh_list) == 1:
        return sh_list[0], alpha_list[0]
    else:
        return torch.tensor(sh_list, device=device), torch.tensor(alpha_list, device=device)


def _integer_approximation_bias(target, sh_b, alpha_b=0):
    device = target.device
    target = target.clone().detach().cpu()

    params = []
    sh_list = []
    alpha_list = []
    upper_bound = 2**(alpha_b - 1) - 1 if alpha_b != 0 else 1

    for idx in range(len(target)):
        for sh in range(sh_b):
            params.append(
                [sh, _binary_search(2**sh, -upper_bound-1, upper_bound, target[idx])]
                )
        min_diff = float('inf')
        min_sh, min_alpha = None, None
        for param in params:
            sh, alpha = param[0], param[1]
            diff = abs(alpha*2**sh - target[idx])
            if diff < min_diff:
                min_diff = diff
                min_sh, min_alpha = sh, alpha
        sh_list.append(min_sh)
        alpha_list.append(min_alpha)
        # print(f'Approx Error: {(100 * min_diff / target[idx]).item()}%')
    if len(sh_list) == 1:
        return sh_list[0], alpha_list[0]
    else:
        return torch.tensor(sh_list, device=device), torch.tensor(alpha_list, device=device)


def _binary_search(div, low, high, x):
    if high != low:
        mid = (low + high) // 2
        # diff_mid = abs(mid*div - x)
        # diff_high = abs(high*div - x)

        # if diff_mid < diff_high:
        if x == mid*div:
            return mid
        if x < mid*div:
            return _binary_search(div, low, mid-1, x)
        else:
            return _binary_search(div, mid+1, high, x)
    else:
        return low


def _compute_b_fakeint(b_8, n_b, alpha):
    b_fakeint = b_8 * 2**n_b / alpha
    return b_fakeint


def _compute_s_y_fakeint(alpha, n_sh, s_x_fakeint, s_w):
    s_y_fakeint = (s_w * s_x_fakeint * 2**n_sh) / alpha
    return s_y_fakeint


def _extract_qinfo(module):
    s_x, s_w, b_16 = dict(), dict(), dict()

    # q_a = module.mix_activ.mix_activ[0]
    for q_a in module.mix_activ.mix_activ:
        s_x[q_a.num_bits] = q_a.clip_val / (2**q_a.num_bits - 1)
        # s_x[q_a.num_bits] = torch.exp2(-torch.log2((2**q_a.num_bits - 1) / q_a.clip_val).floor())

    # q_w = module.mix_weight.mix_weight[0]
    for q_w in module.mix_weight.mix_weight:
        # s_w[q_w.num_bits] = torch.exp(q_w.scale_param) / (2**(q_w.num_bits - 1) - 1)
        if q_w.num_bits != 2:
            s_w[q_w.num_bits] = torch.exp2(
                torch.log2(q_w.scale_param).floor()) / (2**(q_w.num_bits - 1) - 1)
        else:
            s_w[q_w.num_bits] = torch.exp(q_w.scale_param) / (2**(q_w.num_bits - 1) - 1)
        b_16[q_w.num_bits] = module.mix_weight.conv.bias
    return s_x, s_w, b_16


class QuantizationTracer(fx.Tracer):
    """Consider layers contained in `target_layers` as leaf modules.

    :param target_layers: modules that should be considered as a leaf
    :type target_layers: tuple[Type[nn.Module]
    """

    def __init__(self, target_layers: Tuple[Type[nn.Module], ...]):
        super().__init__()
        self.target_layers = target_layers

    def is_leaf_module(
        self,
        m: nn.Module,
        module_qualified_name: str
    ) -> bool:
        if isinstance(m, self.target_layers):
            return True
        elif isinstance(m, obs.ObserverBase):
            return True
        elif isinstance(m, qm.QuantPaCTActiv):
            return True
        else:
            return m.__module__.startswith('torch.nn') and \
                not isinstance(m, torch.nn.Sequential)


# MR: questa funzione si puo' specializzare poi per diversi backend
def build_qgraph(
    model: nn.Module,
    output_classes: int,
    target_layers: Tuple[Type[nn.Module], ...],
    mode: IntegerizationMode
) -> nn.Module:
    """
    Performs the following steps traversing from output to input:
        1. Integerize weights and biases
        2. Propagate scale factors and compute integer quantization params and
        annotate nodes.
        3. Convert fake-quantized layer with integer counterparts

    :param model: nn.Module whit quantization information
    :type model: nn.Module
    :param output_classes: number of output classes
    :type output_classes: int
    :param target_layers: set of nn.Module where quantization information
    should be extracted
    :type target_layers: tuple[Type[nn.Module], ...]
    :param mode: integerization mode. Use `IntegerizationMode.Int` or
    `IntegerizationMode.FakeInt`
    :type mode: IntegerizationMode
    :return: a `model` copy with annotated quantization information
    :rtype: nn.Module
    """

    if mode not in list(IntegerizationMode):
        err_msg = f'Supported `IntegerizationMode` are {list(IntegerizationMode)}'
        raise ValueError(err_msg)

    tracer = QuantizationTracer(target_layers)
    graph = tracer.trace(model.eval())
    name = model.__class__.__name__
    mod = fx.GraphModule(tracer.root, graph, name)
    modules = dict(mod.named_modules())
    device = next(model.parameters()).device

    # TODO: find better way than hard-coding 7 for act
    s_y = {7: torch.tensor((1,), device=device)}
    s_y_fakeint = torch.tensor((1,), device=device)

    # lut_analog_w = torch.tensor([
    #                 [a / 2**sh for a in range(A_MIN, A_MAX+1) if a != 0]
    #                 for sh in range(SH_MAX+1)],
    #                 device=device)
    lut_analog_w = torch.tensor([
                    [a / 2**sh for a in range(1, A_MAX+1)]
                    for sh in range(ASH_MAX)],
                    device=device)
    lut_analog_b = torch.tensor([
                    [b * 2**sh for b in range(A_MIN, A_MAX+1)]
                    for sh in range(ASH_MAX)],
                    device=device)
    # lut_digital_w = torch.tensor([1 / 2**sh for sh in range(DSH_MAX)],
    #                              device=device)
    # lut_digital_w = torch.tensor([
    #                 [a / 2**sh for a in range(1, A_MAX+1)]  # a to be removed
    #                 for sh in range(DSH_MAX)],
    #                 device=device)

    # Backward - Annotate graph
    for n in reversed(mod.graph.nodes):
        m = modules.get(n.target)
        if isinstance(m, target_layers):
            with torch.no_grad():
                if isinstance(m, qm.QuantAvgPool2d):
                    s_x = dict()
                    for q_a in m.mix_activ.mix_activ:
                        s_x[q_a.num_bits] = q_a.clip_val / (2**q_a.num_bits - 1)
                    n.meta['s_x'] = s_x[7]
                    n.meta['s_y'] = s_y[7]
                elif isinstance(m, qm.QuantAdd):
                    s_x = dict()
                    for q_a in m.mix_activ.mix_activ:
                        s_x[q_a.num_bits] = q_a.clip_val / (2**q_a.num_bits - 1)
                    n.meta['s_x'] = s_x[7]

                    target = s_x[7] / s_y[7]
                    n_sh, alpha = _integer_approximation(target,
                                                         sh_b=32, alpha_b=3)
                    n.meta['alpha'] = alpha
                    n.meta['n_sh'] = n_sh
                else:
                    s_x, s_w, b_16 = _extract_qinfo(m)
                    n.meta['s_x'] = s_x
                    n.meta['s_w'] = s_w
                    n.meta['b_16'] = b_16
                    alpha, n_sh = dict(), dict()
                    b_8, n_b = dict(), dict()
                    for wbit in m.wbits:
                        if wbit == 2:
                            max_act = modules.get(n.next.target).max_val
                            alpha[wbit], n_sh[wbit] = \
                                _compute_analog_wparams_old(s_w[wbit], s_x[7], s_y[7], max_act,
                                                            lut_analog_w)
                            # target = s_w * s_x / s_y
                            # n_sh, alpha = _integer_approximation(target, sh_b=16, alpha_b=8)
                            # b_8, n_b = _compute_analog_bparams(s_w, s_x, s_y, b_16, n_sh,
                            #                                    lut_analog_b)
                            b_8[wbit], n_b[wbit] = \
                                _compute_analog_bparams_v2_old(alpha[wbit], b_16[wbit],
                                                               lut_analog_b)
                            # target = alpha * b_16
                            # n_b, b_8 = _integer_approximation_bias(target, sh_b=8, alpha_b=8)
                        elif wbit == 8:
                            # alpha to be removed
                            # n_sh_old, alpha_old = _compute_digital_wparams_old(s_w, s_x, s_y,
                            #                                                    lut_digital_w)
                            # TODO: find better way than hard-coding 7 for act
                            target = s_w[wbit] * s_x[7] / s_y[7]
                            n_sh[wbit], alpha[wbit] = _integer_approximation(target,
                                                                             sh_b=32, alpha_b=0)
                        else:
                            raise ValueError('2 and 8 are only supported wbits')
                    n.meta['alpha'] = alpha
                    n.meta['n_sh'] = n_sh
                    n.meta['b_8'] = b_8
                    n.meta['n_b'] = n_b
                if 'downsample' not in n.target:  # TODO: dirty, fix
                    s_y = s_x  # propagate s_x backward

    # Forward - Transform graph
    first_layer = True
    for n in mod.graph.nodes:
        m = modules.get(n.target)
        if isinstance(m, target_layers+(qm.QuantPaCTActiv,)):
            with torch.no_grad():
                # Knowing which one is the first layer is needed for autoconvert
                if first_layer:
                    n.meta['first'] = True
                    first_layer = False
                else:
                    n.meta['first'] = False

                if mode is IntegerizationMode.FakeInt:
                    if n.meta['first']:
                        s_y_fakeint = n.meta['s_x'][7]
                    n.meta['s_x_fakeint'] = s_y_fakeint
                    # Compute new s_y_fakeint
                    if isinstance(m, qm.QuantAvgPool2d):
                        s_y_fakeint = n.meta['s_x']
                    elif isinstance(m, qm.QuantAdd):
                        s_y_fakeint = 2**n.meta['n_sh'] * n.meta['s_x_fakeint'] \
                            / n.meta['alpha']  # 's_x_fakeint' or 's_x' ??
                    elif isinstance(m, qm.QuantPaCTActiv):
                        continue
                    else:
                        p = m.wbits[0]
                        if m.wbits == [2]:
                            if 'downsample' not in n.target:  # TODO: dirty, fix
                                s_y_fakeint = 2**n.meta['n_sh'][p] * n.meta['s_w'][p] * \
                                    n.meta['s_x'][p] / n.meta['alpha'][p]
                            n.meta['bias'] = 2**n.meta['n_b'][p] * n.meta['b_8'][p] / \
                                n.meta['alpha'][p]
                        elif m.wbits == [8]:
                            if 'downsample' not in n.target:  # TODO: dirty, fix
                                s_y_fakeint = 2**n.meta['n_sh'][p] * \
                                    n.meta['s_w'][p] * \
                                    n.meta['s_x_fakeint'] / \
                                    n.meta['alpha'][p]  # 's_x_fakeint'or's_x'?
                            if n.meta['b_16'] is not None:
                                # n.meta['bias'] = 2**n.meta['n_sh'] * n.meta['b_16'] * \
                                #     n.meta['s_w'] * n.meta['s_x']  # 's_x_fakeint' or 's_x' ??
                                n.meta['bias'] = n.meta['b_16'][p] * n.meta['s_w'][p] \
                                    * n.meta['s_x_fakeint']  # 's_x_fakeint' or 's_x' ??
                            else:
                                n.meta['bias'] = None

                m.autoconvert(n, mod, mode)
        # elif isinstance(m, qm.QuantPaCTActiv):
        #     # new_submodule = nn.Identity()
        #     # mod.add_submodule(str(n.target), new_submodule)
        #     mod.delete_submodule(n.target)
        #     # _erase_node(n, mod.graph)
        #     # mod.graph.erase_node(n)

    mod.delete_all_unused_submodules()
    mod.graph.lint()
    mod.recompile()
    return mod


def _erase_node(n, graph):
    users_list = list(n.users)
    length = len(users_list)
    if length > 0:
        while length > 0:
            _erase_node(users_list[length-1], graph)
            length -= 1
    else:
        graph.erase_node(n)
