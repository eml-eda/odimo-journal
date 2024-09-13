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

import argparse
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.passes.graph_drawer import FxGraphDrawer

from models.model_diana import analog_cycles, digital_cycles
from models.quant_resnet import quantres8_fp, quantres20_fp, \
    quantres18_fp, quantres18_fp_reduced, quantres18_fp_c100
from models.quant_mobilenetv1 import quantmobilenetv1_fp
from models.hw_models import DianaPower

power = DianaPower()

_ARCH_FUNC = {
    'mobilenetv1': quantmobilenetv1_fp,
    'resnet8': quantres8_fp,
    'resnet20': quantres20_fp,
    'resnet18-64': quantres18_fp,
    'resnet18-64-reduced': quantres18_fp_reduced,
    'resnet18-224': quantres18_fp,
    'resnet18-32': quantres18_fp_c100,
}

_INP_SHAPE = {
    'mobilenetv1': (1, 3, 96, 96),
    'resnet8': (1, 3, 32, 32),
    'resnet20': (1, 3, 32, 32),
    'resnet18-64': (1, 3, 64, 64),
    'resnet18-64-reduced': (1, 3, 64, 64),
    'resnet18-224': (1, 3, 224, 224),
    'resnet18-32': (1, 3, 32, 32),
}

_BAR_WIDTH = 0.1


def profile_cycles(arch,
                   input_shape='default',
                   pretrained_arch=None,
                   plot_layer_cycles=False,
                   plot_layer_power=False,
                   plot_disc_net=False,
                   plot_net_graph=False,
                   save_pkl=False):
    model = _ARCH_FUNC[arch](None)
    if input_shape == 'default':
        dummy_input = torch.randn(_INP_SHAPE[arch])
    else:
        img_size = tuple(input_shape, input_shape)
        inp_shape = (1, 3) + img_size
        dummy_input = torch.randn(inp_shape)
    gm = fx.symbolic_trace(model)
    modules = dict(gm.named_modules())
    arch_details = dict()

    if pretrained_arch is not None:
        path = Path(pretrained_arch)
        if not path.exists():
            raise ValueError(f'{path} does not exists!')
        state_dict = torch.load(path)['state_dict']
        strength = path.parts[-3].split('_')[-1]

    if plot_net_graph:
        gd = FxGraphDrawer(gm, str(arch))
        gd.get_dot_graph().write_png(f'{str(arch)}_graph.png')
        print(f'Graph Plot saved @ {str(arch)}_graph.png', end='\n')

    ShapeProp(gm).propagate(dummy_input)

    # Build dict with layers' shapes and cycles
    for node in gm.graph.nodes:
        if node.target in modules.keys():
            if isinstance(modules[node.target], nn.Conv2d):
                name = '.'.join(node.target.split('.')[:-1])
                conv = modules[node.target]
                out_shape = node.meta['tensor_meta'].shape
                ch_in = conv.in_channels
                ch_out = conv.out_channels
                groups = conv.groups
                k_x = conv.kernel_size[0]
                k_y = conv.kernel_size[1]
                out_x = out_shape[-2]
                out_y = out_shape[-1]
                arch_details[name] = {
                    'ch_in': ch_in,
                    'ch_out': ch_out,
                    'groups': groups,
                    'k_x': k_x,
                    'k_y': k_y,
                    'out_x': out_x,
                    'out_y': out_y,
                }
                arch_details[name]['x_ch'] = np.arange(1, ch_out+1)
                arch_details[name]['analog_func'] = np.array([
                    analog_cycles(ch_in, ch, k_x, k_y, out_x, out_y)[1]
                    for ch in range(1, ch_out+1)])
                arch_details[name]['analog_latency'] = \
                    arch_details[name]['analog_func'].max()
                if groups == 1:
                    arch_details[name]['digital_func'] = np.array([
                        digital_cycles(ch_in, ch, k_x, k_y, out_x, out_y)[1]
                        for ch in range(1, ch_out+1)])
                else:
                    arch_details[name]['digital_func'] = np.array([
                        digital_cycles(ch, ch, k_x, k_y, out_x, out_y, groups=ch)[1]
                        for ch in range(1, ch_out+1)])
                d_f = arch_details[name]['digital_func']
                a_f = np.flip(arch_details[name]['analog_func'])
                min_da = np.minimum(d_f, a_f)
                arch_details[name]['power_func'] = \
                    power.p_hyb * min_da + power.p_ana * (d_f - min_da) + \
                    power.p_dig * (a_f - min_da)
                arch_details[name]['digital_latency'] = \
                    arch_details[name]['digital_func'].max()
                idx = np.argmin(np.abs(np.flip(
                    arch_details[name]['analog_func']) - arch_details[name]['digital_func']))
                arch_details[name]['min_latency'] = min(
                    arch_details[name]['digital_func'][idx],
                    arch_details[name]['analog_func'][idx])
                if pretrained_arch is not None:
                    alpha = state_dict[f'{name}.mix_weight.alpha_weight'].detach().cpu().numpy()
                    prec = alpha.argmax(axis=0)
                    ch_d = sum(prec == 0)
                    ch_a = sum(prec == 1)
                    _, nas_d = digital_cycles(ch_in, ch_d, k_x, k_y, out_x, out_y, groups)
                    _, nas_a = analog_cycles(ch_in, ch_a, k_x, k_y, out_x, out_y)
                    nas_max = max(nas_d, nas_a)
                    arch_details[name]['NAS_digital'] = nas_d
                    arch_details[name]['NAS_analog'] = nas_a
                    arch_details[name]['NAS_max'] = nas_max

    if plot_layer_cycles:
        df = pd.DataFrame(arch_details)
        n_layer = len(df.columns)
        # figsize = [1*x for x in plt.rcParams["figure.figsize"]]
        figsize = [n_layer, 2*n_layer]
        fig, axis = plt.subplots(n_layer, figsize=figsize)

        for idx, col in enumerate(df):
            if df[col]['groups'] == 1:
                axis[idx].plot(df[col]['x_ch'], df[col]['analog_func'][::-1],
                               color='#ff595e', label='analog')
                axis[idx].plot(df[col]['x_ch'], df[col]['digital_func'],
                               color='#1982c4', label='digital')
                axis[idx].set_title(col)

        handles, labels = axis[-1].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=2)
        fig.set_tight_layout(True)
        fig.savefig(f'{str(arch)}_cycles.png')
        print(f'Layer-wise cycles profile saved @ {str(arch)}_cycles.png', end='\n')

    if plot_layer_power:
        df = pd.DataFrame(arch_details)
        n_layer = len(df.columns)
        # figsize = [1*x for x in plt.rcParams["figure.figsize"]]
        figsize = [n_layer, 2*n_layer]
        fig, axis = plt.subplots(n_layer, figsize=figsize)

        for idx, col in enumerate(df):
            if df[col]['groups'] == 1:
                axis[idx].plot(df[col]['x_ch'], df[col]['power_func'],
                               color='#1982c4', label='power')
                axis[idx].set_title(col)

        handles, labels = axis[-1].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=2)
        fig.set_tight_layout(True)
        fig.savefig(f'{str(arch)}_power.png')
        print(f'Layer-wise power profile saved @ {str(arch)}_power.png', end='\n')

    if plot_disc_net and pretrained_arch is not None:
        layers = len(arch_details.keys())
        br1 = np.arange(layers)
        br2 = [x + _BAR_WIDTH for x in br1]
        br3 = [x + _BAR_WIDTH for x in br2]
        br4 = [x + _BAR_WIDTH for x in br3]
        br5 = [x + _BAR_WIDTH for x in br4]
        br6 = [x + _BAR_WIDTH for x in br5]
        analog_latency = []
        digital_latency = []
        min_latency = []
        NAS_analog = []
        NAS_digital = []
        NAS_max = []
        for key in arch_details.keys():
            analog_latency.append(arch_details[key]["analog_latency"])
            digital_latency.append(arch_details[key]["digital_latency"])
            min_latency.append(arch_details[key]["min_latency"])
            NAS_analog.append(arch_details[key]["NAS_analog"])
            NAS_digital.append(arch_details[key]["NAS_digital"])
            NAS_max.append(arch_details[key]["NAS_max"])
        fig, ax1 = plt.subplots(figsize=(layers, layers))
        ax1.barh(br1, analog_latency, height=_BAR_WIDTH, edgecolor='k', color="blue",
                 label="latency analog")
        ax1.barh(br2, digital_latency, height=_BAR_WIDTH, edgecolor='k', color="green",
                 label="latency digital")
        ax1.barh(br3, min_latency, height=_BAR_WIDTH, edgecolor='k', color="r",
                 label="minimum latency hybrid")
        ax1.barh(br4, NAS_analog, height=_BAR_WIDTH, edgecolor='k', color="blue",
                 label="NAS analog channels latency", hatch="//")
        ax1.barh(br5, NAS_digital, height=_BAR_WIDTH, edgecolor='k', color="green",
                 label="NAS digital channels latency", hatch="//")
        ax1.barh(br6, NAS_max, height=_BAR_WIDTH, edgecolor='k', color="r",
                 label="NAS latency", hatch="//")
        fig.legend()
        fig.set_tight_layout(True)
        fig.savefig(f'{str(arch)}_{strength}_profile.png')
        print(f'Discovered arch profile saved @ {str(arch)}_{strength}_profile.png', end='\n')

    if save_pkl:
        with open(f'details_{str(arch)}_{strength}.pickle', 'wb') as h:
            pickle.dump(arch_details, h, protocol=pickle.HIGHEST_PROTOCOL)

    return arch_details


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Model')
    parser.add_argument('arch', type=str, help='Seed Architecture name')
    parser.add_argument('--path', type=str, default=None, help='path to discovered network')
    parser.add_argument('--plot-net-graph', action='store_true', default=False,
                        help='Whether to plot net graph')
    parser.add_argument('--plot-layer-cycles', action='store_true', default=False,
                        help='Whether to plot layer cycles detail')
    parser.add_argument('--plot-layer-power', action='store_true', default=False,
                        help='Whether to plot layer power detail')
    parser.add_argument('--plot-disc-net', action='store_true', default=False,
                        help='Whether to plot discovered net detail')
    parser.add_argument('--save-pkl', action='store_true', default=False,
                        help='Architecture name')
    args = parser.parse_args()

    if args.arch not in _ARCH_FUNC:
        raise ValueError(
            f'''{args.arch} is not supported. List of supported models: {_ARCH_FUNC.keys()}''')

    profile_cycles(args.arch,
                   pretrained_arch=args.path,
                   plot_layer_cycles=args.plot_layer_cycles,
                   plot_layer_power=args.plot_layer_power,
                   plot_disc_net=args.plot_disc_net,
                   plot_net_graph=args.plot_net_graph,
                   save_pkl=args.save_pkl)
