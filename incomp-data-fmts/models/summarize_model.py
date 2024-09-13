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
import json
from pathlib import Path

# import numpy as np
import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

import models.quant_module_pow2 as qm
# from models.model_diana import analog_cycles, digital_cycles
from models.quant_resnet import quantres18_diana_full, quantres20_diana_full, quantres18_w8a7_foldbn, \
    quantres18_w8a7_pow2_foldbn_c100, quantres18_minlat_max8_pow2_foldbn_c100, quantres18_pow2_diana_full_c100, \
    quantres18_minlat_max8_foldbn, quantres18_diana_full
from models.quant_mobilenetv1 import quantmobilenetv1_diana_full

from deployment.quantization import QuantizationTracer

_ARCH_FUNC = {
    'mobilenetv1': quantmobilenetv1_diana_full,
    'resnet18': quantres18_diana_full,
    'resnet18_c100': quantres18_pow2_diana_full_c100,
    'resnet18_w8a7': quantres18_w8a7_foldbn,
    'resnet18_w2a7': quantres18_minlat_max8_foldbn,
    'resnet18_imn': quantres18_diana_full,
    'resnet18_w8a7_c100': quantres18_w8a7_pow2_foldbn_c100,
    'resnet18_w2a7_c100': quantres18_minlat_max8_pow2_foldbn_c100,
    'resnet20': quantres20_diana_full,
}

_INP_SHAPE = {
    'mobilenetv1': (1, 3, 96, 96),
    'resnet18': (1, 3, 64, 64),
    'resnet18_c100': (1, 3, 32, 32),
    'resnet18_w8a7': (1, 3, 224, 224),
    'resnet18_w2a7': (1, 3, 224, 224),
    'resnet18_imn': (1, 3, 224, 224),
    'resnet18_w8a7_c100': (1, 3, 32, 32),
    'resnet18_w2a7_c100': (1, 3, 32, 32),
    'resnet20': (1, 3, 32, 32),
}


def main(arch,
         input_shape='default',
         pretrained=None,
         disc_net=None,
         output_name='output.json'):

    if arch == 'resnet18':
        model = _ARCH_FUNC[arch](pretrained, std_head=True)
    else:
        model = _ARCH_FUNC[arch](pretrained)
    if input_shape == 'default':
        dummy_input = torch.randn(_INP_SHAPE[arch])
    else:
        img_size = tuple(input_shape, input_shape)
        inp_shape = (1, 3) + img_size
        dummy_input = torch.randn(inp_shape)

    if disc_net is not None:
        path = Path(disc_net)
        if not path.exists():
            raise ValueError(f'{path} does not exists!')
        state_dict = torch.load(path)['model_state_dict']
        model.load_state_dict(state_dict)

    # gm = fx.symbolic_trace(model)
    # modules = dict(gm.named_modules())
    target_layers = (qm.QuantMultiPrecActivConv2d,
                     qm.QuantAvgPool2d,
                     qm.QuantAdd)
    tracer = QuantizationTracer(target_layers)
    graph = tracer.trace(model.eval())
    name = model.__class__.__name__
    gm = fx.GraphModule(tracer.root, graph, name)
    modules = dict(gm.named_modules())
    arch_details = dict()

    ShapeProp(gm).propagate(dummy_input)

    # Build dict with layers' details
    for n in gm.graph.nodes:
        m = modules.get(n.target)
        if isinstance(m, target_layers):
            layer_name = n.target
            layer_details = dict()
            # Find previous layer(s)
            prev_layer = list()
            for inp in n.all_input_nodes:
                prev_layer += _prev_layer(inp, modules, target_layers)
            if prev_layer == [None]:  # Input Layer
                layer_details['in_shape'] = list(dummy_input.shape)[2:]
            else:
                layer_details['in_shape'] = arch_details[prev_layer[0]]['out_shape']
            layer_details['out_shape'] = list(n.meta['tensor_meta'].shape)[2:]
            layer_details['prev_layer'] = prev_layer
            if isinstance(m, qm.QuantMultiPrecActivConv2d):
                layer_details['op'] = 'CONV'
                conv = m.mix_weight.conv
                layer_details['ch_in'] = conv.in_channels
                layer_details['ch_out'] = conv.out_channels
                layer_details['groups'] = conv.groups
                layer_details['k_x'] = conv.kernel_size[0]
                layer_details['k_y'] = conv.kernel_size[1]
                layer_details['stride'] = list(conv.stride)
                layer_details['padding'] = list(conv.padding)
                if conv.groups == 1:
                    alpha = m.mix_weight.alpha_weight.detach().cpu().numpy()
                    if alpha.shape[0] > 1:
                        prec = alpha.argmax(axis=0)
                        ch_d = sum(prec == 0)
                        ch_a = sum(prec == 1)
                    else:  # shape == 1
                        prec = m.mix_weight.bits[0]
                        ch_d = conv.out_channels if prec == 8 else 0
                        ch_a = conv.out_channels if prec == 2 else 0
                else:  # depthwise
                    ch_d = conv.out_channels
                    ch_a = 0
                layer_details['digital_ch'] = int(ch_d)
                layer_details['analog_ch'] = int(ch_a)
            elif isinstance(m, qm.QuantAvgPool2d):
                layer_details['op'] = 'AVG_POOL'
                pool = m.pool
                layer_details['k'] = pool.kernel_size
                layer_details['stride'] = pool.stride
                layer_details['padding'] = pool.padding
            elif isinstance(m, qm.QuantAdd):
                layer_details['op'] = 'ADD'
            arch_details[layer_name] = layer_details

    with open(output_name, 'w') as f:
        json.dump(arch_details, f, indent=4)

    return arch_details


def _prev_layer(n, modules, target):
    if n.op == 'placeholder':
        return [None]
    elif n.op == 'call_module':
        m = modules.get(n.target)
        if isinstance(m, target):
            return [n.target]
    for inp in n.all_input_nodes:
        return _prev_layer(inp, modules, target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize Model')
    parser.add_argument('arch', type=str, help='Seed Architecture name')
    parser.add_argument('--pretrained', type=str, help='path to pretrained network')
    parser.add_argument('--path', type=str, help='path to discovered network')
    parser.add_argument('--name', type=str, default='output.json', help='output file name')
    args = parser.parse_args()

    if args.arch not in _ARCH_FUNC:
        raise ValueError(
            f'{args.arch} is not supported. List of supported models: {_ARCH_FUNC.keys()}')

    main(args.arch,
         pretrained=args.pretrained,
         disc_net=args.path,
         output_name=args.name)
