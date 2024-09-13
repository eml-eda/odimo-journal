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

from abc import abstractmethod
from typing import Any, Type, Tuple

import torch
import torch.fx as fx
import torch.nn as nn

import deployment.utils as utils

import models.quant_module_pow2 as qm

__all__ = [
    'ObserverBase',
    'PerChannelMaxObserver',
    'insert_observers',
    'remove_observers',
]


class ObserverBase(nn.Module):
    """Base observer Module.
    Any observer implementation should derive from this class.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args: Any) -> Any:
        raise NotImplementedError


# Code inspired by https://tinyurl.com/torc-qtz
class PerChannelMaxObserver(ObserverBase):
    """Observer module for tracing the per channel maximum input value.
    The forward method performs tracing and returns exactly its input.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer('max_val', torch.tensor([]))

    def forward(self, x_orig):
        x = x_orig.detach()  # avoid keeping autograd tape
        max_val = self.max_val
        # assume format [batch, ch, x, y]
        # transpose batch and ch and then flatten maintaining ch dimension
        x = torch.flatten(x.transpose(0, 1), start_dim=1)
        if max_val.numel() == 0:  # initially self.max_val is empty
            max_val = torch.amax(x, dim=1)
        else:
            max_val_cur = torch.amax(x, dim=1)
            max_val = torch.max(max_val_cur, max_val)
        self.max_val.resize_(max_val.shape)
        self.max_val.copy_(max_val)

        return x_orig


class ObserverTracer(fx.Tracer):
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
        elif isinstance(m, qm.QuantPaCTActiv):
            return True
        else:
            return m.__module__.startswith('torch.nn') and \
                not isinstance(m, torch.nn.Sequential)


def insert_observers(
    model: nn.Module,
    target_layers: Tuple[Type[nn.Module], ...],
    observer: ObserverBase = PerChannelMaxObserver
) -> nn.Module:
    """
    Attaches an `observer` to the output of every `target_layers` found
    in `model`.

    :param model: nn.Module where observers shoud be inserted
    :type model: nn.Module
    :param target_layers: set of nn.Module where an observer should be inserted
    :type target_layers: Iterable[Type[nn.Module]]
    :param observer: observer module to be used,
    default to PerChannelMaxObserver
    :type observer: ObserverBase, optional
    :return: a `model` copy with `observer`
    :rtype: nn.Module
    """
    # model_device = next(model.parameters()).device
    tracer = ObserverTracer(target_layers)
    graph = tracer.trace(model.eval())
    name = model.__class__.__name__
    mod = fx.GraphModule(tracer.root, graph, name)
    modules = dict(mod.named_modules())
    for n in mod.graph.nodes:
        if n.target in modules.keys():
            if isinstance(modules[n.target], target_layers):
                if isinstance(modules[n.target], qm.QuantAvgPool2d):
                    continue  # TODO: dirty asf to be fixed!!!!
                if isinstance(modules[n.target], qm.QuantAdd):
                    continue  # TODO: dirty asf to be fixed!!!!
                if isinstance(modules[n.target], qm.QuantPaCTActiv):
                    continue  # TODO: dirty asf to be fixed!!!!
                new_obs_name = f'{n.target}_observer'
                new_obs = observer()
                mod.add_submodule(new_obs_name, new_obs)
                with mod.graph.inserting_after(n):
                    mod.graph.create_node('call_module', new_obs_name, (n,))
    mod.graph.lint()
    mod.recompile()
    # Re-add removed custom attributes contained in model to mod
    utils.add_attributes(model, mod)
    return mod


def remove_observers(
    model: nn.Module,
    target_layers: Tuple[Type[nn.Module], ...],
    observer: ObserverBase = PerChannelMaxObserver
) -> nn.Module:
    """
    Attaches an `observer` to the output of every `target_layers` found
    in `model`.

    :param model: nn.Module where observers shoud be inserted
    :type model: nn.Module
    :param target_layers: set of nn.Module where an observer should be inserted
    :type target_layers: Iterable[Type[nn.Module]]
    :param observer: observer module to be used,
    default to PerChannelMaxObserver
    :type observer: ObserverBase, optional
    :return: a `model` copy with `observer`
    :rtype: nn.Module
    """
    # model_device = next(model.parameters()).device
    tracer = ObserverTracer(target_layers+(observer,))
    graph = tracer.trace(model.eval())
    name = model.__class__.__name__
    mod = fx.GraphModule(tracer.root, graph, name)
    modules = dict(mod.named_modules())
    for n in mod.graph.nodes:
        if n.target in modules.keys():
            if isinstance(modules[n.target], observer):
                mod.delete_submodule(n.target)
                # with mod.graph.inserting_after(n):
                #     new_node = mod.graph.call_function(torch.identity, n.args, n.kwargs)
                #     n.replace_all_uses_with(new_node)
                mod.graph.erase_node(n)
    mod.graph.lint()
    mod.recompile()
    # Re-add removed custom attributes contained in model to mod
    exclude_attr = (observer)
    utils.add_attributes(model, mod, exclude_attr)
    return mod
