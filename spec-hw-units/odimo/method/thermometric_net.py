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

from typing import Tuple, Any, Iterator, Dict, cast, Literal
import torch
import torch.nn as nn
from plinio.methods.dnas_base import DNAS
from .nn import ThermometricCombiner
from .graph import convert


class ThermometricNet(DNAS):
    """
    A class that wraps a nn.Module with the functionality of a Thermometric NAS tool.
    Basically it learns a split between the two choices defined by means of a
    ThermometricModule within the input `model`.

    :param model: the inner nn.Module instance optimized by the NAS
    :type model: nn.Module
    :param input_shape: the shape of an input tensor, without batch size, required for symbolic
    tracing
    :type input_shape: Tuple[int, ...]
    :param regularizer: a string defining the type of cost regularizer, defaults to 'latency'
    :type regularizer: Optional[str], optional
    :param cost: a string defining the type of cost function to be used, defaults to 'naive'
    :type cost: Literal['naive', 'darkside']
    :param init_strategy: '1st': initially 1st layer always selected.
    '2nd': initially 2nd layer always selected. Default to 'half'.
    :type init_strategy: Literal['half', '1st', '2nd']
    :param warmup_strategy: 'std', the same scheme of search is used.
    'coarse', select with 50% probability 1st or 2nd layer.
    'fine', select with 1/`out_channels` probability a thermometric assignement.
    Default to 'std'.
    :type warmup_strategy: Literal['std', 'coarse', 'fine']
    """
    def __init__(
            self,
            model: nn.Module,
            input_shape: Tuple[int, ...],
            regularizer: str = 'latency',
            cost: Literal['naive', 'darkside', 'darkside-power'] = 'naive',
            init_strategy: Literal['half', '1st', '2nd'] = 'half',
            warmup_strategy: Literal['std', 'coarse', 'fine'] = 'std'
            ):
        super(ThermometricNet, self).__init__(regularizer)

        self._input_shape = input_shape
        self._regularizer = regularizer
        self._cost = cost
        self.seed, self._target_combiners = convert(
            model,
            self._input_shape
        )
        # Init
        self._init_strategy = init_strategy
        self.set_init_strategy(strategy=self.init_strategy)
        # Set cost
        self.set_cost(cost=self._cost)
        # Set phase
        self._phase = 'warmup'
        self._warmup_strategy = warmup_strategy
        self.change_train_phase(phase=self.phase, strategy=self.warmup_strategy)

    def forward(self, *args: Any, comp_reg_loss: bool = False) -> torch.Tensor:
        """Forward function for the DNAS model. Simply invokes the inner model's forward

        :return: the output tensor
        :rtype: torch.Tensor
        """
        if comp_reg_loss:
            return self.seed(*args), self.get_regularization_loss()
        else:
            return self.seed.forward(*args)

    def supported_regularizers(self) -> Tuple[str, ...]:
        """Returns a list of names of supported regularizers

        :return: a tuple of strings with the name of supported regularizers
        :rtype: Tuple[str, ...]
        """
        return ('size', 'macs', 'latency')

    def freeze_alpha(self):
        """Freeze the alpha coefficients disabling the gradient computation.
        Useful for fine-tuning the model without changing its architecture
        """
        for combiner in self._target_combiners:
            _, layer = combiner
            layer.alpha.requires_grad = False

    def get_size(self) -> torch.Tensor:
        """Computes the total number of parameters of all NAS-able modules

        :return: the total number of parameters
        :rtype: torch.Tensor
        """
        size = torch.tensor(0, dtype=torch.float32)
        for _, module in self._target_combiners:
            size = size + module.get_size()

        return size

    def get_macs(self) -> torch.Tensor:
        """Computes the total number of MACs in all NAS-able modules

        :return: the total number of MACs
        :rtype: torch.Tensor
        """
        macs = torch.tensor(0, dtype=torch.float32)
        for _, module in self._target_combiners:
            macs = macs + module.get_macs()

        return macs

    def get_latency(self) -> torch.Tensor:
        """Computes the total latency in all NAS-able modules

        :return: the total latency
        :rtype: torch.Tensor
        """
        lat = torch.tensor(0, dtype=torch.float32)
        for _, module in self._target_combiners:
            lat = lat + module.get_latency()

        return lat

    def get_macs_overhead(self) -> torch.Tensor:
        """Computes the overhead of MACs in all NAS-able modules

        :return: the total number of MACs
        :rtype: torch.Tensor
        """
        macs = torch.tensor(0, dtype=torch.float32)
        for _, module in self._target_combiners:
            macs = macs + module.get_macs_overhead()

        return macs

    def get_real_latency(self) -> torch.Tensor:
        """Computes the real latency of net

        :return: the total latency
        :rtype: torch.Tensor
        """
        lat = torch.tensor(0, dtype=torch.float32)
        for _, module in self._target_combiners:
            lat = lat + module.get_real_latency()

        return lat

    def set_init_strategy(self, strategy: Literal['1st', '2nd', 'half']):
        for _, module in self._target_combiners:
            module = cast(ThermometricCombiner, module)
            module.init_strategy = strategy
            module.init_alpha()

    def set_cost(self, cost: Literal['naive', 'darkside', 'darkside-power']):
        for _, module in self._target_combiners:
            module = cast(ThermometricCombiner, module)
            module.update_cost_fn(cost)

    def change_train_phase(self, phase: Literal['warmup', 'search'],
                           strategy: Literal['std', 'coarse', 'fine']):
        for _, module in self._target_combiners:
            module = cast(ThermometricCombiner, module)
            module.phase = phase
            module.warmup_strategy = strategy
            module.update_combiner_behavior(phase=phase, strategy=strategy)

    @property
    def init_strategy(self) -> Literal['1st', '2nd', 'half']:
        return self._init_strategy

    @init_strategy.setter
    def init_strategy(self, val: Literal['1st', '2nd', 'half']):
        self._init_strategy = val

    @property
    def phase(self) -> Literal['warmup', 'search']:
        return self._phase

    @phase.setter
    def phase(self, val: Literal['warmup', 'search']):
        self._phase = val

    @property
    def warmup_strategy(self) -> Literal['coarse', 'fine']:
        return self._warmup_strategy

    @warmup_strategy.setter
    def warmup_strategy(self, val: Literal['coarse', 'fine']):
        self._warmup_strategy = val

    @property
    def regularizer(self) -> str:
        """Returns the regularizer type

        :raises ValueError: for unsupported conversion types
        :return: the string identifying the regularizer type
        :rtype: str
        """
        return self._regularizer

    @regularizer.setter
    def regularizer(self, value: str):
        if value == 'size':
            self.get_regularization_loss = self.get_size
        elif value == 'macs':
            self.get_regularization_loss = self.get_macs
        elif value == 'latency':
            self.get_regularization_loss = self.get_latency
        else:
            raise ValueError(f"Invalid regularizer {value}")
        self._regularizer = value

    def arch_export(self) -> nn.Module:
        """Export the architecture found by the NAS as a 'nn.Module'
        It replaces each PITSuperNetModule found in the model with a single layer.

        :return: the architecture found by the NAS
        :rtype: nn.Module
        """
        model = self.seed
        model, _ = convert(model, self._input_shape, 'export')
        return model

    def arch_summary(self) -> Dict[str, Dict[str, Any]]:
        """Generates a dictionary representation of the architecture found by the NAS.
        Only optimized layers are reported

        :return: a dictionary representation of the architecture found by the NAS
        :rtype: Dict[str, Dict[str, Any]]
        """
        arch = {}
        for name, layer in self._target_combiners:
            layer = cast(ThermometricCombiner, layer)
            arch[name] = layer.summary()
            arch[name]['type'] = layer.__class__.__name__
        return arch

    def named_nas_parameters(
            self, prefix: str = '', recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the architectural parameters of the NAS, yielding
        both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names.
        :type prefix: str
        :param recurse: kept for uniformity with pytorch API
        :type recurse: bool
        :return: an iterator over the architectural parameters of the NAS
        :rtype: Iterator[nn.Parameter]
        """
        for mod_name, module in self._target_combiners:
            prfx = prefix
            prfx += "." if len(prefix) > 0 else ""
            prfx += mod_name
            prfx += "." if len(prfx) > 0 else ""
            for name, param in module.named_nas_parameters():
                prfx = prfx + name
                yield prfx, param

    def named_net_parameters(
            self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the inner network parameters, EXCEPT the NAS architectural
        parameters, yielding both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names.
        :type prefix: str
        :param recurse: kept for uniformity with pytorch API, not actually used
        :type recurse: bool
        :return: an iterator over the inner network parameters
        :rtype: Iterator[nn.Parameter]
        """
        exclude = set(_[0] for _ in self.named_nas_parameters())

        for name, param in self.seed.named_parameters():
            if name not in exclude:
                yield name, param

    def __str__(self):
        """Prints the architecture found by the NAS to screen

        :return: a str representation of the current architecture
        :rtype: str
        """
        arch = self.arch_summary()
        return str(arch)
