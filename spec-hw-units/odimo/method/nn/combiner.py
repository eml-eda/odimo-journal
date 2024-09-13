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

from typing import List, cast, Iterator, Tuple, Any, Dict, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from .binarizer import Binarizer
from ..cost import cost_naive, cost_darkside, POWER_DARKSIDE


class ThermometricCombiner(nn.Module):
    """Currently support only two layers.
    Split calculation of the N `out_channels` within the two provided
    `input_layers` in a thermometric fashion.
    I.e., combine the first `alpha` channels produced by the first layer with the
    (N - `alpha`) channels produced by the other.
    `alpha` is a trainable NAS parameter.
    The thermometric behavior is disabled by setting the `thermometric` parameter to False.

    :param input_layers: list of possible alternative layers to be selected
    :type input_layers: List[nn.Module]
    :param out_channels: number of output channels
    :type out_channels: int
    :param thermometric: False to disable the thermometric behavior.
    :type thermometric: bool
    :param init_strategy: '1st': initially 1st layer always selected.
    '2nd': initially 2nd layer always selected. Default to 'half'.
    :type init_strategy: Literal['half', '1st', '2nd']
    :param warmup_strategy: 'std', the same scheme of search is used.
    'coarse', select with 50% probability 1st or 2nd layer.
    'fine', select with 1/`out_channels` probability a thermometric assignement.
    Default to 'std'.
    :type warmup_strategy: Literal['std', 'coarse', 'fine']
    :param binarization_threshold: the binarization threshold for PIT masks, defaults to 0.5
    :type binarization_threshold: float, optional
    """
    def __init__(self, input_layers: nn.ModuleList, out_channels: int,
                 thermometric: bool,
                 init_strategy: Literal['half', '1st', '2nd'] = 'half',
                 warmup_strategy: Literal['std', 'coarse', 'fine'] = 'std',
                 binarization_threshold: float = 0.5):
        super(ThermometricCombiner, self).__init__()
        self.sn_input_layers = [_ for _ in input_layers]
        self.n_layers = len(input_layers)
        self.out_channels = out_channels
        self.thermometric = thermometric
        self._binarization_threshold = binarization_threshold

        # Init
        self._init_strategy = init_strategy
        self.alpha = nn.Parameter(
            torch.ones(self.out_channels, dtype=torch.float32))
        self.init_alpha()

        # Cost
        self._cost = 'naive'
        self.update_cost_fn(cost=self._cost)

        # Combiner behavior and phase (initially 'warmup')
        self._phase = 'warmup'
        self._warmup_strategy = warmup_strategy
        self.update_combiner_behavior(phase=self.phase, strategy=self.warmup_strategy)

        self._norm = self._generate_norm_constants()
        self.register_buffer('ch_eff', torch.tensor(self.out_channels, dtype=torch.float32))
        self.register_buffer('_c_alpha', self._generate_c_matrix())
        self.layers_sizes = []
        self.layers_macs = []

    def forward(self, layers_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Forward function for the ThermometricCombiner.
        It returns a thermometric weighted sum of the outputs of the two
        alternative layers.

        :param layers_outputs: outputs of the two modules
        :type layers_outputs: torch.Tensor
        :return: the output tensor (weighted sum of the two layers output)
        :rtype: torch.Tensor
        """
        bin_theta = self.sample_theta()
        # bin_theta = Binarizer.apply(self.theta, self._binarization_threshold)
        # bin_theta_flip = torch.flip(bin_theta, dims=(0,))
        bin_theta_flip = 1 - bin_theta

        y = []
        y.append(torch.mul(layers_outputs[0], bin_theta.view(1, -1, 1, 1)))
        y.append(torch.mul(layers_outputs[1], bin_theta_flip.view(1, -1, 1, 1)))
        y = torch.stack(y, dim=0).sum(dim=0)

        # save info for regularization
        # norm_theta = torch.mul(self.theta, cast(torch.Tensor, self._norm))
        # self.ch_eff = torch.sum(norm_theta)
        self.ch_eff = torch.sum(bin_theta)

        return y

    def update_cost_fn(self, cost: Literal['naive', 'darkside', 'darkside-power']):
        if cost == 'naive':
            self.cost_fn = cost_naive
            self.use_power = False
        elif cost == 'darkside':
            self.cost_fn = cost_darkside
            self.use_power = False
        elif cost == 'darkside-power':
            self.cost_fn = cost_darkside
            self.use_power = True
        else:
            raise ValueError(f'{cost} is not supported.')

    def update_combiner_behavior(self,
                                 phase: Literal['warmup', 'search'] = 'search',
                                 strategy: Literal['std', 'coarse', 'fine'] = 'std'):
        """The behavior of the combiner changes depending on the:
        - The phase (i.e., 'warmup' or 'search').
        - The strategy (i.e., 'std', 'coarse' or 'fine').

        :param phase: the specific training phase, default to 'search'.
        :type phase: Literal['warmup', 'search']
        :param strategy: the warmup granularity, default to 'std'.
        :type strategy: Literal['std', 'coarse', 'fine']
        """
        if phase == 'warmup':
            # During warmup self.alpha is not trained
            self.alpha.requires_grad = False
            # Update sampling strategy
            if strategy == 'coarse':
                self.sample_theta = self.sample_theta_warmup_coarse
            elif strategy == 'fine':
                self.sample_theta = self.sample_theta_warmup_fine
            elif strategy == 'std':
                self.sample_theta = self.sample_theta_search
            else:
                msg = f'{strategy} is not supported.'
                msg += ' Supported granularity: "std", "coarse" or "fine".'
                raise ValueError(msg)
        elif phase == 'search':
            # Reset alpha
            with torch.no_grad():
                self.init_alpha()
            # During warmup self.alpha is trained
            self.alpha.requires_grad = True
            # Update sampling strategy
            self.sample_theta = self.sample_theta_search
        else:
            msg = f'{phase} is not supported. Supported phase: "warmup" or "search".'
            raise ValueError(msg)

    def init_alpha(self):
        if self.init_strategy == '1st':
            with torch.no_grad():
                self.alpha.data.fill_(1.)
        elif self.init_strategy == '2nd':
            with torch.no_grad():
                self.alpha.data.fill_(0.)
        elif self.init_strategy == 'half':
            # Set half of self.alpha to 0 and other half to 1.
            with torch.no_grad():
                self.alpha[int(self.out_channels / 2):] = 0.
                self.alpha[:int(self.out_channels / 2)] = 1.
        else:
            msg = f'{self.init_strategy} is not supported.'
            msg += 'Supported strategies: "1st", "2nd", "half"'
            raise ValueError(msg)

    def sample_theta_warmup_coarse(self) -> torch.Tensor:
        # Sampling from U[0, 1)
        p = torch.rand(1)
        # Change alpha values accordingly
        self.alpha.fill_(float(p.ge(0.5)))
        # Compute binarized theta
        bin_theta = Binarizer.apply(self.theta, self._binarization_threshold)
        return bin_theta

    def sample_theta_warmup_fine(self) -> torch.Tensor:
        device = self.alpha.device  # TODO: define property for device
        cout = self.out_channels
        # Sampling from {0, cout} with equal probability
        p = torch.randint(0, cout+1, (1,)).item()
        if p == 0:
            bin_theta = torch.zeros(cout, device=device)
        elif p == cout:
            bin_theta = torch.ones(cout, device=device)
        else:
            bin_theta = torch.cat((
                torch.ones(p, device=device),
                torch.zeros(cout-p, device=device)
                ))
        return bin_theta

    def sample_theta_search(self) -> torch.Tensor:
        bin_theta = Binarizer.apply(self.theta, self._binarization_threshold)
        return bin_theta

    def update_input_layers(self, input_layers: nn.Module):
        """Updates the list of input layers after torch.fx tracing, which "explodes" nn.Sequential
        and nn.ModuleList, causing the combiner to wrongly reference to the pre-tracing version.
        """
        il = [cast(nn.Module, input_layers.__getattr__(str(_))) for _ in range(self.n_layers)]
        self.sn_input_layers = il

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
    def theta(self) -> torch.Tensor:
        """The forward function that generates the binary masks from the trainable floating point
        shadow copies

        Implemented similarly to plinio/methods/pit/nn/timestep_masker.py
        if self.thermometric is True (default).
        Unless, it is implemented as plinio/methods/pit/nn/feature_masker.py

        :return: the binary masks
        :rtype: torch.Tensor
        """
        if self.thermometric:
            c_alpha = cast(torch.Tensor, self._c_alpha)
            theta_alpha = torch.matmul(c_alpha, torch.abs(self.alpha))
        else:
            theta_alpha = torch.abs(self.alpha)
        return theta_alpha

    def compute_layers_sizes(self):
        """Computes the effective size of the two layers of the PITSuperNetModule
        and stores the values in a list.
        The effective size is given by the size of each layer weighted by the
        number of channels effectively assigned to the layer.
        """
        # First Layer
        layer_size = 0
        for layer in self.sn_input_layers[0]._modules.values():
            if isinstance(layer, nn.Conv2d):
                layer_size = torch.prod(torch.tensor(layer.weight.shape))
        self.layers_sizes.append(layer_size)
        layer_size = 0
        # Second Layer
        for layer in self.sn_input_layers[1]._modules.values():
            if isinstance(layer, nn.Conv2d):
                layer_size = torch.prod(torch.tensor(layer.weight.shape))
        self.layers_sizes.append(layer_size)

    def compute_layers_macs(self, input_shape: Tuple[int, ...]):
        """Computes the effective MACs of the two layers of the PITSuperNetModule
        and stores the values in a list.
        The effective MACs are given by the MACs of each layer weighted by the
        number of channels effectively assigned to the layer.
        """
        # First Layer
        for layer in self.sn_input_layers[0]._modules.values():
            if isinstance(layer, nn.Conv2d):
                stats = summary(layer, input_shape, verbose=0, mode='eval')
                self.layers_macs.append(stats.total_mult_adds)
        # Second Layer
        # TODO: FIx hard-coding
        for layer in self.sn_input_layers[1]._modules['depthwise']._modules.values():
            if isinstance(layer, nn.Conv2d):
                stats = summary(layer, input_shape, verbose=0, mode='eval')
                self.layers_macs.append(stats.total_mult_adds)

    def register_layers_shapes(self, o_x: int, o_y: int):
        """Store the shapes of the two layers in two dicts.
        """
        # First Layer
        self.layer0_shapes = list()
        for layer in filter(lambda x: isinstance(x, nn.Conv2d),
                            self.sn_input_layers[0].modules()):
            shape_dict = dict()
            shape_dict['o_x'] = o_x
            shape_dict['o_y'] = o_y
            shape_dict['c_in'] = layer.in_channels
            shape_dict['c_out'] = layer.out_channels
            shape_dict['k_x'] = layer.kernel_size[0]
            shape_dict['k_y'] = layer.kernel_size[1]
            shape_dict['groups'] = layer.groups
            self.layer0_shapes.append(shape_dict)
        # Second Layer
        self.layer1_shapes = list()
        for layer in filter(lambda x: isinstance(x, nn.Conv2d),
                            self.sn_input_layers[1].modules()):
            shape_dict = dict()
            shape_dict['o_x'] = o_x
            shape_dict['o_y'] = o_y
            shape_dict['c_in'] = layer.in_channels
            shape_dict['c_out'] = layer.out_channels
            shape_dict['k_x'] = layer.kernel_size[0]
            shape_dict['k_y'] = layer.kernel_size[1]
            shape_dict['groups'] = layer.groups
            self.layer1_shapes.append(shape_dict)

        # Determine latency_fn depending on the len of layer0_shapes and layer1_shapes
        if len(self.layer0_shapes) == 2 and len(self.layer1_shapes) == 1:
            self.latency_fn = self.get_latency_dws_dw
        elif len(self.layer0_shapes) == 1 and len(self.layer1_shapes) == 2:
            self.latency_fn = self.get_latency_conv_dws
        elif len(self.layer0_shapes) == 1 and len(self.layer1_shapes) == 1:
            self.latency_fn = self.get_latency_conv_dw
        else:
            raise ValueError("The number of shapes in layer0_shapes and layer1_shapes must be 1 or 2.")

    def get_size(self) -> torch.Tensor:
        """Method that returns the number of weights for the module
        computed as a weighted sum of the number of weights of each layer.

        :return: number of weights of the module (weighted sum)
        :rtype: torch.Tensor
        """
        # First Layer
        # Re-weight layer_size with the number of effective assigned channels
        total_size = self.layers_sizes[0] * self.ch_eff / self.out_channels
        # Second Layer
        # Re-weight layer_size with the number of effective assigned channels
        total_size = total_size + \
            (self.layers_sizes[1] * (1 - self.ch_eff / self.out_channels))
        return total_size

    def get_macs(self) -> torch.Tensor:
        """Method that computes the number of MAC operations for the module

        :return: the number of MACs
        :rtype: torch.Tensor
        """
        # First Layer
        # Re-weight layer_size with the number of effective assigned channels
        total_macs = self.layers_macs[0] * self.ch_eff / self.out_channels
        # Second Layer
        # Re-weight layer_size with the number of effective assigned channels
        total_macs = total_macs + \
            (self.layers_macs[1] * (1 - self.ch_eff / self.out_channels))
        return total_macs

    def get_latency(self) -> torch.Tensor:
        """Method that computes the overall latency of the module denoted in cycles.
        This naive implementation assumes 1 MAC = 1 cycles for both normal and
        depthwise convolutions.

        :return: the latency
        :rtype: torch.Tensor
        """
        lat = self.latency_fn()
        return lat

    def get_latency_dws_dw(self) -> torch.Tensor:
        self.layer0_shapes[0]['c_out'] = torch.tensor(self.out_channels)
        self.layer0_shapes[1]['c_out'] = self.ch_eff
        self.layer1_shapes[0]['c_out'] = self.out_channels - self.ch_eff

        lat_dw_0 = self.cost_fn(self.layer0_shapes[0])
        lat_pw_0 = self.cost_fn(self.layer0_shapes[1])
        lat_dw_1 = self.cost_fn(self.layer1_shapes[0])
        lat = lat_dw_0 + lat_pw_0 + lat_dw_1

        return lat

    def get_latency_conv_dws(self) -> torch.Tensor:
        self.layer0_shapes[0]['c_out'] = self.ch_eff
        self.layer1_shapes[0]['c_out'] = torch.tensor(self.out_channels)
        self.layer1_shapes[1]['c_out'] = self.out_channels - self.ch_eff

        lat_conv = self.cost_fn(self.layer0_shapes[0])
        lat_dw = self.cost_fn(self.layer1_shapes[0])
        lat_pw = self.cost_fn(self.layer1_shapes[1])
        lat = lat_dw + max(lat_conv - lat_dw, 0) + lat_pw

        return lat

    def get_latency_conv_dw(self) -> torch.Tensor:
        # device = self.alpha.device
        cycles = []
        # First Layer
        # Re-weight layer_size with the number of effective assigned channels
        # cycles.append(
        #     torch.tensor(self.layers_macs[0] * self.ch_eff / self.out_channels,
        #                  dtype=torch.float32, device=device)
        #                  )
        # cycles.append(self.layers_macs[0] * self.ch_eff / self.out_channels)
        self.layer0_shapes[0]['c_out'] = self.ch_eff
        cycles.append(self.cost_fn(self.layer0_shapes[0]))
        # Second Layer
        # Re-weight layer_size with the number of effective assigned channels
        # cycles.append(
        #     torch.tensor(self.layers_macs[1] * (1 - self.ch_eff / self.out_channels),
        #                  dtype=torch.float32, device=device)
        #                  )
        # cycles.append(self.layers_macs[1] * (1 - self.ch_eff / self.out_channels))
        self.layer1_shapes[0]['c_out'] = self.out_channels - self.ch_eff
        cycles.append(self.cost_fn(self.layer1_shapes[0]))

        # Build tensor of cycles
        # NB: torch.tensor() does not preserve gradients!!!
        t_cycles = torch.stack(cycles)
        # Compute softmax
        temp = 1e+7
        # temp = 1
        s_c = F.softmax(t_cycles / temp, dim=0)
        t_c = torch.dot(s_c, t_cycles)

        if self.use_power:
            p_idle = (POWER_DARKSIDE['IDLE'] * (t_c - t_cycles[0]) +
                      POWER_DARKSIDE['IDLE'] * (t_c - t_cycles[1]))
            p_gap8 = POWER_DARKSIDE['GAP8'] * t_cycles[0]
            p_dwe = POWER_DARKSIDE['DWE'] * t_cycles[1]
            return p_idle + p_gap8 + p_dwe

        return t_c

    def get_macs_overhead(self) -> torch.Tensor:
        """Method that computes the total number of MAC operations for the module
        as the overhead of the module

        :return: the number of MACs
        :rtype: torch.Tensor
        """
        return self.layers_macs[0] + self.layers_macs[1]

    def get_real_latency(self) -> torch.Tensor:
        """Method that computes the overall latency of the module denoted in cycles.
        This naive implementation assumes 1 MAC = 1 cycles for both normal and
        depthwise convolutions.

        :return: the latency
        :rtype: torch.Tensor
        """
        lat = self.latency_fn()
        return lat

    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized SN hyperparameters
        TODO: cleanup

        :return: a dictionary containing the optimized layer hyperparameter values
        :rtype: Dict[str, Any]
        """
        with torch.no_grad():
            bin_theta = Binarizer.apply(self.theta, self._binarization_threshold)
            # bin_theta_flip = torch.flip(bin_theta, dims=(0,))
            bin_theta_flip = 1 - bin_theta
            theta = [bin_theta, bin_theta_flip]

        res = {"supernet_branches": {}}
        for i, branch in enumerate(self.sn_input_layers):
            if hasattr(branch, "summary") and callable(branch.summary):
                branch_arch = branch.summary()
            else:
                branch_arch = {}
            branch_arch['type'] = branch.__class__.__name__
            branch_arch['theta'] = sum(theta[i])
            branch_layers = branch._modules
            for layer_name in branch_layers:
                layer = cast(nn.Module, branch_layers[layer_name])
                if hasattr(layer, "summary") and callable(layer.summary):
                    layer_arch = branch_layers[layer_name].summary()
                    layer_arch['type'] = branch_layers[layer_name].__class__.__name__
                    branch_arch[layer_name] = layer_arch
            res["supernet_branches"][f"branch_{i}"] = branch_arch
        return res

    @property
    def train_selection(self) -> bool:
        """True if the choice of layers is being optimized by the Combiner

        :return: True if the choice of layers is being optimized by the Combiner
        :rtype: bool
        """
        return self.alpha.requires_grad

    @train_selection.setter
    def train_selection(self, value: bool):
        """Set to True in order to let the Combiner optimize the choice of layers

        :param value: set to True in order to let the Combine optimize the choice of layers
        :type value: bool
        """
        self.alpha.requires_grad = value

    def named_nas_parameters(
            self, prefix: str = '', recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the architectural parameters of this module, yielding
        both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names, defaults to ''
        :type prefix: str, optional
        :param recurse: kept for uniformity with pytorch API, defaults to False
        :type recurse: bool, optional
        :yield: an iterator over the architectural parameters of all layers of the module
        :rtype: Iterator[Tuple[str, nn.Parameter]]
        """
        prfx = prefix
        prfx += "." if len(prefix) > 0 else ""
        prfx += "alpha"
        yield prfx, self.alpha

    def nas_parameters(self, recurse: bool = False) -> Iterator[nn.Parameter]:
        """Returns an iterator over the architectural parameters of this module

        :param recurse: kept for uniformity with pytorch API, defaults to False
        :type recurse: bool, optional
        :yield: an iterator over the architectural parameters of all layers of the module
        :rtype: Iterator[nn.Parameter]
        """
        for _, param in self.named_nas_parameters(recurse=recurse):
            yield param

    def _generate_c_matrix(self) -> torch.Tensor:
        """Method called at creation time, to generate the C_alpha matrix.

        The C_alpha matrix is used to combine different channel mask elements (alpha_i), as
        described in the PIT journal paper for receptive-field.

        :return: the C_alpha matrix as tensor
        :rtype: torch.Tensor
        """
        c_alpha = torch.triu(torch.ones((self.out_channels, self.out_channels),
                             dtype=torch.float32))
        return c_alpha

    def _generate_norm_constants(self) -> torch.Tensor:
        """Method called at construction time to generate the normalization constants for the
        correct evaluation of the effective number of ch split.

        The details of how these constants are computed are found in the PIT journal paper.

        :return: tensor of normalization constants.
        :rtype: torch.Tensor
        """
        norm = torch.tensor(
            [1.0 / (self.out_channels - i) for i in range(self.out_channels)],
        )
        return norm
