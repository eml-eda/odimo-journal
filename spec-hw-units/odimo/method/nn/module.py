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

from typing import Iterable
import torch
import torch.nn as nn
from .combiner import ThermometricCombiner


class ThermometricModule(nn.Module):
    """A nn.Module containing two layer alternatives.
    One of these layers will be selected by the PITSuperNet NAS tool for the current layer.
    This module is for the most part just a placeholder and its logic is contained inside
    the `sn_combiner` of which instance is defined in the constructor.

    :param input_layers: iterable of possible alternative layers to be selected
    :type input_layers: Iterable[nn.Module]
    :param out_channels: the number of output channel of each layer in `input_layers`
    :type out_channels: int
    :param thermometric: False to disable the thermometric behavior. Default to True.
    :type thermometric: bool
    """
    def __init__(self,
                 input_layers: Iterable[nn.Module],
                 out_channels: int,
                 thermometric: bool = True):
        if len(list(input_layers)) > 2:
            msg = 'Currently only selection among two alternatives is supported'
            raise ValueError(msg)
        super(ThermometricModule, self).__init__()
        self.sn_input_layers = nn.ModuleList(list(input_layers))
        self.sn_combiner = ThermometricCombiner(self.sn_input_layers,
                                                out_channels, thermometric)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward function for the ThermometricModule that returns a weighted
        sum of all the outputs of the different input layers.
        It computes all possible layer outputs and passes the list to the sn_combiner
        which computes the weighted sum.

        :param input: the input tensor
        :type input: torch.Tensor
        :return: the output tensor (weighted sum of all layers output)
        :rtype: torch.Tensor
        """
        layers_outputs = [layer(input) for layer in self.sn_input_layers]
        return self.sn_combiner(layers_outputs)

    def __getitem__(self, pos: int) -> nn.Module:
        """Get the layer at position pos in the list of all the possible
        layers for the ThermometricModule

        :param pos: position of the required module in the list input_layers
        :type pos: int
        :return: module at postion pos in the list input_layers
        :rtype: nn.Module
        """
        return self.sn_input_layers[pos]
