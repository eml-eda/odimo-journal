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

import torch
from odimo.method import ThermometricNet
from exp.common import models

FREQ = 260e+6

device = 'cpu'
rnd_inp = torch.rand((1, 3, 32, 32), device=device)

# ## Width-Multiplier = 0.25x ## #
model_fn = models.__dict__['mbv1_search_8']
model = model_fn((3, 32, 32), 10)
therm_model = ThermometricNet(model,
                              input_shape=(3, 32, 32),
                              init_strategy='1st').to(device)

therm_model.set_cost(cost='naive')
therm_model(rnd_inp)
print(f'mbv1_conv_8 - naive: {therm_model.get_real_latency()}')

therm_model.set_cost(cost='darkside')
therm_model(rnd_inp)
print(f'mbv1_conv_8 - darkside: {therm_model.get_real_latency()}')

therm_model.set_cost(cost='darkside-power')
therm_model(rnd_inp)
print(f'mbv1_conv_8 - darkside-power: {therm_model.get_real_latency() / FREQ}')

with torch.no_grad():
    for combiner in therm_model._target_combiners:
        _, layer = combiner
        layer.alpha.data.fill_(0.)

therm_model.set_cost(cost='naive')
therm_model(rnd_inp)
print(f'mbv1_dw_8 - naive: {therm_model.get_real_latency()}')

therm_model.set_cost(cost='darkside')
therm_model(rnd_inp)
print(f'mbv1_dw_8 - darkside: {therm_model.get_real_latency()}')

therm_model.set_cost(cost='darkside-power')
therm_model(rnd_inp)
print(f'mbv1_dw_8 - darkside-power: {therm_model.get_real_latency() / FREQ}')

# ## Width-Multiplier = 0.5x ## #
model_fn = models.__dict__['mbv1_search_16']
model = model_fn((3, 32, 32), 10)
therm_model = ThermometricNet(model,
                              input_shape=(3, 32, 32),
                              init_strategy='1st').to(device)

therm_model.set_cost(cost='naive')
therm_model(rnd_inp)
print(f'mbv1_conv_16 - naive: {therm_model.get_real_latency()}')

therm_model.set_cost(cost='darkside')
therm_model(rnd_inp)
print(f'mbv1_conv_16 - darkside: {therm_model.get_real_latency()}')

therm_model.set_cost(cost='darkside-power')
therm_model(rnd_inp)
print(f'mbv1_conv_16 - darkside-power: {therm_model.get_real_latency() / FREQ}')

with torch.no_grad():
    for combiner in therm_model._target_combiners:
        _, layer = combiner
        layer.alpha.data.fill_(0.)

therm_model.set_cost(cost='naive')
therm_model(rnd_inp)
print(f'mbv1_dw_16 - naive: {therm_model.get_real_latency()}')

therm_model.set_cost(cost='darkside')
therm_model(rnd_inp)
print(f'mbv1_dw_16 - darkside: {therm_model.get_real_latency()}')

therm_model.set_cost(cost='darkside-power')
therm_model(rnd_inp)
print(f'mbv1_dw_16 - darkside-power: {therm_model.get_real_latency() / FREQ}')

# ## Width-Multiplier = 1x ## #
model_fn = models.__dict__['mbv1_search_32']
model = model_fn((3, 32, 32), 10)
therm_model = ThermometricNet(model,
                              input_shape=(3, 32, 32),
                              init_strategy='1st').to(device)

therm_model.set_cost(cost='naive')
therm_model(rnd_inp)
print(f'mbv1_conv_32 - naive: {therm_model.get_real_latency()}')

therm_model.set_cost(cost='darkside')
therm_model(rnd_inp)
print(f'mbv1_conv_32 - darkside: {therm_model.get_real_latency()}')

therm_model.set_cost(cost='darkside-power')
therm_model(rnd_inp)
print(f'mbv1_conv_32 - darkside-power: {therm_model.get_real_latency() / FREQ}')

with torch.no_grad():
    for combiner in therm_model._target_combiners:
        _, layer = combiner
        layer.alpha.data.fill_(0.)

therm_model.set_cost(cost='naive')
therm_model(rnd_inp)
print(f'mbv1_dw_32 - naive: {therm_model.get_real_latency()}')

therm_model.set_cost(cost='darkside')
therm_model(rnd_inp)
print(f'mbv1_dw_32 - darkside: {therm_model.get_real_latency()}')

therm_model.set_cost(cost='darkside-power')
therm_model(rnd_inp)
print(f'mbv1_dw_32 - darkside-power: {therm_model.get_real_latency() / FREQ}')
