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

device = 'cpu'
rnd_inp = torch.rand((1, 3, 224, 224), device=device)

# ## Width-Multiplier = 1x ## #
model_fn = models.__dict__['mbv1_search_32']
model = model_fn((3, 224, 224), 1000)
therm_model = ThermometricNet(model,
                              input_shape=(3, 224, 224),
                              init_strategy='1st').to(device)

therm_model.set_cost(cost='naive')
therm_model(rnd_inp)
print(f'mbv1_conv_32 - naive: {therm_model.get_real_latency()}')

therm_model.set_cost(cost='darkside')
therm_model(rnd_inp)
print(f'mbv1_conv_32 - darkside: {therm_model.get_real_latency()}')

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

'''
dws - darkside
cost_pw = 0.

layer_3 = {'o_x': 56, 'o_y': 56, 'c_in': 128, 'c_out': torch.tensor(128), 'k_x': 1, 'k_y': 1, 'groups': 1}
cost_pw += cost_darkside(layer_3).item()

layer5 = {'o_x': 28, 'o_y': 28, 'c_in': 256, 'c_out': torch.tensor(256), 'k_x': 1, 'k_y': 1, 'groups': 1}
cost_pw += cost_darkside(layer5).item()

layer7 = {'o_x': 14, 'o_y': 14, 'c_in': 512, 'c_out': torch.tensor(512), 'k_x': 1, 'k_y': 1, 'groups': 1}
cost_pw += (cost_darkside(layer7).item()) * 5

layer13 = {'o_x': 7, 'o_y': 7, 'c_in': 1024, 'c_out': torch.tensor(1024), 'k_x': 1, 'k_y': 1, 'groups': 1}
cost_pw += cost_darkside(layer13).item()

cost_pw -> 26203576
'''
