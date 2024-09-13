# Author: Matteo Risso <matteo.risso@polito.it>

import torch
import torch.nn.functional as F
from odimo.method.cost import cost_darkside


TEMP = 1_000_000

print(f'{"="*40} ImageNet Cycles {"="*40}')
cycles_imagenet = []
layer_3_conv_imagenet = {'o_x': 56, 'o_y': 56,
                         'c_in': 128, 'c_out': torch.tensor(64), 'groups': 1,
                         'k_x': 3, 'k_y': 3}
layer_3_dw_imagenet = {'o_x': 56, 'o_y': 56,
                       'c_in': 128, 'c_out': torch.tensor(64), 'groups': 64,
                       'k_x': 3, 'k_y': 3}

cycles_imagenet.append(cost_darkside(layer_3_conv_imagenet))
cycles_imagenet.append(cost_darkside(layer_3_dw_imagenet))
print(f'Cycles: {cycles_imagenet}')

t_cycles_imagenet = torch.stack(cycles_imagenet)
# Compute softmax
s_c_imagenet = F.softmax(t_cycles_imagenet / TEMP, dim=0)
print(f'Softmax coeff: {s_c_imagenet}')

t_c_imagenet = torch.dot(s_c_imagenet, t_cycles_imagenet)
print(f'Actual Cycles: {t_c_imagenet}')

print(f'{"="*40} CIFAR10 Cycles {"="*40}')
cycles_cifar = []
layer_3_conv_cifar = {'o_x': 8, 'o_y': 8,
                      'c_in': 128, 'c_out': torch.tensor(64), 'groups': 1,
                      'k_x': 3, 'k_y': 3}
layer_3_dw_cifar = {'o_x': 8, 'o_y': 8,
                    'c_in': 128, 'c_out': torch.tensor(64), 'groups': 64,
                    'k_x': 3, 'k_y': 3}

cycles_cifar.append(cost_darkside(layer_3_conv_cifar))
cycles_cifar.append(cost_darkside(layer_3_dw_cifar))
print(f'Cycles: {cycles_cifar}')

t_cycles_cifar = torch.stack(cycles_cifar)
# Compute softmax
s_c_cifar = F.softmax(t_cycles_cifar / TEMP, dim=0)
print(f'Softmax coeff: {s_c_cifar}')

t_c_cifar = torch.dot(s_c_cifar, t_cycles_cifar)
print(f'Actual Cycles: {t_c_cifar}')
