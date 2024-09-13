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

import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    'analog_cycles',
    'digital_cycles',
]

F = 260000000  # Hz


def _floor(ch, N):
    return np.floor((ch + N - 1) / N)


def _ox_unroll_base(ch_in, ch_out, k_x, k_y):
    ox_unroll_base = 1
    channel_input_unroll = 64 if ch_in < 64 else ch_in
    for ox_unroll in [1, 2, 4, 8]:
        if (ch_out * ox_unroll <= 512) and \
                (channel_input_unroll * k_y * (k_x + ox_unroll - 1) <= 1152):
            ox_unroll_base = ox_unroll
    return ox_unroll_base


def digital_cycles(ch_in, ch_out, k_x, k_y, out_x, out_y, groups=1):
    cycles = _floor(ch_out / groups, 16) * ch_in * _floor(out_x, 16) * out_y * k_x * k_y
    gate = float(ch_out >= 1.)
    cycles_load_store = out_x * out_y * (ch_out + ch_in) / 8
    MACs = ch_in * ch_out * out_x * out_y * k_x * k_y
    if gate != 0.:
        MAC_cycles = MACs / (cycles + cycles_load_store)
    else:
        MAC_cycles = 0.
    return MAC_cycles, (cycles + gate * cycles_load_store)


def analog_cycles(ch_in, ch_out, k_x, k_y, out_x, out_y,):
    ox_unroll_base = _ox_unroll_base(ch_in, ch_out, k_x, k_y)
    cycles_computation = _floor(ch_out, 512) * _floor(ch_in, 128) * out_x * out_y / ox_unroll_base
    gate = float(ch_out >= 1.)
    # cycles_weights = gate * 4 * 2 * 1152
    cycles_weights = gate * 4 * 2 * ch_in * k_x * k_y
    MACs = ch_in * ch_out * out_x * out_y * k_x * k_y
    if gate != 0.:
        MAC_cycles = MACs / ((cycles_computation * 70 / (1000000000 / F) + cycles_weights))
    else:
        MAC_cycles = 0.
    return MAC_cycles, (cycles_computation * 70 / (1000000000 / F) + cycles_weights)


if __name__ == '__main__':
    analog = []
    digital = []
    analog_cyc = []
    digital_cyc = []
    ox_unroll = []
    ch_in = 512  # 64
    ch_max = 1000  # 64
    out_x = 1  # 16
    out_y = 1  # 16
    k_x = 1  # 3
    k_y = 1  # 3
    for ch in np.arange(1, ch_max):
        MAC_cycles_digital, cycles_digital = digital_cycles(ch_in, ch, k_x, k_y, out_x, out_y)
        MAC_cycles_analog, cycles_analog = analog_cycles(ch_in, ch, k_x, k_y, out_x, out_y)
        analog.append(MAC_cycles_analog)
        digital_cyc.append(cycles_digital)
        analog_cyc.append(cycles_analog)
        digital.append(MAC_cycles_digital)
        ox_unroll.append(_ox_unroll_base(ch_in, ch, k_x, k_y))
    plt.plot(np.arange(1, ch_max), analog, label="analog")
    plt.plot(np.arange(1, ch_max), digital, label="digital")
    plt.legend()
    plt.savefig("MAC_cycles.png")
    plt.figure()
    plt.plot(np.arange(1, ch_max), ox_unroll)
    # plt.savefig("ox_unroll.png")

    plt.figure()
    plt.plot(np.arange(1, ch_max), analog_cyc, label="analog")
    plt.plot(np.arange(1, ch_max), digital_cyc, label="digital")
    plt.legend()
    plt.savefig("cycles.png")
