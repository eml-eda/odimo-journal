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

import copy
from math import ceil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from . import utils
from . import quant_module as qm
from . import quant_module_pow2 as qm2
from . import hw_models as hw


__all__ = [
    "quanttemponet_fp",
    "quanttemponet_fp_foldbn",
    "quanttemponet_w8a7_pow2_foldbn",
    "quanttemponet_w2a7_pow2_foldbn",
    "quanttemponet_w2a7_true_pow2_foldbn",
    "quanttemponet_pow2_diana_full",
]


class TempConvBlock(nn.Module):
    """
    Temporal Convolutional Block composed of one temporal convolutional layer.
    The block is composed of :
    - Conv1d layer
    - ReLU layer
    - BatchNorm1d layer
    :param ch_in: Number of input channels
    :param ch_out: Number of output channels
    :param k_size: Kernel size
    :param dil: Amount of dilation
    :param pad: Amount of padding
    """

    def __init__(
        self,
        conv_func,
        ch_in,
        ch_out,
        k_size,
        dil,
        pad,
        abits,
        wbits,
        bias,
        bn,
        **kwargs,
    ):
        self.bn = bn
        self.use_bias = bias
        self.use_bn = bn
        self.fp = conv_func is qm.FpConv2d
        super().__init__()
        self.tcn = conv_func(
            ch_in,
            ch_out,
            wbits,
            abits,
            kernel_size=k_size,
            dilation=dil,
            padding=pad,
            bias=self.use_bias,
            groups=1,
            **kwargs,
        )

        if self.use_bn:
            self.bn = nn.BatchNorm2d(num_features=ch_out)

    def forward(self, x):
        x = self.tcn(x)
        if self.use_bn:
            x = self.bn(x)
        if self.fp:
            x = F.relu(x)
        return x


class ConvBlock(nn.Module):
    """
    Convolutional Block composed of:
    - Conv1d layer
    - AvgPool1d layer
    - ReLU layer
    - BatchNorm1d layer
    :param ch_in: Number of input channels
    :param ch_out: Number of output channels
    :param k_size: Kernel size
    :param s: Amount of stride
    :param pad: Amount of padding
    """

    def __init__(
        self,
        conv_func,
        ch_in,
        ch_out,
        k_size,
        s,
        pad,
        dilation,
        abits,
        wbits,
        bias,
        bn,
        **kwargs,
    ):
        self.bn = bn
        self.use_bias = bias
        self.use_bn = bn
        self.fp = conv_func is qm.FpConv2d
        super(ConvBlock, self).__init__()
        self.conv = conv_func(
            ch_in,
            ch_out,
            wbits,
            abits,
            kernel_size=k_size,
            stride=s,
            dilation=dilation,
            padding=pad,
            bias=self.use_bias,
            groups=1,
            **kwargs,
        )
        if self.fp:
            self.pool = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        else:
            self.pool = qm2.QuantAvgPool2d(abits, kernel_size=(2, 1), stride=(2, 1))
        if self.use_bn:
            self.bn = nn.BatchNorm2d(ch_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        if self.use_bn:
            x = self.bn(x)
        if self.fp:
            x = F.relu(x)
        return x


class Regressor(nn.Module):
    """
    Regressor block composed of:
    - Linear layer
    - ReLU layer
    - BatchNorm1d layer
    :param ft_in: Number of input channels
    :param ft_out: Number of output channels
    """

    def __init__(
        self,
        conv_func,
        ft_in,
        ft_out,
        abits,
        wbits,
        bias,
        bn,
        qtz_fc,
        **kwargs,
    ):
        self.bn = bn
        self.use_bias = bias
        self.use_bn = bn
        self.fp = conv_func is qm.FpConv2d
        super().__init__()
        self.fc = conv_func(
            ft_in,
            ft_out,
            wbits,
            abits,
            kernel_size=kwargs.pop("kernel_size", (1, 1)),
            stride=(1, 1),
            bias=self.use_bias,
            fc=qtz_fc,
            groups=1,
            **kwargs,
        )
        if self.use_bn:
            self.bn = nn.BatchNorm2d(num_features=ft_out)

    def forward(self, x):
        x = self.fc(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class TEMPONet(nn.Module):
    """
    TEMPONet architecture:
    Three repeated instances of TemporalConvBlock and ConvBlock organized as follows:
    - TemporalConvBlock
    - ConvBlock
    Two instances of Regressor followed by a final Linear layer with a single neuron.
    """

    def __init__(
        self,
        conv_func,
        hw_model,
        archws,
        archas,
        qtz_fc=None,
        bn=True,
        target="latency",
        **kwargs,
    ):
        print("archas: {}".format(archas))
        print("archws: {}".format(archws))

        # Parameters
        self.input_shape = (4, 256)  # default for PPG-DALIA dataset
        self.dil = [2, 2, 1, 4, 4, 8, 8]
        self.rf = [5, 5, 5, 9, 9, 17, 17]
        self.ch = [32, 32, 64, 64, 64, 128, 128, 128, 128, 256, 128]

        self.conv_func = conv_func
        self.fp = conv_func is qm.FpConv2d
        self.hw_model = hw_model
        self.search_types = ["fixed", "mixed", "multi"]
        if qtz_fc in self.search_types:
            self.qtz_fc = qtz_fc
        else:
            self.qtz_fc = False
        self.bn = bn
        self.use_bias = not bn
        self.target = target
        if target == "latency":
            self.fetch_arch_info = self._fetch_arch_latency
        # elif target == "power":
        #     self.power = hw.DianaPower()
        #     self.fetch_arch_info = self._fetch_arch_power
        else:
            raise ValueError('Use "latency" or "power" as target.')
        super().__init__()

        # 1st instance of two TempConvBlocks and ConvBlock
        k_tcb00 = ceil(self.rf[0] / self.dil[0])
        self.tcb00 = TempConvBlock(
            conv_func=self.conv_func,
            ch_in=4,
            ch_out=self.ch[0],
            k_size=(k_tcb00, 1),
            dil=(self.dil[0], 1),
            pad=(((k_tcb00 - 1) * self.dil[0] + 1) // 2, 0),
            abits=archas[0],
            wbits=archws[0],
            bias=self.use_bias,
            bn=self.bn,
            max_inp_val=kwargs.pop("max_inp_val", 1.0),
            signed=True,
            **kwargs,
        )
        k_tcb01 = ceil(self.rf[1] / self.dil[1])
        self.tcb01 = TempConvBlock(
            conv_func=self.conv_func,
            ch_in=self.ch[0],
            ch_out=self.ch[1],
            k_size=(k_tcb01, 1),
            dil=(self.dil[1], 1),
            pad=(((k_tcb01 - 1) * self.dil[1] + 1) // 2, 0),
            abits=archas[1],
            wbits=archws[1],
            bias=self.use_bias,
            bn=self.bn,
            **kwargs,
        )
        k_cb0 = ceil(self.rf[2] / self.dil[2])
        self.cb0 = ConvBlock(
            conv_func=self.conv_func,
            ch_in=self.ch[1],
            ch_out=self.ch[2],
            k_size=(k_cb0, 1),
            s=(1, 1),
            pad=(((k_cb0 - 1) * self.dil[2] + 1) // 2, 0),
            dilation=(self.dil[2], 1),
            abits=archas[2],
            wbits=archws[2],
            bias=self.use_bias,
            bn=self.bn,
            **kwargs,
        )

        # 2nd instance of two TempConvBlocks and ConvBlock
        k_tcb10 = ceil(self.rf[3] / self.dil[3])
        self.tcb10 = TempConvBlock(
            conv_func=self.conv_func,
            ch_in=self.ch[2],
            ch_out=self.ch[3],
            k_size=(k_tcb10, 1),
            dil=(self.dil[3], 1),
            pad="same",
            abits=archas[3],
            wbits=archws[3],
            bias=self.use_bias,
            bn=self.bn,
            **kwargs,
        )
        k_tcb11 = ceil(self.rf[4] / self.dil[4])
        self.tcb11 = TempConvBlock(
            conv_func=self.conv_func,
            ch_in=self.ch[3],
            ch_out=self.ch[4],
            k_size=(k_tcb11, 1),
            dil=(self.dil[4], 1),
            pad="same",
            abits=archas[4],
            wbits=archws[4],
            bias=self.use_bias,
            bn=self.bn,
            **kwargs,
        )
        self.cb1 = ConvBlock(
            conv_func=self.conv_func,
            ch_in=self.ch[4],
            ch_out=self.ch[5],
            k_size=(5, 1),
            s=(2, 1),
            pad=(2, 0),
            dilation=(1, 1),
            abits=archas[5],
            wbits=archws[5],
            bias=self.use_bias,
            bn=self.bn,
            **kwargs,
        )

        # 3td instance of TempConvBlock and ConvBlock
        k_tcb20 = ceil(self.rf[5] / self.dil[5])
        self.tcb20 = TempConvBlock(
            conv_func=self.conv_func,
            ch_in=self.ch[5],
            ch_out=self.ch[6],
            k_size=(k_tcb20, 1),
            dil=(self.dil[5], 1),
            pad="same",
            abits=archas[6],
            wbits=archws[6],
            bias=self.use_bias,
            bn=self.bn,
            **kwargs,
        )
        k_tcb21 = ceil(self.rf[6] / self.dil[6])
        self.tcb21 = TempConvBlock(
            conv_func=self.conv_func,
            ch_in=self.ch[6],
            ch_out=self.ch[7],
            k_size=(k_tcb21, 1),
            dil=(self.dil[6], 1),
            pad="same",
            abits=archas[7],
            wbits=archws[7],
            bias=self.use_bias,
            bn=self.bn,
            **kwargs,
        )
        self.cb2 = ConvBlock(
            conv_func=self.conv_func,
            ch_in=self.ch[7],
            ch_out=self.ch[8],
            k_size=(5, 1),
            s=(4, 1),
            pad=(4, 0),
            dilation=(1, 1),
            abits=archas[8],
            wbits=archws[8],
            bias=self.use_bias,
            bn=self.bn,
            **kwargs,
        )

        # 1st instance of regressor
        self.regr0 = Regressor(
            conv_func=self.conv_func,
            ft_in=self.ch[8],
            ft_out=self.ch[9],
            abits=archas[9],
            wbits=archws[9],
            bias=self.use_bias,
            bn=self.bn,
            qtz_fc=self.qtz_fc,
            kernel_size=(4, 1),
            **kwargs,
        )

        # 2nd instance of regressor
        self.regr1 = Regressor(
            conv_func=self.conv_func,
            ft_in=self.ch[9],
            ft_out=self.ch[10],
            abits=archas[10],
            wbits=archws[10],
            bias=self.use_bias,
            bn=self.bn,
            qtz_fc=self.qtz_fc,
            **kwargs,
        )

        # Output layer
        self.out_neuron = conv_func(
            self.ch[10],
            1,
            kernel_size=(1, 1),
            stride=(1, 1),
            abits=archas[11],
            wbits=archws[11],
            bias=True,
            fc=self.qtz_fc,
            groups=1,
            **kwargs,
        )

    def forward(self, input):
        # 1st instance of two TempConvBlocks and ConvBlock
        x = self.tcb00(input.unsqueeze(3))
        x = self.tcb01(x)
        x = self.cb0(x)
        # 2nd instance of two TempConvBlocks and ConvBlock
        x = self.tcb10(x)
        x = self.tcb11(x)
        x = self.cb1(x)
        # 3td instance of TempConvBlock and ConvBlock
        x = self.tcb20(x)
        x = self.tcb21(x)
        x = self.cb2(x)
        # Flatten
        # x = x.flatten(1)
        # 1st instance of regressor
        x = self.regr0(x)
        # 2nd instance of regressor
        x = self.regr1(x)
        # Output layer
        x = self.out_neuron(x)[:, :, 0, 0]
        return x

    def _fetch_arch_latency(self):
        sum_cycles, sum_bita, sum_bitw = 0, 0, 0
        layer_idx = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                size_product = m.size_product.item()
                memory_size = m.memory_size.item()
                wbit = m.wbits[0]
                abit = m.abits[0]
                sum_bitw += size_product * wbit

                cycles_analog, cycles_digital = 0, 0
                for idx, wb in enumerate(m.wbits):
                    if len(m.wbits) > 1:
                        ch_out = m.mix_weight.alpha_weight[idx].sum()
                    else:
                        ch_out = torch.tensor(m.ch_out)
                    # Define dict whit shape infos used to model accelerators perf
                    conv_shape = {
                        "ch_in": m.ch_in,
                        "ch_out": ch_out,
                        "groups": m.mix_weight.conv.groups,
                        "k_x": m.k_x,
                        "k_y": m.k_y,
                        "out_x": m.out_x,
                        "out_y": m.out_y,
                    }
                    if wb == 2:
                        cycles_analog = self.hw_model("analog", **conv_shape)
                    else:
                        cycles_digital = self.hw_model("digital", **conv_shape)
                if m.mix_weight.conv.groups == 1:
                    cycles = max(cycles_analog, cycles_digital)
                else:
                    cycles = cycles_digital

                bita = memory_size * abit
                bitw = m.param_size * wbit
                sum_cycles += cycles
                sum_bita += bita
                sum_bitw += bitw
                layer_idx += 1
        return sum_cycles, sum_bita, sum_bitw


def quanttemponet_fp(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 12, [[8]] * 12
    model = TEMPONet(
        qm.FpConv2d,
        hw.diana(analog_speedup=5.0),
        archws,
        archas,
        qtz_fc="multi",
        **kwargs,
    )
    return model


def quanttemponet_fp_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[8]] * 12, [[8]] * 12
    model = TEMPONet(qm.FpConv2d, None, archws, archas, qtz_fc="multi", **kwargs)
    fp_state_dict = torch.load(arch_cfg_path)["state_dict"]
    model.load_state_dict(fp_state_dict)

    model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(model)
    folded_model.train()  # Put folded model in train mode

    return folded_model


def quanttemponet_w8a7_pow2_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 12, [[8]] * 12
    s_up = kwargs.pop("analog_speedup", 5.0)
    fp_model = TEMPONet(
        qm.FpConv2d,
        hw.diana(analog_speedup=s_up),
        archws,
        archas,
        qtz_fc="multi",
        **kwargs,
    )
    q_model = TEMPONet(
        qm2.QuantMultiPrecActivConv2d,
        hw.diana(analog_speedup=s_up),
        archws,
        archas,
        qtz_fc="multi",
        bn=False,
        **kwargs,
    )

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)["state_dict"]
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quanttemponet_w2a7_pow2_foldbn(arch_cfg_path, target="latency", **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 12, [[2]] * 12
    # Set first and last layer weights precision to 8bit
    archws[0] = [8]
    archws[-1] = [8]
    s_up = kwargs.pop("analog_speedup", 5.0)
    fp_model = TEMPONet(
        qm.FpConv2d,
        hw.diana(analog_speedup=s_up),
        archws,
        archas,
        qtz_fc="multi",
        **kwargs,
    )
    q_model = TEMPONet(
        qm2.QuantMultiPrecActivConv2d,
        hw.diana(analog_speedup=s_up),
        archws,
        archas,
        qtz_fc="multi",
        bn=False,
        target=target,
        **kwargs,
    )

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)["state_dict"]
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quanttemponet_w2a7_true_pow2_foldbn(arch_cfg_path, target="latency", **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 12, [[2]] * 12
    archws[-1] = [8]
    s_up = kwargs.pop("analog_speedup", 5.0)
    fp_model = TEMPONet(
        qm.FpConv2d,
        hw.diana(analog_speedup=s_up),
        archws,
        archas,
        qtz_fc="multi",
        **kwargs,
    )
    q_model = TEMPONet(
        qm2.QuantMultiPrecActivConv2d,
        hw.diana(analog_speedup=s_up),
        archws,
        archas,
        qtz_fc="multi",
        bn=False,
        target=target,
        **kwargs,
    )

    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)["state_dict"]
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quanttemponet_pow2_diana_full(arch_cfg_path, **kwargs):
    wbits, abits = [8, 2], [7]

    # ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch["alpha_activ"]]
    archws = [wbits for w_ch in best_arch["alpha_weight"]]
    # if len(archws) == 21:
    #     # Case of fixed-precision on last fc layer
    #     archws.append(8)
    # assert len(archas) == 22  # 10 insead of 8 because conv1 and fc activations are also quantized
    # assert len(archws) == 22  # 10 instead of 8 because conv1 and fc weights are also quantized
    ##

    kwargs.pop("analog_speedup", 5.0)
    model = TEMPONet(
        qm2.QuantMultiPrecActivConv2d,
        hw.diana(),
        archws,
        archas,
        qtz_fc="multi",
        bn=False,
        **kwargs,
    )
    utils.init_scale_param(model)

    return _quanttemponet_diana(arch_cfg_path, model, **kwargs)


def _load_arch_multi_prec(arch_path):
    checkpoint = torch.load(arch_path, map_location="cpu")
    # state_dict = checkpoint["model_state_dict"]
    state_dict = checkpoint["state_dict"]
    best_arch, worst_arch = {}, {}
    best_arch["alpha_activ"], worst_arch["alpha_activ"] = [], []
    best_arch["alpha_weight"], worst_arch["alpha_weight"] = [], []
    for name, params in state_dict.items():
        name = name.split(".")[-1]
        if name == "alpha_activ":
            alpha = params.cpu().numpy()
            best_arch[name].append(alpha.argmax())
            worst_arch[name].append(alpha.argmin())
        elif name == "alpha_weight":
            alpha = params.cpu().numpy()
            best_arch[name].append(alpha.argmax(axis=0))
            worst_arch[name].append(alpha.argmin(axis=0))

    return best_arch, worst_arch


def _load_alpha_state_dict(arch_path):
    checkpoint = torch.load(arch_path)
    state_dict = checkpoint["state_dict"]
    alpha_state_dict = dict()
    for name, params in state_dict.items():
        full_name = name
        name = name.split(".")[-1]
        if name == "alpha_activ" or name == "alpha_weight":
            alpha_state_dict[full_name] = params

    return alpha_state_dict


def _quanttemponet_diana(arch_cfg_path, model, **kwargs):
    if kwargs.get("fine_tune", True):
        # Load all weights
        state_dict = torch.load(arch_cfg_path)["state_dict"]
        model.load_state_dict(state_dict)
    else:
        # Load only alphas weights
        alpha_state_dict = _load_alpha_state_dict(arch_cfg_path)
        model.load_state_dict(alpha_state_dict, strict=False)
    return model
