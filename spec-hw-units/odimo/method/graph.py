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

from typing import Tuple, List, cast, Optional
import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

from plinio.graph.annotation import add_node_properties, add_features_calculator, \
        associate_input_features
from plinio.graph.inspection import is_layer, get_graph_inputs, \
        all_output_nodes
from plinio.graph.features_calculation import SoftMaxFeaturesCalculator

from .nn import ThermometricCombiner, ThermometricModule


class ThermometricNetTracer(fx.Tracer):
    def __init__(self) -> None:
        super().__init__()  # type: ignore

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, ThermometricCombiner):
            return True
        else:
            return m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)


def convert(model: nn.Module, input_shape: Tuple[int, ...],
            conversion_type: str = 'import',
            ) -> Tuple[nn.Module, List]:
    """Converts a nn.Module, to/from "NAS-able" Thermometric format

    :param model: the input nn.Module
    :type model: nn.Module
    :param input_shape: the shape of an input tensor, without batch size, required for symbolic
    tracing
    :type input_shape: Tuple[int, ...]
    :param conversion_type: a string specifying the type of conversion. Supported types:
    ('import', 'export')
    :type conversion_type: str
    :raises ValueError: for unsupported conversion types
    :return: the converted model, and the list of target layers for the NAS (only for imports)
    :rtype: Tuple[nn.Module, List]
    """

    if conversion_type not in ('import', 'export'):
        raise ValueError("Unsupported conversion type {}".format(conversion_type))

    tracer = ThermometricNetTracer()
    graph = tracer.trace(model.eval())
    name = model.__class__.__name__
    mod = fx.GraphModule(tracer.root, graph, name)
    batch_example = torch.stack([torch.rand(input_shape)] * 1, 0)
    device = next(model.parameters()).device
    ShapeProp(mod).propagate(batch_example.to(device))
    # add_node_properties(mod)
    # set_combiner_properties(mod, add=['shared_input_features', 'features_propagating'])
    if conversion_type in ('import',):
        # TODO: Do I need features_calculator and other stuff????
        # add_features_calculator(mod, [pit_graph.pit_features_calc, combiner_features_calc])
        # set_combiner_properties(mod, add=['features_defining'], remove=['features_propagating'])
        # associate_input_features(mod)
        # lastly, import SuperNet selection layers
        sn_combiners = import_sn_combiners(mod)
    else:
        export_graph(mod)
        sn_combiners = []

    mod.graph.lint()
    mod.recompile()
    return mod, sn_combiners


def import_sn_combiners(mod: fx.GraphModule) -> List[Tuple[str, ThermometricCombiner]]:
    """Finds and prepares "Combiner" layers used to select SuperNet branches during the search phase

    :param mod: a torch.fx.GraphModule with tensor shapes annotations.
    :type mod: fx.GraphModule
    :return: the list of "Combiner" layers that will be optimized by the NAS
    :rtype: List[Tuple[str, ThermometricCombiner]]
    """
    target_layers = []
    for n in mod.graph.nodes:
        if is_layer(n, mod, (ThermometricCombiner,)):
            # parent_name = n.target.removesuffix('.sn_combiner')
            parent_name = n.target.replace('.sn_combiner', '')
            sub_mod = cast(ThermometricCombiner, mod.get_submodule(str(n.target)))
            parent_mod = cast(ThermometricModule, mod.get_submodule(parent_name))
            # TODO: fix this mess
            prev = n.all_input_nodes[0]
            while '.sn_input_layers' in prev.target:
                prev = prev.all_input_nodes[0]
            input_shape = prev.meta['tensor_meta'].shape
            sub_mod.compute_layers_sizes()
            sub_mod.compute_layers_macs(input_shape)
            o_x = n.meta['tensor_meta'].shape[-1]
            o_y = n.meta['tensor_meta'].shape[-2]
            sub_mod.register_layers_shapes(o_x, o_y)
            sub_mod.update_input_layers(parent_mod.sn_input_layers)
            sub_mod.train_selection = True
            target_layers.append((str(n.target), sub_mod))
    return target_layers


def export_graph(mod: fx.GraphModule):
    """Exports the graph of the final NN, selecting the appropriate SuperNet branches.

    :param mod: a torch.fx.GraphModule of a SuperNet
    :type mod: fx.GraphModule
    """
    for n in mod.graph.nodes:
        if 'sn_combiner' in str(n.target):
            sub_mod = cast(ThermometricCombiner, mod.get_submodule(n.target))
            best_idx = sub_mod.best_layer_index()
            best_branch_name = 'sn_input_layers.' + str(best_idx)
            to_erase = []
            for ni in n.all_input_nodes:
                if best_branch_name in str(ni.target):
                    n.replace_all_uses_with(ni)
                else:
                    to_erase.append(ni)
            n.args = ()
            mod.graph.erase_node(n)
            for ni in to_erase:
                ni.args = ()
                mod.graph.erase_node(ni)
    mod.graph.eliminate_dead_code()
    mod.delete_all_unused_submodules()


def set_combiner_properties(
        mod: fx.GraphModule,
        add: List[str] = [],
        remove: List[str] = []):
    """Searches for the combiner nodes in the graph and sets their properties

    :param mod: module
    :type mod: fx.GraphModule
    """
    g = mod.graph
    queue = get_graph_inputs(g)
    visited = []
    while queue:
        n = queue.pop(0)
        if n in visited:
            continue

        if is_layer(n, mod, (ThermometricCombiner,)):
            for p in add:
                n.meta[p] = True
            for p in remove:
                n.meta[p] = False

        for succ in all_output_nodes(n):
            queue.append(succ)
        visited.append(n)


def combiner_features_calc(n: fx.Node, mod: fx.GraphModule) -> Optional[SoftMaxFeaturesCalculator]:
    """Sets the feature calculator for a PITSuperNetCombiner node

    :param n: node
    :type n: fx.Node
    :param mod: the parent module
    :type mod: fx.GraphModule
    :return: optional feature calculator object for the combiner node
    :rtype: SoftMaxFeaturesCalculator
    """
    if is_layer(n, mod, (ThermometricCombiner,)):
        sub_mod = mod.get_submodule(str(n.target))
        prev_features = [_.meta['features_calculator'] for _ in n.all_input_nodes]
        return SoftMaxFeaturesCalculator(sub_mod, 'alpha', prev_features)
    else:
        return None
