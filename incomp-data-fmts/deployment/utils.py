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

from enum import Enum, auto
import torch.nn as nn

__all__ = [
    'add_attributes',
    'IntegerizationMode'
]


def add_attributes(ref_mod: nn.Module, target_mod: nn.Module,
                   exclude_attr: tuple = ()):
    for attr in dir(ref_mod):
        if attr not in dir(target_mod) and not isinstance(getattr(ref_mod, attr), exclude_attr):
            setattr(target_mod, attr, getattr(ref_mod, attr))


class IntegerizationMode(Enum):
    Int = auto()
    FakeInt = auto()
