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
from exp.common import models

import pytorch_benchmarks.image_classification as icl

device = 'cpu'

# Get the Data and Criterion
datasets = icl.get_data(data_dir='icl_data')
dataloaders = icl.build_dataloaders(datasets)
train_dl, val_dl, test_dl = dataloaders
criterion = icl.get_default_criterion()

# Depthwise #
# Build model
model_fn_dw = models.__dict__['mbv1_dw_8']
model_dw = model_fn_dw((3, 32, 32), 10)
# Load pretrained checkpoint
ckp_dw = torch.load('mbv1_dw_8warmup.ckp')
# ckp_dw = torch.load('warmup/warmup.ckp')
model_dw.load_state_dict(ckp_dw['model_state_dict'])
# Eval
test_metrics_dw = icl.evaluate(model_dw, criterion, test_dl, device)
print("Test Set Accuracy DepthWise:", test_metrics_dw['acc'])

# Conv #
# Build model
model_fn_c = models.__dict__['mbv1_conv_8']
model_c = model_fn_c((3, 32, 32), 10)
# Load pretrained checkpoint
ckp_c = torch.load('mbv1_conv_8warmup.ckp')
model_c.load_state_dict(ckp_c['model_state_dict'])
# Eval
test_metrics_c = icl.evaluate(model_c, criterion, test_dl, device)
print("Test Set Accuracy Conv:", test_metrics_c['acc'])
