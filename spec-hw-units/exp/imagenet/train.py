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


import logging
from time import perf_counter
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler  # , CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from pytorch_benchmarks.utils import AverageMeter, accuracy
from odimo.method.nn import ThermometricCombiner


def get_default_optimizer(net: nn.Module) -> optim.Optimizer:
    # Filter parameters that do not require weight decay, namely the biases and
    # BatchNorm weights

    no_decay_layers = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                       ThermometricCombiner)

    def filter_fn(x):
        if isinstance(x, no_decay_layers):
            return x.requires_grad
        elif isinstance(x, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            return x.requires_grad if x.ndim > 1 else False
        else:
            return False

    # Filter
    no_decay_params = filter(filter_fn, net.parameters())
    decay_params = filter(lambda x: not filter_fn(x), net.parameters())

    parameters = [
        {'params': decay_params, 'weight_decay': 4e-5},
        {'params': no_decay_params, 'weight_decay': 0.0},
        # {'params': net.nas_parameters(), 'weight_decay': 0.0}
        ]

    # return optim.SGD(parameters, lr=0.4, momentum=0.9, nesterov=True)
    # return optim.SGD(parameters, lr=0.4)
    return optim.Adam(parameters, lr=1e-4)


def train_one_epoch(
        epoch: int,
        search: bool,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: LRScheduler,
        train: DataLoader,
        val: DataLoader,
        device: torch.device,
        scaler: torch.cuda.amp.GradScaler,
        reg_strength: torch.Tensor = torch.tensor(0.),
        source: Literal['original', 'huggingface'] = 'original',
        ema_model: Optional[nn.Module] = None
        ) -> Dict[str, float]:
    assert source in ['original', 'huggingface'], 'Unknown source'
    model.train()
    avgacc = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    avglosstask = AverageMeter('2.5f')
    avglossreg = AverageMeter('2.5f')
    step = 0

    train.sampler.set_epoch(epoch=epoch)

    t0 = perf_counter()
    for batch in train:
        if source == 'huggingface':
            image, target = batch['pixel_values'], batch['label']
        else:
            image, target = batch[0], batch[1]
        scheduler.step(epoch=epoch, iteration=None)  # None -> iteration++
        image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)

        # Option 1: no AMP
        if search:
            output, reg = model(image, comp_reg_loss=search)
            loss_reg = reg_strength * reg
        else:
            output = model(image, comp_reg_loss=search)
            loss_reg = 0
        loss_task = criterion(output, target)
        # loss_task = 0.
        loss = loss_task + loss_reg
        loss.backward()
        optimizer.step()

        # Option 2: AMP
        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        #     output = model(image)
        #     loss_task = criterion(output, target)
        #     if search:
        #         loss_reg = reg_strength * model.regularization_loss()
        #         loss = loss_task + loss_reg
        #     else:
        #         loss = loss_task
        #         loss_reg = 0
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        optimizer.zero_grad(set_to_none=True)

        if ema_model is not None:
            ema_model.update_parameters(model)
        acc_val = accuracy(output, target, topk=(1,))
        avgacc.update(acc_val[0], image.size(0))
        avgloss.update(loss, image.size(0))
        avglosstask.update(loss_task, image.size(0))
        avglossreg.update(loss_reg, image.size(0))
        if step % 500 == 0:
            logging.info(f'GPU {device}, Epoch: {epoch}, Step: {step}/{len(train)}, '
                         f'Batch/s: {step / (perf_counter() - t0)}, '
                         f'Loss: {avgloss}, Loss Task: {avglosstask}, Loss Reg: {avglossreg}, Acc: {avgacc}')
        step += 1
    logging.info(f'Epoch {epoch}, Time: {perf_counter() - t0}')
    final_metrics = {
            'loss': avgloss.get(),
            'acc': avgacc.get(),
        }
    if val is not None:
        val_metrics = evaluate(search, model, criterion, val, device,
                               reg_strength=reg_strength, source=source)
        val_metrics = {'val_' + k: v for k, v in val_metrics.items()}
        final_metrics.update(val_metrics)
    return final_metrics


def evaluate(
        search: bool,
        model: nn.Module,
        criterion: nn.Module,
        data: DataLoader,
        device: torch.device,
        source: Literal['original', 'huggingface'] = 'original',
        reg_strength: torch.Tensor = torch.tensor(0.),
        ) -> Dict[str, float]:
    assert source in ['original', 'huggingface'], 'Unknown source'
    model.eval()
    avgacc = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    avglosstask = AverageMeter('2.5f')
    avglossreg = AverageMeter('2.5f')
    step = 0
    with torch.no_grad():
        for batch in data:
            if source == 'huggingface':
                image, target = batch['pixel_values'], batch['label']
            else:
                image, target = batch[0], batch[1]
            step += 1
            image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(image)
            loss_task = criterion(output, target)
            if search:
                loss_reg = reg_strength * model.module.get_regularization_loss()
                loss = loss_task + loss_reg
            else:
                loss = loss_task
                loss_reg = 0
            acc_val = accuracy(output, target, topk=(1,))
            avgacc.update(acc_val[0], image.size(0))
            avgloss.update(loss, image.size(0))
            avglosstask.update(loss_task, image.size(0))
            avglossreg.update(loss_reg, image.size(0))
        final_metrics = {
            'loss': avgloss.get(),
            'loss_task': avglosstask.get(),
            'loss_reg': avglossreg.get(),
            'acc': avgacc.get(),
        }
    return final_metrics
