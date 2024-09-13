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
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from pytorch_benchmarks.utils import AverageMeter, accuracy


def train_one_epoch(
        epoch: int,
        search: bool,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        train: DataLoader,
        val: DataLoader,
        device: torch.device,
        warmup_scheduler: LRScheduler,
        WARM_P: int,
        reg_strength: torch.Tensor = torch.tensor(0.),
        ) -> Dict[str, float]:
    model.train()
    avgacc = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    avglosstask = AverageMeter('2.5f')
    avglossreg = AverageMeter('2.5f')
    step = 0

    train.sampler.set_epoch(epoch=epoch)

    t0 = perf_counter()
    for images, target in train:
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)

        # Option 1: no AMP
        if search:
            output, reg = model(images, comp_reg_loss=search)
            loss_reg = reg_strength * reg
        else:
            output = model(images, comp_reg_loss=search)
            loss_reg = 0
        loss_task = criterion(output, target)
        # loss_task = 0.
        loss = loss_task + loss_reg
        loss.backward()
        optimizer.step()

        if epoch <= WARM_P - 1:
            warmup_scheduler.step()

        optimizer.zero_grad(set_to_none=True)

        acc_val = accuracy(output, target, topk=(1,))
        avgacc.update(acc_val[0], images.size(0))
        avgloss.update(loss, images.size(0))
        avglosstask.update(loss_task, images.size(0))
        avglossreg.update(loss_reg, images.size(0))
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
                               reg_strength=reg_strength)
        val_metrics = {'val_' + k: v for k, v in val_metrics.items()}
        final_metrics.update(val_metrics)
    return final_metrics


def evaluate(
        search: bool,
        model: nn.Module,
        criterion: nn.Module,
        data: DataLoader,
        device: torch.device,
        reg_strength: torch.Tensor = torch.tensor(0.),
        ) -> Dict[str, float]:
    model.eval()
    avgacc = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    avglosstask = AverageMeter('2.5f')
    avglossreg = AverageMeter('2.5f')
    step = 0
    with torch.no_grad():
        for images, target in data:
            step += 1
            images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(images)
            loss_task = criterion(output, target)
            if search:
                loss_reg = reg_strength * model.module.get_regularization_loss()
                loss = loss_task + loss_reg
            else:
                loss = loss_task
                loss_reg = 0
            acc_val = accuracy(output, target, topk=(1,))
            avgacc.update(acc_val[0], images.size(0))
            avgloss.update(loss, images.size(0))
            avglosstask.update(loss_task, images.size(0))
            avglossreg.update(loss_reg, images.size(0))
        final_metrics = {
            'loss': avgloss.get(),
            'loss_task': avglosstask.get(),
            'loss_reg': avglossreg.get(),
            'acc': avgacc.get(),
        }
    return final_metrics
