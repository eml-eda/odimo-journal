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
from torch.optim.lr_scheduler import MultiStepLR, LRScheduler
from torch.utils.data import DataLoader

from pytorch_benchmarks.utils import AverageMeter, accuracy


class WarmUpLR(torch.optim.lr_scheduler._LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def get_default_criterion() -> nn.Module:
    return nn.CrossEntropyLoss()


def get_default_optimizer(net: nn.Module, lr, momentum, weight_decay) -> optim.Optimizer:
    return optim.SGD(net.parameters(), lr=lr,
                     momentum=momentum, weight_decay=weight_decay)


def get_default_scheduler(opt: optim.Optimizer) -> LRScheduler:
    scheduler = MultiStepLR(opt, milestones=[60, 120, 160], gamma=0.2)
    return scheduler


def train_one_epoch(
        epoch: int,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        train: DataLoader,
        val: DataLoader,
        device: torch.device,
        warmup_scheduler: LRScheduler,
        WARM_P: int,
        ) -> Dict[str, float]:
    model.train()
    avgacc = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    step = 0

    # DistributedSampler
    train.sampler.set_epoch(epoch=epoch)

    t0 = perf_counter()
    for images, target in train:
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if epoch <= WARM_P - 1:
            warmup_scheduler.step()

        acc_val = accuracy(output, target, topk=(1,))
        avgacc.update(acc_val[0], images.size(0))
        avgloss.update(loss, images.size(0))
        if step % 500 == 0:
            logging.info(f'GPU {device}, Epoch: {epoch}, Step: {step}/{len(train)}, '
                         f'Batch/s: {step / (perf_counter() - t0)}, Loss: {avgloss}, Acc: {avgacc}')
        step += 1
    logging.info(f'Epoch {epoch}, Time: {perf_counter() - t0}')
    final_metrics = {
            'loss': avgloss.get(),
            'acc': avgacc.get(),
        }
    if val is not None:
        val_metrics = evaluate(model, criterion, val, device)
        val_metrics = {'val_' + k: v for k, v in val_metrics.items()}
        final_metrics.update(val_metrics)
    return final_metrics


def evaluate(
        model: nn.Module,
        criterion: nn.Module,
        data: DataLoader,
        device: torch.device,
        ) -> Dict[str, float]:
    model.eval()
    avgacc = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    step = 0
    with torch.no_grad():
        for images, target in data:
            step += 1
            images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            acc_val = accuracy(output, target, topk=(1,))
            avgacc.update(acc_val[0], images.size(0))
            avgloss.update(loss, images.size(0))
        final_metrics = {
            'loss': avgloss.get(),
            'acc': avgacc.get(),
        }
    return final_metrics
