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
import math
from time import perf_counter
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler  # , CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from pytorch_benchmarks.utils import AverageMeter, accuracy


class CosineAnnealingCVNets(LRScheduler):
    def __init__(self, optimizer, warmup_iterations, warmup_init_lr,
                 max_lr, min_lr, max_epochs, last_epoch=-1, verbose=False):
        self.warmup_iterations = warmup_iterations
        self.warmup_init_lr = warmup_init_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.max_epochs = max_epochs

        if self.warmup_iterations > 0:
            self.warmup_step = (
                max_lr - warmup_init_lr
            ) / warmup_iterations

        self.period = max_epochs
        self.last_iter = -1
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_iter < self.warmup_iterations:
            return [self.warmup_init_lr + self.last_iter * self.warmup_step
                    for base_lr in self.base_lrs]
        else:
            return [self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                    1 + math.cos(math.pi * self.last_epoch / self.period))
                    for base_lr in self.base_lrs]

    def step(self, epoch=None, iteration=None):
        if self.last_epoch < 0:
            epoch = 0

        if self.last_iter < 0:
            iteration = 0

        if epoch is None:
            epoch = self.last_epoch + 1

        if iteration is None:
            iteration = self.last_iter + 1

        self.last_epoch = math.floor(epoch)
        self.last_iter = math.floor(iteration)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


def get_default_criterion(label_smoothing: float = 0.1) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def get_default_optimizer(net: nn.Module) -> optim.Optimizer:
    # Filter parameters that do not require weight decay, namely the biases and
    # BatchNorm weights

    def filter_fn(x):
        if isinstance(x, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
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
        {'params': no_decay_params, 'weight_decay': 0.0}
        ]

    return optim.SGD(parameters, lr=0.05, momentum=0.9, nesterov=True)


def get_default_scheduler(opt: optim.Optimizer,
                          warmup_iterations: int = 7500, warmup_init_lr: float = 0.05,
                          max_lr: float = 0.4, min_lr: float = 2e-4,
                          max_epochs: int = 300,
                          verbose=False) -> LRScheduler:
    # scheduler = CosineAnnealingWarmRestarts(opt, T_0=7500, eta_min=2e-4,
    #                                         verbose=verbose)
    scheduler = CosineAnnealingCVNets(opt,
                                      warmup_iterations=warmup_iterations,
                                      warmup_init_lr=warmup_init_lr,
                                      max_lr=max_lr,
                                      min_lr=min_lr,
                                      max_epochs=max_epochs,
                                      verbose=verbose)
    return scheduler


def train_one_epoch(
        epoch: int,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: LRScheduler,
        train: DataLoader,
        val: DataLoader,
        device: torch.device,
        scaler: torch.cuda.amp.GradScaler,
        source: Literal['original', 'huggingface'] = 'original',
        ema_model: Optional[nn.Module] = None
        ) -> Dict[str, float]:
    assert source in ['original', 'huggingface'], 'Unknown source'
    model.train()
    avgacc = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    step = 0

    # DistributedSampler
    train.sampler.set_epoch(epoch=epoch)
    # if VariableBatchSamplerDDP is used uncomment following line
    # train.batch_sampler.set_epoch(epoch=epoch)
    # train.batch_sampler.update_scales(epoch=epoch)

    t0 = perf_counter()
    for batch in train:
        if source == 'huggingface':
            image, target = batch['pixel_values'], batch['label']
        else:
            image, target = batch[0], batch[1]
        scheduler.step(epoch=epoch, iteration=None)  # None -> iteration++
        image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)

        # Option 1: no AMP
        output = model(image)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Option 2: AMP
        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        #     output = model(image)
        #     loss = criterion(output, target)
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        optimizer.zero_grad(set_to_none=True)

        if ema_model is not None:
            ema_model.update_parameters(model)
        acc_val = accuracy(output, target, topk=(1,))
        avgacc.update(acc_val[0], image.size(0))
        avgloss.update(loss, image.size(0))
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
        source: Literal['original', 'huggingface'] = 'original',
        ) -> Dict[str, float]:
    assert source in ['original', 'huggingface'], 'Unknown source'
    model.eval()
    avgacc = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
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
            loss = criterion(output, target)
            acc_val = accuracy(output, target, topk=(1,))
            avgacc.update(acc_val[0], image.size(0))
            avgloss.update(loss, image.size(0))
        final_metrics = {
            'loss': avgloss.get(),
            'acc': avgacc.get(),
        }
    return final_metrics
