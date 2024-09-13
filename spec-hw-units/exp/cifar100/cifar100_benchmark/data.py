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

import os
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms

from .sampler import VariableBatchSamplerDDP


# Define worker init function
def _worker_init_fn(seed):
    def fn(worker_id):
        np.random.seed(seed)
        random.seed(seed)
    return fn


def get_data(data_dir=None,
             val_split=0.2,
             seed=None,
             ) -> Tuple[Dataset, ...]:
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), 'c100_data')

    # Ref: https://github.com/weiaicunzai/pytorch-cifar100/blob/11d8418f415b261e4ae3cb1ffe20d06ec95b98e4/utils.py#L166
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    test_to_tensor = transforms.Compose([
        transforms.ToTensor()
    ])
    ds_train_val = torchvision.datasets.CIFAR100(root=data_dir, train=True,
                                                 download=True, transform=transform)
    ds_test = torchvision.datasets.CIFAR100(root=data_dir, train=False,
                                            download=True, transform=test_to_tensor)

    # Maybe fix seed of RNG
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    if val_split != 0.0:
        val_len = int(val_split * len(ds_train_val))
        train_len = len(ds_train_val) - val_len
        ds_train, ds_val = random_split(ds_train_val, [train_len, val_len],
                                        generator=generator)
    else:
        ds_train, ds_val = ds_train_val, None

    return ds_train, ds_val, ds_test


def build_dataloaders(datasets: Tuple[Dataset, ...],
                      train_batch_size=128,
                      val_batch_size=100,
                      num_workers=4,
                      seed=None,
                      sampler_fn=None,
                      ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_set, val_set, test_set = datasets

    # Maybe fix seed of RNG
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    # Maybe define worker init fn
    if seed is not None:
        worker_init_fn = _worker_init_fn(seed)
    else:
        worker_init_fn = None

    # Turn off shuffling if specific sampler is used
    shuffle = sampler_fn is None

    _collate_fn = None

    # Train
    if sampler_fn is DistributedSampler:
        if seed is None:
            sampler = sampler_fn(train_set)  # Default seed=0, don't put None to avoid errors
        else:
            sampler = sampler_fn(train_set, seed=seed)
        worker_init_fn = None
        batch_sampler = None
    elif sampler_fn is VariableBatchSamplerDDP:
        batch_sampler = sampler_fn(train_set, batch_size=train_batch_size, is_training=True)
        worker_init_fn = None
        sampler = None
        train_batch_size = 1
        shuffle = False
    elif sampler_fn is not None:
        sampler = sampler_fn
    else:
        sampler = None
    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator,
        collate_fn=_collate_fn,
        sampler=sampler,
        batch_sampler=batch_sampler,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Val
    if val_set is not None:
        if sampler_fn is DistributedSampler:
            sampler = sampler_fn(val_set, seed=seed)
            worker_init_fn = None
            batch_sampler = None
        elif sampler_fn is VariableBatchSamplerDDP:
            batch_sampler = sampler_fn(val_set, batch_size=val_batch_size, is_training=False)
            worker_init_fn = None
            sampler = None
            val_batch_size = 1
            shuffle = False
        elif sampler_fn is not None:
            sampler = sampler_fn
        else:
            sampler = None
        val_loader = DataLoader(
            val_set,
            batch_size=val_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            generator=generator,
            collate_fn=_collate_fn,
            sampler=sampler,
            batch_sampler=batch_sampler,
            persistent_workers=True,
        )
    else:
        val_loader = None

    # Test
    if sampler_fn is DistributedSampler:
        if seed is None:
            sampler = sampler_fn(test_set)  # Default seed=0, don't put None to avoid errors
        else:
            sampler = sampler_fn(test_set, seed=seed)
        worker_init_fn = None
        batch_sampler = None
    elif sampler_fn is VariableBatchSamplerDDP:
        batch_sampler = sampler_fn(test_set, batch_size=val_batch_size, is_training=False)
        worker_init_fn = None
        sampler = None
        val_batch_size = 1
        shuffle = False
    elif sampler_fn is not None:
        sampler = sampler_fn
    else:
        sampler = None
    test_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator,
        collate_fn=_collate_fn,
        sampler=sampler,
        batch_sampler=batch_sampler,
        persistent_workers=True,
    )

    return train_loader, val_loader, test_loader
