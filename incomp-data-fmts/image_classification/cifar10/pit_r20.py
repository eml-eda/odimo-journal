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

import argparse
import pathlib
import random
from typing import Dict, Union, Optional, Tuple
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision
import torchvision.transforms as transforms

from plinio.cost import params
from plinio.methods import PIT

from pytorch_benchmarks.utils import AverageMeter, accuracy, EarlyStopping, CheckPoint

from models import quantres20_fp_foldbn


def _evaluate(
    model: nn.Module,
    criterion: nn.Module,
    data: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    avgacc = AverageMeter("6.2f")
    avgloss = AverageMeter("2.5f")
    step = 0
    with torch.no_grad():
        for sample, target in data:
            step += 1
            sample, target = sample.to(device), target.to(device)
            output = model(sample)
            loss = criterion(output, target)
            acc_val = accuracy(output, target, topk=(1,))
            avgacc.update(acc_val[0], sample.size(0))
            avgloss.update(loss, sample.size(0))
        final_metrics = {
            "loss": avgloss.get(),
            "acc": avgacc.get(),
        }
    return final_metrics


def _train_one_epoch(
    search: bool,
    model: nn.Module,
    criterion: nn.Module,
    net_optimizer: torch.optim.Optimizer,
    arch_optimizer: Optional[torch.optim.Optimizer],
    train: torch.utils.data.DataLoader,
    val: torch.utils.data.DataLoader,
    device: torch.device,
    strength: float = 0.0,
) -> Dict[str, float]:
    model.train()
    avgacc = AverageMeter("6.2f")
    avgloss = AverageMeter("2.5f")
    step = 0
    for sample, target in train:
        step += 1
        sample, target = sample.to(device), target.to(device)
        output = model(sample)
        loss = criterion(output, target)
        if search:
            loss = loss + strength * model.cost
        net_optimizer.zero_grad()
        if arch_optimizer is not None:
            arch_optimizer.zero_grad()
        loss.backward()
        net_optimizer.step()
        if arch_optimizer is not None:
            arch_optimizer.step()
        acc_val = accuracy(output, target, topk=(1,))
        avgacc.update(acc_val[0], sample.size(0))
        avgloss.update(loss, sample.size(0))
    val_metrics = _evaluate(model, criterion, val, device)
    val_metrics = {"val_" + k: v for k, v in val_metrics.items()}
    final_metrics = {
        "loss": avgloss.get(),
        "acc": avgacc.get(),
    }
    final_metrics.update(val_metrics)
    return final_metrics


def _train_loop(
    search: bool,
    model: Union[PIT, nn.Module],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    arch_optimizer: Optional[torch.optim.Optimizer],
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epochs: int,
    strength: float = 0.0,
    save_dir: pathlib.Path = pathlib.Path("."),
):
    save_dir = save_dir / "search" if search else save_dir / "finetune"
    save_dir.mkdir(parents=True, exist_ok=True)
    cp = CheckPoint(save_dir, model, optimizer, mode="max", save_best_only=True)
    early_stopping = EarlyStopping(patience=20, mode="max")

    for epoch in range(epochs):
        model.train()
        train_metr = _train_one_epoch(
            search,
            model,
            criterion,
            optimizer,
            arch_optimizer,
            train_loader,
            val_loader,
            "cuda",
            strength=strength,
        )
        print(f"Epoch {epoch}/{epochs}: {train_metr}")
        scheduler.step()
        if search:
            print(f"Cost after epoch {epoch}: {model.cost}")
            if not model.discrete_cost:
                model.discrete_cost = True
                print(f"Discrete cost after epoch {epoch}: {model.cost}")
                model.discrete_cost = False
            print(f"Model: {model.summary()}")

        test_metr = _evaluate(model, criterion, test_loader, "cuda")
        print(f"Test metrics: {test_metr}")

        # Save best model on val
        if epoch > 10:
            cp(epoch, train_metr["val_acc"])
            if early_stopping(train_metr["val_acc"]):
                print("Early Stopping!")
                break
    cp.load_best()
    test_metr_final = _evaluate(model, criterion, test_loader, "cuda")
    print(f"Final test metrics: {test_metr_final}")

    if search:
        if not model.discrete_cost:
            model.discrete_cost = True
            print(f"Discrete cost final: {model.cost}")
            model.discrete_cost = False
        print(f"Final model: {model.summary()}")
        cp.save(save_dir / "best_search.ckp")
    else:
        cp.save(save_dir / "best_finetune.ckp")


def main(args: argparse.Namespace):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )
    args.data = pathlib.Path(args.data)

    # Get the data
    num_classes = 10
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    data_dir = args.data.parent.parent.parent / "data"
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    # Split dataset into train and validation
    train_len = int(len(train_set) * 0.9)
    val_len = len(train_set) - train_len
    # Fix generator seed for reproducibility
    data_gen = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_set, [train_len, val_len], generator=data_gen
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Create model and transform it using PIT
    model = quantres20_fp_foldbn(
        arch_cfg_path="warmup20_fp.pth.tar",
        num_classes=num_classes,
        fine_tune=True,
    )
    pit_model = PIT(model, input_shape=(3, 32, 32), cost=params, discrete_cost=True)
    pit_model = pit_model.to("cuda")
    print(f"Cost before pruning: {pit_model.cost}")

    # Evaluate the model
    pre_search_eval = _evaluate(
        pit_model, nn.CrossEntropyLoss().to("cuda"), test_loader, "cuda"
    )
    print(f"Pre-search evaluation: {pre_search_eval}")

    criterion = nn.CrossEntropyLoss().to("cuda")
    # Use same optimizer setup used in ODiMO search phase
    optimizer = torch.optim.Adam(
        pit_model.net_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[100, 150],
        last_epoch=-1,
    )
    arch_optimizer = torch.optim.Adam(
        pit_model.nas_parameters(),
        lr=args.lra,
    )

    # Search Phase
    _train_loop(
        True,
        pit_model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        arch_optimizer,
        scheduler,
        args.epochs,
        args.strength,
        data_dir,
    )

    # Fine-tuning
    discovered_model = pit_model.cpu().export()
    discovered_model = discovered_model.to("cuda")
    ft_optimizer = torch.optim.SGD(
        discovered_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    ft_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        ft_optimizer,
        milestones=[100, 150],
        last_epoch=-1,
    )
    _train_loop(
        False,
        discovered_model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        ft_optimizer,
        None,
        ft_scheduler,
        args.epochs,
        args.strength,
        data_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PIT-R20 on CIFAR10")
    parser.add_argument("data", metavar="DIR", help="path to dataset")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lrq", default=1e-5, type=float)
    parser.add_argument("--lra", default=0.001, type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strength", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
