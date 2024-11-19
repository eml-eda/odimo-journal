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

from plinio.cost import params as pit_params
from plinio.methods import PIT

from pytorch_benchmarks.utils import AverageMeter, accuracy, CheckPoint

from models import hw_models as hw
from models import quant_module_pow2 as qm2
from models import quantres20_fp_foldbn
from models import utils
from models.quant_resnet import ResNet20


def _build_quantized_model(
    fp_model: nn.Module,
    num_channels: Dict[str, Tuple[int, int]],
) -> nn.Module:
    archas, archws = [[7]] * 22, [[8]] * 22
    s_up = 5.0
    q_model = ResNet20(
        qm2.QuantMultiPrecActivConv2d,
        hw.diana(analog_speedup=s_up),
        archws,
        archas,
        qtz_fc="multi",
        bn=False,
        target="latency",
        **{},
    )

    # Remove input and output channels from q_model following `num_channels`
    for name, module in q_model.named_modules():
        if name in map(
            # lambda x: ".mix_weight".join(x.rsplit(".conv", 1)),
            lambda x: "".join(x.rsplit(".conv", 1)),
            list(num_channels.keys()),
        ):
            # c_in = num_channels[name.replace(".mix_weight", ".conv")][0]
            # c_out = num_channels[name.replace(".mix_weight", ".conv")][1]
            c_in = num_channels[name + ".conv"][0]
            c_out = num_channels[name + ".conv"][1]

            module.ch_in = c_in
            module.ch_out = c_out

            module.mix_weight.cout = c_out
            with torch.no_grad():
                module.mix_weight.alpha_weight = nn.Parameter(
                    module.mix_weight.alpha_weight[:, :c_out]
                )

            module.mix_weight.mix_weight[0].cout = c_out
            module.mix_weight.mix_bias[0].cout = c_out

            module.mix_weight.conv.out_channels = c_out
            module.mix_weight.conv.in_channels = c_in
            # Update the weight and bias tensor
            with torch.no_grad():
                module.mix_weight.conv.weight = nn.Parameter(
                    module.mix_weight.conv.weight[
                        : module.mix_weight.conv.out_channels,
                        : module.mix_weight.conv.in_channels,
                    ]
                )
                module.mix_weight.conv.bias = nn.Parameter(
                    module.mix_weight.conv.bias[: module.mix_weight.conv.out_channels]
                )

    q_state_dict = utils.fpfold_to_q(fp_model.state_dict())
    q_model.load_state_dict(q_state_dict, strict=False)

    utils.init_scale_param(q_model)

    return q_model


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


def _extract_num_channels(model: nn.Module) -> Dict[str, Tuple[int, int]]:
    num_channels = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            num_channels[name] = (module.in_channels, module.out_channels)
        if isinstance(module, nn.Linear):
            num_channels[name] = (module.in_features, module.out_features)
    return num_channels


def _train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    net_optimizer: torch.optim.Optimizer,
    q_optimizer: Optional[torch.optim.Optimizer],
    train: torch.utils.data.DataLoader,
    val: torch.utils.data.DataLoader,
    device: torch.device,
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
        net_optimizer.zero_grad()
        q_optimizer.zero_grad()
        loss.backward()
        net_optimizer.step()
        q_optimizer.step()
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
    model: Union[PIT, nn.Module],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    q_optimizer: Optional[torch.optim.Optimizer],
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    q_scheduler: torch.optim.lr_scheduler._LRScheduler,
    epochs: int,
    save_dir: pathlib.Path = pathlib.Path("."),
):
    save_dir = save_dir / "qtz"
    save_dir.mkdir(parents=True, exist_ok=True)
    cp = CheckPoint(save_dir, model, optimizer, mode="max", save_best_only=True)

    for epoch in range(epochs):
        model.train()
        train_metr = _train_one_epoch(
            model,
            criterion,
            optimizer,
            q_optimizer,
            train_loader,
            val_loader,
            "cuda",
        )
        print(f"Epoch {epoch}/{epochs}: {train_metr}")
        scheduler.step()
        q_scheduler.step()

        test_metr = _evaluate(model, criterion, test_loader, "cuda")
        print(f"Test metrics: {test_metr}")

        # Save best model on val
        if epoch > 10:
            cp(epoch, train_metr["val_acc"])
    cp.load_best()
    test_metr_final = _evaluate(model, criterion, test_loader, "cuda")
    print(f"Final test metrics: {test_metr_final}")

    cp.save(save_dir / "best_quantized.ckp")


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
    pit_model = PIT(model, input_shape=(3, 32, 32), cost=pit_params, discrete_cost=True)
    pit_model = pit_model.to("cuda")
    print(f"Cost before pruning: {pit_model.cost}")

    # Evaluate the model
    pre_search_eval = _evaluate(
        pit_model, nn.CrossEntropyLoss().to("cuda"), test_loader, "cuda"
    )
    print(f"Pre-search evaluation: {pre_search_eval}")

    # Load search checkpoint
    search_sd = torch.load(args.data / "search" / "best_search.ckp")["model_state_dict"]
    pit_model.load_state_dict(search_sd)
    print(f"Cost after loading search checkpoint: {pit_model.cost}")

    # Export the model and load fine-tuning checkpoint
    discovered_model = pit_model.cpu().export()
    finetune_sd = torch.load(args.data / "finetune" / "best_finetune.ckp")[
        "model_state_dict"
    ]
    discovered_model.load_state_dict(finetune_sd)

    # Evaluate the discovered model
    discovered_model = discovered_model.to("cuda")
    post_ft_eval = _evaluate(
        discovered_model, nn.CrossEntropyLoss().to("cuda"), test_loader, "cuda"
    )
    print(f"Post-fine-tuning evaluation: {post_ft_eval}")

    # Extract the number of channels/features for every layer
    num_channels = _extract_num_channels(discovered_model)

    # Build the quantized model and test it
    q_model = _build_quantized_model(discovered_model, num_channels)
    q_model = q_model.to("cuda")
    q_model_eval = _evaluate(
        q_model, nn.CrossEntropyLoss().to("cuda"), test_loader, "cuda"
    )
    # Get model complexity
    cycles, _, _ = q_model.fetch_arch_info()
    print(f"Model complexity: {cycles}")
    print(f"Quantized model evaluation: {q_model_eval}")

    criterion = nn.CrossEntropyLoss().to("cuda")

    # group model/quantization parameters
    params, q_params = [], []
    for name, param in q_model.named_parameters():
        if ("clip_val" in name) or ("scale_param" in name):
            q_params += [param]
        else:
            params += [param]

    optimizer = torch.optim.SGD(
        params, args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], last_epoch=-1
    )

    if q_params:
        q_optimizer = torch.optim.SGD(q_params, args.lrq)
        q_scheduler = torch.optim.lr_scheduler.StepLR(q_optimizer, 50)
    else:
        q_optimizer = None
        q_scheduler = None

    # Train the quantized model
    _train_loop(
        q_model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        q_optimizer,
        scheduler,
        q_scheduler,
        args.epochs,
        args.data,
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
    parser.add_argument("--lrft", default=0.01, type=float)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
