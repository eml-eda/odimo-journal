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

import argparse
import pathlib
from typing import Dict

from torchinfo import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_benchmarks.hr_detection as hrd
from pytorch_benchmarks.utils import seed_all, CheckPoint, EarlyStopping, AverageMeter

from exp.ppg import models


# Simply parse all models' names contained in model file
model_names = sorted(
    name
    for name in models.__dict__
    if not name.startswith("__") and callable(models.__dict__[name])
)


# Definition of evaluation function
def evaluate(
    search: bool,
    model: nn.Module,
    criterion: nn.Module,
    data: DataLoader,
    device: torch.device,
    reg_strength: torch.Tensor = torch.tensor(0.0),
) -> Dict[str, float]:
    model.eval()
    avgmae = AverageMeter("2.5f")
    avgloss = AverageMeter("2.5f")
    avglosstask = AverageMeter("2.5f")
    avglossreg = AverageMeter("2.5f")
    step = 0
    with torch.no_grad():
        for sample, target in data:
            step += 1
            sample, target = sample.to(device), target.to(device)
            output = model(sample)
            loss_task = criterion(output, target)
            if search:
                loss_reg = reg_strength * model.get_regularization_loss()
                loss = loss_task + loss_reg
            else:
                loss = loss_task
                loss_reg = 0.0
            mae_val = F.l1_loss(output, target)
            avgmae.update(mae_val, sample.size(0))
            avgloss.update(loss, sample.size(0))
            avglosstask.update(loss_task, sample.size(0))
            avglossreg.update(loss_reg, sample.size(0))
        final_metrics = {
            "loss": avgloss.get(),
            "loss_task": avglosstask.get(),
            "loss_reg": avglossreg.get(),
            "mae": avgmae.get(),
        }
    return final_metrics


# Definition of the function to train for one epoch
def train_one_epoch(
    epoch: int,
    search: bool,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dl: DataLoader,
    val_dl: DataLoader,
    test_dl: DataLoader,
    device: torch.device,
    reg_strength: torch.Tensor = torch.tensor(0.0),
) -> Dict[str, float]:
    model.train()
    avgmae = AverageMeter("2.5f")
    avgloss = AverageMeter("2.5f")
    avglosstask = AverageMeter("2.5f")
    avglossreg = AverageMeter("2.5f")
    step = 0
    with tqdm(total=len(train_dl), unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")
        for sample, target in train_dl:
            step += 1
            tepoch.update(1)
            sample, target = sample.to(device), target.to(device)
            output = model(sample)
            loss_task = criterion(output, target)
            if search:
                loss_reg = reg_strength * model.get_regularization_loss()
                loss = loss_task + loss_reg
            else:
                loss = loss_task
                loss_reg = 0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mae_val = F.l1_loss(output, target)
            avgmae.update(mae_val, sample.size(0))
            avgloss.update(loss, sample.size(0))
            avglosstask.update(loss_task, sample.size(0))
            avglossreg.update(loss_reg, sample.size(0))
            if step % 100 == 99:
                tepoch.set_postfix(
                    {
                        "loss": avgloss,
                        "loss_task": avglosstask,
                        "loss_reg": avglossreg,
                        "mae": avgmae,
                    }
                )
        val_metrics = evaluate(search, model, criterion, val_dl, device, reg_strength)
        val_metrics = {"val_" + k: v for k, v in val_metrics.items()}
        test_metrics = evaluate(search, model, criterion, test_dl, device)
        test_metrics = {"test_" + k: v for k, v in test_metrics.items()}
        final_metrics = {
            "loss": avgloss.get(),
            "loss_task": avglosstask.get(),
            "loss_reg": avglossreg.get(),
            "mae": avgmae.get(),
        }
        final_metrics.update(val_metrics)
        final_metrics.update(test_metrics)
        tepoch.set_postfix(final_metrics)
        tepoch.close()
        print(f"===Epoch {epoch}===")
        print(f"Metrics: {final_metrics}")
        print(f'Train Set Task Loss: {final_metrics["loss_task"]}')
        print(f'Train Set Reg Loss: {final_metrics["loss_reg"]}')
        print(f'Train Set MAE: {final_metrics["mae"]}')
        print(f'Val Set Loss: {final_metrics["val_loss"]}')
        print(f'Val Set MAE: {final_metrics["val_mae"]}')
        print(f'Test Set Loss: {final_metrics["test_loss"]}')
        print(f'Test Set MAE: {final_metrics["test_mae"]}')
        return final_metrics


def train_loop(
    model, epochs, checkpoint_dir, train_dl, val_dl, test_dl, device, train_again=False
):
    criterion = hrd.get_default_criterion()
    optimizer = hrd.get_default_optimizer(model)
    # optimizer = optim.SGD(model.parameters(),
    #                       lr=5e-2, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # Set EarlyStop with a patience of 50 epochs and CheckPoint
    earlystop = EarlyStopping(patience=50, mode="max")
    finetune_checkpoint = CheckPoint(
        checkpoint_dir / "finetune", model, optimizer, "min"
    )
    skip_finetune = True
    if (checkpoint_dir / "finetune.ckp").exists():
        if train_again:
            skip_finetune = False
            print("Running training again")
        else:
            print("Skipping training")
    else:
        skip_finetune = False
        print("Running training")

    if not skip_finetune:
        for epoch in range(epochs):
            metrics = train_one_epoch(
                epoch,
                False,
                model,
                criterion,
                optimizer,
                train_dl,
                val_dl,
                test_dl,
                device,
            )

            if epoch > 5:
                finetune_checkpoint(epoch, metrics["val_mae"])
                if earlystop(metrics["val_mae"]):
                    print(f"Stopping at epoch {epoch}")
                    break

        finetune_checkpoint.load_best()
        finetune_checkpoint.save(checkpoint_dir / "finetune.ckp")
    val_metrics = evaluate(False, model, criterion, val_dl, device)
    test_metrics = evaluate(False, model, criterion, test_dl, device)
    print("Training Best Val Set Loss:", val_metrics["loss"])
    print("Training Best Val Set MAE:", val_metrics["mae"])
    print("Training Test Set Loss @ Best on Val:", test_metrics["loss"])
    print("Training Test Set MAE @ Best on Val:", test_metrics["mae"])


def main(args):
    DATA_DIR = args.data_dir
    CHECKPOINT_DIR = pathlib.Path(args.checkpoint_dir)
    N_EPOCHS = args.epochs

    # Check CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)

    # Ensure determinstic execution
    seed_all(seed=args.seed)

    # Get the Data
    data_dir = pathlib.Path(DATA_DIR)
    data_gen = hrd.get_data(data_dir=data_dir, cross_val=True)
    datasets = next(data_gen)
    test_subj = datasets[2].test_subj
    dataloaders = hrd.build_dataloaders(datasets, seed=args.seed)
    train_dl, val_dl, test_dl = dataloaders

    # Get and build the Model
    model_fn = models.__dict__[args.arch]
    model = model_fn()
    model = model.to(device)

    # Model Summary
    stats = summary(model, (1,) + model.input_shape, mode="eval")
    print(stats)

    # Eventually load pretrained model
    if args.pretrained_model is not None:
        state_dict = torch.load(args.pretrained_model)["model_state_dict"]
        model.load_state_dict(state_dict)
        # Eval
        criterion = hrd.get_default_criterion()
        pretrained_metrics = evaluate(False, model, criterion, test_dl, device)
        print("Pretrained Test Set MAE:", pretrained_metrics["mae"])

    # Training Phase
    train_loop(model, N_EPOCHS, CHECKPOINT_DIR, train_dl, val_dl, test_dl, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Training")
    parser.add_argument("--arch", type=str, help=f"Arch name taken from {model_names}")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to Directory with Training Data",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Path to Directory where to save checkpoints",
    )
    parser.add_argument("--epochs", type=int, help="Number of Training Epochs")
    parser.add_argument(
        "--pretrained-model", type=str, default=None, help="Path to pretrained model"
    )
    parser.add_argument("--seed", type=int, default=14, help="Random Seed")
    args = parser.parse_args()
    main(args)
