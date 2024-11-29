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
import os
import pathlib
import random
import shutil
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision

import pytorch_benchmarks.hr_detection as hrd

import models

# Uncomment to disable debug features and to go faster
# torch.autograd.set_detect_anomaly(False)
# torch.autograd.profiler.profile(False)
# torch.autograd.profiler.emit_nvtx(False)

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch Dalia Training")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="quanttemponet_fp",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet8)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
# MR
parser.add_argument(
    "-d", "--dataset", default="None", type=str, help="cifar10 or cifar100"
)
parser.add_argument(
    "--epochs", default=200, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--patience",
    default=20,
    type=int,
    metavar="N",
    help="number of epochs wout improvements to wait before early stopping",
)
parser.add_argument(
    "--step-epoch",
    default=50,
    type=int,
    metavar="N",
    help="number of epochs to decay learning rate",
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=128,
    type=int,
    metavar="N",
    help="mini-batch size (default: 128), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "--lrq",
    "--learning-rate-q",
    default=1e-5,
    type=float,
    metavar="LR",
    help="initial q learning rate",
    dest="lrq",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=100,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--arch-cfg",
    "--ac",
    default="",
    type=str,
    metavar="PATH",
    help="path to architecture configuration",
)
# MR
parser.add_argument(
    "-ft",
    "--fine-tune",
    dest="fine_tune",
    action="store_true",
    help="use pre-trained weights from search phase",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--test",
    dest="test",
    action="store_true",
    help="evaluate model on test set",
)
parser.add_argument(
    "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://224.66.41.62:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)
parser.add_argument(
    "--visualization",
    dest="visualization",
    action="store_true",
    help="visualize training logs using wandb",
)
parser.add_argument(
    "-pr", "--project", default="misc", type=str, help="wandb project name"
)
parser.add_argument("--tags", nargs="+", default=None, help="wandb tags")


best_mae = float("inf")


def main():
    args = parser.parse_args()
    print(args)

    complexity_decay = args.data.split("_")[-1]

    args.data = pathlib.Path(args.data)

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

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_mae
    global best_mae_test
    best_mae_test = float("inf")
    mae_test = float("inf")
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    # Get the data
    data_dir = args.data.parent.parent.parent / "data"
    data_gen = hrd.get_data(data_dir=data_dir, cross_val=True)
    datasets = next(data_gen)
    test_subj = datasets[2].test_subj
    dataloaders = hrd.build_dataloaders(datasets, seed=args.seed)
    train_dl, val_dl, test_dl = dataloaders

    # Get max (abs) input val
    max_inp_val = max(abs(datasets[0].samples.flatten()))

    # create model
    print("=> creating model '{}'".format(args.arch))
    if len(args.arch_cfg) > 0:
        if os.path.isfile(args.arch_cfg):
            print("=> loading architecture config from '{}'".format(args.arch_cfg))
        else:
            print("=> no architecture found at '{}'".format(args.arch_cfg))
    model_fn = models.__dict__[args.arch]
    model = model_fn(args.arch_cfg, fine_tune=args.fine_tune, max_inp_val=max_inp_val)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if "alex" in args.arch or "vgg" in args.arch:
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = hrd.get_default_criterion()
    # group model/quantization parameters
    params, q_params = [], []
    for name, param in model.named_parameters():
        if ("clip_val" in name) or ("scale_param" in name):
            q_params += [param]
        else:
            params += [param]

    optimizer = torch.optim.Adam(
        params,
        args.lr,
    )
    scheduler = None

    if q_params:
        q_optimizer = torch.optim.SGD(q_params, args.lrq)
        q_scheduler = torch.optim.lr_scheduler.StepLR(q_optimizer, 50)
    else:
        q_optimizer = None
        q_scheduler = None

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            best_mae = checkpoint["best_acc1"]
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_mae = best_mae.to(args.gpu)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_dl, model, criterion, 0, args)
        return

    if args.test:
        validate(test_dl, model, criterion, 0, args)
        return

    best_epoch = args.start_epoch
    epoch_wout_improve = 0

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_dl, model, criterion, optimizer, q_optimizer, epoch, args)

        # evaluate on validation set
        mae = validate(val_dl, model, criterion, epoch, args)
        mae_test = validate(test_dl, model, criterion, epoch, args)

        if scheduler is not None:
            scheduler.step()
        if q_scheduler is not None:
            q_scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = mae < best_mae
        if is_best:
            best_epoch = epoch
            best_mae = mae
            best_mae_test = mae_test
            epoch_wout_improve = 0
            print(f"New best MAE Val: {best_mae}")
            print(f"New best MAE Test: {best_mae_test}")
        else:
            epoch_wout_improve += 1
            print(f"Epoch without improvement: {epoch_wout_improve}")

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            save_checkpoint(
                args.data,
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_mae": best_mae,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                epoch,
                args.step_epoch,
            )

        # Early-Stop
        if epoch_wout_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    best_mae_val = best_mae
    print("Best MAE_val@1 {0} @ epoch {1}".format(best_mae_val, best_epoch))

    test_mae = best_mae_test
    print("Test MAE_val@1 {0} @ epoch {1}".format(test_mae, best_epoch))


def train(train_loader, model, criterion, optimizer, q_optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    avgloss = AverageMeter("Loss", ":2.5f")
    avgmae = AverageMeter("MAE", ":2.5f")
    curr_lr = optimizer.param_groups[0]["lr"]
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, avgloss, avgmae],
        prefix="Epoch: [{}/{}]\t" "LR: {}\t".format(epoch, args.epochs, curr_lr),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        mae_val = F.l1_loss(output, target)
        avgmae.update(mae_val, images.size(0))
        avgloss.update(loss.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if q_optimizer is not None:
            q_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if q_optimizer is not None:
            q_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    avgmae = AverageMeter("MAE", ":6.2f")
    avgloss = AverageMeter("Loss", ":2.5f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, avgloss, avgmae], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            avgloss.update(loss.item(), images.size(0))
            avgmae.update(F.l1_loss(output, target), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(f" * MAE {avgmae.avg:.6f}")

    return avgmae.avg


def save_checkpoint(
    root, state, is_best, epoch, step_epoch, filename="checkpoint.pth.tar"
):
    torch.save(state, root / filename)
    if is_best:
        shutil.copyfile(root / filename, root / "model_best.pth.tar")
    if (epoch + 1) % step_epoch == 0:
        shutil.copyfile(
            root / filename, root / "checkpoint_ep{}.pth.tar".format(epoch + 1)
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    initial_learning_rate = 0.001
    decay_per_epoch = 0.99
    lrate = initial_learning_rate * (decay_per_epoch**epoch)
    for opt in optimizer.param_groups:
        opt["lr"] = lrate


if __name__ == "__main__":
    main()
