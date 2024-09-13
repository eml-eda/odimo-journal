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
from datetime import datetime
import logging
import math
import pathlib

from torchinfo import summary
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import cifar100_benchmark as c100
from pytorch_benchmarks.utils import seed_all

from exp.common import models
from exp.common.utils import DDPCheckPoint, ddp_setup, get_free_port

# Simply parse all models' names contained in model file
model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))


def train_loop(model, epochs, checkpoint_dir, train_dl,
               val_dl, test_dl, device,
               lr, momentum, weight_decay,
               WARMP,
               start_epoch=-1,
               optimizer_state_dict=None,
               train_again=False):
    criterion = c100.get_default_criterion()
    optimizer = c100.get_default_optimizer(model,
                                           lr=lr,
                                           momentum=momentum,
                                           weight_decay=weight_decay)
    iter_per_epoch = len(train_dl)
    warmup_scheduler = c100.WarmUpLR(optimizer, iter_per_epoch * WARMP)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    scheduler = c100.get_default_scheduler(optimizer)
    checkpoint = DDPCheckPoint(checkpoint_dir, model, optimizer, 'max',
                               save_best_only=True, save_last_epoch=True)

    # Train
    for epoch in range(start_epoch+1, epochs):
        if epoch > WARMP - 1:
            scheduler.step()
        metrics = c100.train_one_epoch(
            epoch, model, criterion, optimizer, train_dl, val_dl,
            device, warmup_scheduler, WARMP)
        if val_dl is not None:
            logging.info(f"Val Set Loss: {metrics['val_loss']}")
            logging.info(f"Val Set Accuracy: {metrics['val_acc']}")
        logging.info(f"Actual LR: {scheduler.get_last_lr()}")
        # Test
        test_metrics = c100.evaluate(model, criterion, test_dl, device)
        logging.info(f"Test Set Loss: {test_metrics['loss']}")
        logging.info(f"Test Set Accuracy: {test_metrics['acc']}")

        if device == 0:
            if val_dl is not None:
                checkpoint(epoch, metrics['val_acc'])
            else:
                checkpoint(epoch, test_metrics['acc'])

    if device == 0:
        checkpoint.load_best()
        if val_dl is not None:
            val_metrics = c100.evaluate(model, criterion, val_dl, device)
            test_metrics = c100.evaluate(model, criterion, test_dl, device)
            logging.info(f"Best Val Set Loss: {val_metrics['loss']}")
            logging.info(f"Best Val Set Accuracy: {val_metrics['acc']}")
            logging.info(f"Test Set Loss @ Best on Val: {test_metrics['loss']}")
            logging.info(f"Test Set Accuracy @ Best on Val: {test_metrics['acc']}")
        else:
            test_metrics = c100.evaluate(model, criterion, test_dl, device)
            logging.info(f"Test Set Loss: {test_metrics['loss']}")
            logging.info(f"Test Set Accuracy: {test_metrics['acc']}")


def main(rank, world_size, port, args):
    # Extract arguments
    DATA_DIR = args.data_dir
    CHECKPOINT_DIR = args.checkpoint_dir
    N_EPOCHS = args.epochs
    VAL_SPLIT = args.val_split
    WARMP = args.warm

    # Set up logging in the worker process
    logging.basicConfig(filename=CHECKPOINT_DIR / 'log.txt',
                        level=logging.INFO, format='%(asctime)s [%(process)d] %(message)s')

    # Setup ddp
    ddp_setup(rank, world_size, port)

    # Ensure determinstic execution
    seed_all(seed=args.seed)

    # Get the Data
    data_dir = DATA_DIR
    datasets = c100.get_data(data_dir=data_dir, val_split=VAL_SPLIT, seed=args.seed)
    dataloaders = c100.build_dataloaders(datasets, seed=args.seed, sampler_fn=DistributedSampler)
    # dataloaders = c100.build_dataloaders(datasets, seed=args.seed,
    #                                     sampler_fn=c100.VariableBatchSamplerDDP)
    train_dl, val_dl, test_dl = dataloaders
    input_shape = (3, 32, 32)

    # Get and build the Model
    model_fn = models.__dict__[args.arch]
    model = model_fn(input_shape, 100, 1)
    model = model.to(rank)

    # Eventually load previous checkpoint
    if args.pretrained_model is not None:
        # Load checkpoint, extract info and load weights
        logging.info(f"Loading checkpoint from {args.pretrained_model}")
        ckp = torch.load(args.pretrained_model, map_location='cpu')
        model_state_dict = ckp['model_state_dict']
        optimizer_state_dict = ckp['optimizer_state_dict']
        last_epoch = ckp['epoch']
        ckp_test_accuracy = ckp['val']
        model.load_state_dict(model_state_dict)

        # Run eval with pretrained model
        criterion = c100.get_default_criterion()
        pretrained_metrics = c100.evaluate(model, criterion, test_dl, rank)
        logging.info(f"Pretrained Test Set Accuracy: {pretrained_metrics['acc']}")
        logging.info(f"Checkpoint Test Set Accuracy: {ckp_test_accuracy}")

        # Add eval consistency check
        msg = 'Mismatch in test set accuracy'
        assert math.isclose(ckp_test_accuracy, pretrained_metrics['acc'], rel_tol=5e-1), msg
    else:
        optimizer_state_dict = None
        last_epoch = -1

    # Simply return in case of dry-run
    if args.dry_run:
        msg = 'To run dry-run you need to provide path to a --pretrained-model'
        assert args.pretrained_model is not None, msg
        return

    # Move model to DDP
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    # Model Summary
    if rank == 0:
        stats = summary(model, (1,) + input_shape, mode='eval')
        logging.info(stats)

    # Training Phase
    train_loop(model, N_EPOCHS, CHECKPOINT_DIR, train_dl,
               val_dl, test_dl, rank,
               lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
               WARMP=WARMP,
               start_epoch=last_epoch,
               optimizer_state_dict=optimizer_state_dict)

    destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline Training')
    parser.add_argument('--arch', type=str, help=f'Arch name taken from {model_names}')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to Directory with Training Data')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Path to Directory where to save checkpoints')
    parser.add_argument('--timestamp', type=str, default=None,
                        help='Timestamp, if not provided will be current time')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of Training Epochs')
    parser.add_argument('--warm', default=1, type=int, metavar='N',
                        help='number of warmup epochs')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 128), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--patience', default=40, type=int, metavar='N',
                        help='number of epochs wout improvements to wait before early stopping')
    parser.add_argument('--val-split', default=0.1, type=float,
                        help='Percentage of training data to be used as validation set')
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--pretrained-model', type=str, default=None,
                        help='Path to pretrained model')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='Dry run with test-set using passed pretrained model')
    parser.add_argument('--seed', type=int, default=None,
                        help='RNG Seed, if not provided will be random')
    parser.add_argument('--world-size', type=int, default=1,
                        help='Number of GPUs to use')
    args = parser.parse_args()

    # Set-up directories
    if args.checkpoint_dir is None:
        args.checkpoint_dir = pathlib.Path().cwd()
    else:
        args.checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    if args.timestamp is None:
        args.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.checkpoint_dir = args.checkpoint_dir / args.timestamp
    # Maybe create directories
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging in the main process
    logging.basicConfig(filename=args.checkpoint_dir / 'log.txt',
                        level=logging.INFO, format='%(asctime)s [%(process)d] %(message)s')

    world_size = args.world_size
    port = get_free_port()
    mp.spawn(main, args=(world_size, port, args), nprocs=world_size)
