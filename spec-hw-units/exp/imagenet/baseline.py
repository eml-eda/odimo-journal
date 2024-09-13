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
from torch.optim import swa_utils
from torch.utils.data.distributed import DistributedSampler

import imagenet_benchmark as imn
from pytorch_benchmarks.utils import seed_all

from exp.common import models
from exp.common.utils import DDPCheckPoint, ddp_setup, get_free_port

# Simply parse all models' names contained in model file
model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))


def train_loop(model, epochs, checkpoint_dir, train_dl,
               val_dl, test_dl, device,
               use_ema=None,
               start_epoch=-1,
               optimizer_state_dict=None,
               train_again=False):
    criterion = imn.get_default_criterion()
    optimizer = imn.get_default_optimizer(model)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    scheduler = imn.get_default_scheduler(optimizer, verbose=False)
    checkpoint = DDPCheckPoint(checkpoint_dir, model, optimizer, 'max',
                               save_best_only=True, save_last_epoch=True)

    # Train
    scaler = torch.cuda.amp.GradScaler()
    if use_ema:
        # `use_buffers=True` ensures update of bn statistics.
        # torch doc says that it may increase accuracy.
        ema_model = swa_utils.AveragedModel(model,
                                            multi_avg_fn=swa_utils.get_ema_multi_avg_fn(0.9995),
                                            use_buffers=True)
        checkpoint_ema = DDPCheckPoint(checkpoint_dir / 'ema', ema_model, optimizer, 'max',
                                       save_best_only=True, save_last_epoch=True)
    else:
        ema_model = None

    for epoch in range(start_epoch+1, epochs):
        metrics = imn.train_one_epoch(
            epoch, model, criterion, optimizer, scheduler, train_dl, val_dl,
            device, scaler, ema_model=ema_model)
        if val_dl is not None:
            logging.info(f"Val Set Loss: {metrics['val_loss']}")
            logging.info(f"Val Set Accuracy: {metrics['val_acc']}")
        logging.info(f"Actual LR: {scheduler.get_last_lr()}")
        # Test
        test_metrics = imn.evaluate(model, criterion, test_dl, device)
        logging.info(f"Test Set Loss: {test_metrics['loss']}")
        logging.info(f"Test Set Accuracy: {test_metrics['acc']}")
        # EMA
        if ema_model is not None and device == 0:
            ema_test_metrics = imn.evaluate(ema_model, criterion, test_dl, device)
            logging.info(f"EMA Test Set Loss: {ema_test_metrics['loss']}")
            logging.info(f"EMA Test Set Accuracy: {ema_test_metrics['acc']}")

        if device == 0:
            if val_dl is not None:
                checkpoint(epoch, metrics['val_acc'])
            else:
                checkpoint(epoch, test_metrics['acc'])
            if use_ema:
                checkpoint_ema(epoch, ema_test_metrics['acc'])

    if device == 0:
        checkpoint.load_best()
        if val_dl is not None:
            val_metrics = imn.evaluate(model, criterion, val_dl, device)
            test_metrics = imn.evaluate(model, criterion, test_dl, device)
            logging.info(f"Best Val Set Loss: {val_metrics['loss']}")
            logging.info(f"Best Val Set Accuracy: {val_metrics['acc']}")
            logging.info(f"Test Set Loss @ Best on Val: {test_metrics['loss']}")
            logging.info(f"Test Set Accuracy @ Best on Val: {test_metrics['acc']}")
        else:
            test_metrics = imn.evaluate(model, criterion, test_dl, device)
            logging.info(f"Test Set Loss: {test_metrics['loss']}")
            logging.info(f"Test Set Accuracy: {test_metrics['acc']}")
        if use_ema:
            checkpoint_ema.load_best()
            ema_test_metrics = imn.evaluate(ema_model, criterion, test_dl, device)
            logging.info(f"EMA Test Set Loss: {ema_test_metrics['loss']}")
            logging.info(f"EMA Test Set Accuracy: {ema_test_metrics['acc']}")


def main(rank, world_size, port, args):
    # Extract arguments
    DATA_DIR = args.data_dir
    CHECKPOINT_DIR = args.checkpoint_dir
    N_EPOCHS = args.epochs
    USE_EMA = args.use_ema
    # USE_AMP = # TODO

    # Set up logging in the worker process
    logging.basicConfig(filename=CHECKPOINT_DIR / 'log.txt',
                        level=logging.INFO, format='%(asctime)s [%(process)d] %(message)s')

    # Setup ddp
    ddp_setup(rank, world_size, port)

    # Ensure determinstic execution
    seed_all(seed=args.seed)

    # Get the Data
    data_dir = DATA_DIR
    datasets = imn.get_data(data_dir=data_dir, val_split=0.0, seed=args.seed)
    dataloaders = imn.build_dataloaders(datasets, seed=args.seed, sampler_fn=DistributedSampler)
    # dataloaders = imn.build_dataloaders(datasets, seed=args.seed,
    #                                     sampler_fn=imn.VariableBatchSamplerDDP)
    train_dl, val_dl, test_dl = dataloaders
    input_shape = (3, 224, 224)

    # Get and build the Model
    model_fn = models.__dict__[args.arch]
    model = model_fn(input_shape, 1000)
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
        criterion = imn.get_default_criterion()
        pretrained_metrics = imn.evaluate(model, criterion, test_dl, rank)
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
               val_dl, test_dl, rank, USE_EMA,
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
    parser.add_argument('--epochs', type=int, help='Number of Training Epochs')
    parser.add_argument('--pretrained-model', type=str, default=None,
                        help='Path to pretrained model')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='Dry run with test-set using passed pretrained model')
    parser.add_argument('--seed', type=int, default=None,
                        help='RNG Seed, if not provided will be random')
    parser.add_argument('--world-size', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--use-amp', action='store_true', default=False,
                        help='Use Automatic Mixed Precision')
    parser.add_argument('--use-ema', action='store_true', default=False,
                        help='Use Exponential Moving Average')
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
