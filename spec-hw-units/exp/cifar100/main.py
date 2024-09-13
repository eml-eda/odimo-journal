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
import copy
from datetime import datetime
import logging
import pathlib

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, all_reduce, ReduceOp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import cifar100_benchmark as c100
from pytorch_benchmarks.utils import seed_all, EarlyStopping

from odimo.method import ThermometricNet

from exp.common import models
from exp.common.utils import DDPCheckPoint, ddp_setup, get_free_port

from exp.cifar100.train import train_one_epoch, evaluate

# Simply parse all models' names contained in model file
model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))


def resume_ckp(ckp, model, test_dl, rank):
    # Load checkpoint, extract info and load weights
    ckp = torch.load(ckp, map_location='cpu')
    model_state_dict = ckp['model_state_dict']
    optimizer_state_dict = ckp['optimizer_state_dict']
    last_epoch = ckp['epoch']
    # ckp_test_accuracy = ckp['val']
    model.module.load_state_dict(model_state_dict)

    # Run eval with pretrained model
    criterion = c100.get_default_criterion()
    pretrained_metrics = evaluate(False, model, criterion, test_dl, rank)
    logging.info(f"Pretrained Test Set Accuracy: {pretrained_metrics['acc']}")

    # Add eval consistency check
    # msg = 'Mismatch in test set accuracy'
    # assert ckp_test_accuracy == pretrained_metrics['acc'], msg

    return model, optimizer_state_dict, last_epoch


def warmup_loop(model, epochs, checkpoint_dir, train_dl,
                val_dl, test_dl, device,
                lr, momentum, weight_decay,
                WARMP,
                start_epoch=-1,
                optimizer_state_dict=None):
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
    warmup_checkpoint = DDPCheckPoint(checkpoint_dir, model, optimizer, 'max',
                                      save_best_only=True, save_last_epoch=True)
    skip_warmup = True
    # If exists it means that warmup phase completed
    if (checkpoint_dir.parent / 'warmup.ckp').exists():
        warmup_checkpoint.load(checkpoint_dir.parent / 'warmup.ckp')
        logging.info("Skipping warmup")
    else:
        skip_warmup = False
        logging.info("Running warmup")

    # Train
    if not skip_warmup:
        for epoch in range(start_epoch + 1, epochs):
            if epoch > WARMP - 1:
                scheduler.step()
            metrics = train_one_epoch(
                epoch, False, model, criterion, optimizer,
                train_dl, val_dl, device, warmup_scheduler, WARMP,
                )
            if val_dl is not None:
                logging.info(f"Val Set Loss: {metrics['val_loss']}")
                logging.info(f"Val Set Accuracy: {metrics['val_acc']}")
            logging.info(f"Actual LR: {scheduler.get_last_lr()}")
            # Test
            test_metrics = evaluate(False, model, criterion, test_dl, device)
            logging.info(f"Test Set Loss: {test_metrics['loss']}")
            logging.info(f"Test Set Accuracy: {test_metrics['acc']}")

            if device == 0:
                if val_dl is not None:
                    warmup_checkpoint(epoch, metrics['val_acc'])
                else:
                    warmup_checkpoint(epoch, test_metrics['acc'])

    if device == 0 and not skip_warmup:
        if not skip_warmup:
            warmup_checkpoint.load_best()
            # Save warmup checkpoint as warmup.ckp in parent folder
            warmup_checkpoint.save(checkpoint_dir.parent / 'warmup.ckp')
        if val_dl is not None:
            val_metrics = evaluate(False, model, criterion, val_dl, device)
            test_metrics = evaluate(False, model, criterion, test_dl, device)
            logging.info("Warmup Best Val Set Loss:", val_metrics['loss'])
            logging.info("Warmup Best Val Set Accuracy:", val_metrics['acc'])
            logging.info("Warmup Test Set Loss @ Best on Val:", test_metrics['loss'])
            logging.info("Warmup Test Set Accuracy @ Best on Val:", test_metrics['acc'])
        else:
            test_metrics = evaluate(False, model, criterion, test_dl, device)
            logging.info("Warmup Test Set Loss:", test_metrics['loss'])
            logging.info("Warmup Test Set Accuracy:", test_metrics['acc'])


def search_loop(model, epochs, strength, strength_increment, checkpoint_dir,
                train_dl, val_dl, test_dl, device,
                lr, momentum, weight_decay,
                WARMP,
                start_epoch=-1, optimizer_state_dict=None):
    criterion = c100.get_default_criterion()
    # optimizer = get_default_optimizer(model.module)
    # optimizer = c100.get_default_optimizer(model,
    #                                        lr=lr,
    #                                        momentum=momentum,
    #                                        weight_decay=weight_decay)
    param_dicts = [
        {'params': model.module.nas_parameters(), 'weight_decay': 0},
        {'params': model.module.net_parameters()}]
    optimizer = torch.optim.Adam(param_dicts, lr=0.001, weight_decay=1e-4)
    iter_per_epoch = len(train_dl)
    warmup_scheduler = c100.WarmUpLR(optimizer, iter_per_epoch * WARMP)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    scheduler = c100.get_default_scheduler(optimizer)
    earlystop = EarlyStopping(patience=100, mode='max')
    earlystop_flag = torch.zeros(1).to(device)
    search_checkpoint = DDPCheckPoint(checkpoint_dir, model, optimizer, 'max',
                                      save_best_only=True, save_last_epoch=True)
    skip_search = True
    # If exists it means that search phase is completed
    if (checkpoint_dir.parent / 'search.ckp').exists():
        search_checkpoint.load(checkpoint_dir.parent / 'search.ckp')
        logging.info("Skipping search")
    else:
        skip_search = False
        logging.info("Running search")

    # Train
    if not skip_search:
        for epoch in range(start_epoch + 1, epochs):
            if epoch > WARMP - 1:
                scheduler.step()
            # new_strength = min(strength/100 + strength_increment * epoch, strength)
            new_strength = strength
            logging.info(f"Epoch: {epoch}, Strength: {new_strength}")
            metrics = train_one_epoch(
                epoch, True, model, criterion, optimizer,
                train_dl, val_dl, device, warmup_scheduler, WARMP,
                reg_strength=new_strength)
            if val_dl is not None:
                logging.info(f"Val Set Loss: {metrics['val_loss']}")
                logging.info(f"Val Set Accuracy: {metrics['val_acc']}")
            logging.info(f"Actual LR: {scheduler.get_last_lr()}")
            # Test
            test_metrics = c100.evaluate(model, criterion, test_dl, device)
            logging.info(f"Test Set Loss: {test_metrics['loss']}")
            logging.info(f"Test Set Accuracy: {test_metrics['acc']}")

            if device == 0 and epoch > 200:
                if val_dl is not None:
                    search_checkpoint(epoch, metrics['val_acc'])
                    if earlystop(metrics['val_acc']):
                        earlystop_flag += 1
                else:
                    search_checkpoint(epoch, test_metrics['acc'])
            all_reduce(earlystop_flag, op=ReduceOp.SUM)
            if earlystop_flag > 0:
                logging.info(f"GPU {device}, early stopping at epoch: {epoch}")
                break
            if device == 0:
                logging.info("architectural summary:")
                logging.info(model.module)
                logging.info(f"model regularization: {model.module.get_latency()}")
        if device == 0:
            logging.info("Load best model")
            search_checkpoint.load_best()
            search_checkpoint.save(checkpoint_dir.parent / 'search.ckp')
    if device == 0:
        logging.info("final architectural summary:")
        logging.info(model.module)
        with torch.no_grad():
            logging.info(f"Final regularization: {model.module.get_latency()}")
    if val_dl is not None:
        val_metrics = evaluate(True, model, criterion, val_dl, device)
        test_metrics = evaluate(True, model, criterion, test_dl, device)
        logging.info(f"Search Best Val Set Loss: {val_metrics['loss']}")
        logging.info(f"Search Best Val Set Accuracy: {val_metrics['acc']}")
        logging.info(f"Search Test Set Loss @ Best on Val: {test_metrics['loss']}")
        logging.info(f"Search Test Set Accuracy @ Best on Val: {test_metrics['acc']}")
    else:
        test_metrics = evaluate(True, model, criterion, test_dl, device)
        logging.info(f"Search Test Set Loss: {test_metrics['loss']}")
        logging.info(f"Search Test Set Accuracy: {test_metrics['acc']}")


def finetune_loop(model, epochs, checkpoint_dir, train_dl,
                  val_dl, test_dl, device,
                  lr, momentum, weight_decay,
                  WARMP,
                  start_epoch=-1, optimizer_state_dict=None,
                  ft_again=False, ft_scratch=False):
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
    finetune_checkpoint = DDPCheckPoint(checkpoint_dir, model, optimizer, 'max',
                                        save_best_only=True, save_last_epoch=True)
    skip_finetune = True
    ft_again = False  # TODO: modify
    # If exists it means that fine-tune phase is completed
    if (checkpoint_dir.parent / 'finetune.ckp').exists():
        if not ft_scratch:
            finetune_checkpoint.load(checkpoint_dir.parent / 'finetune.ckp')
            logging.info("Skipping finetune")
        if ft_again:
            skip_finetune = False
            logging.info("Running finetune again")
        else:
            logging.info("Skipping finetune")
    else:
        skip_finetune = False
        logging.info("Running finetune")

    # Train
    if not skip_finetune:
        for epoch in range(start_epoch + 1, epochs):
            if epoch > WARMP - 1:
                scheduler.step()
            metrics = train_one_epoch(
                epoch, False, model, criterion, optimizer,
                train_dl, val_dl, device, warmup_scheduler, WARMP,
                )
            if val_dl is not None:
                logging.info(f"Val Set Loss: {metrics['val_loss']}")
                logging.info(f"Val Set Accuracy: {metrics['val_acc']}")
            logging.info(f"Actual LR: {scheduler.get_last_lr()}")
            # Test
            test_metrics = c100.evaluate(model, criterion, test_dl, device)
            logging.info(f"Test Set Loss: {test_metrics['loss']}")
            logging.info(f"Test Set Accuracy: {test_metrics['acc']}")

            if device == 0 and epoch > 5:
                if val_dl is not None:
                    finetune_checkpoint(epoch, metrics['val_loss'])
                else:
                    finetune_checkpoint(epoch, metrics['val_acc'])

        if device == 0:
            finetune_checkpoint.load_best()
            finetune_checkpoint.save(checkpoint_dir.parent / 'finetune.ckp')
    if device == 0:
        if val_dl is not None:
            val_metrics = evaluate(False, model, criterion, val_dl, device)
            test_metrics = evaluate(False, model, criterion, test_dl, device)
            logging.info(f"Finetune Best Val Set Loss: {val_metrics['loss']}")
            logging.info(f"Finetune Best Val Set Accuracy: {val_metrics['acc']}")
            logging.info(f"Finetune Test Set Loss @ Best on Val: {test_metrics['loss']}")
            logging.info(f"Finetune Test Set Accuracy @ Best on Val: {test_metrics['acc']}")
        else:
            test_metrics = evaluate(False, model, criterion, test_dl, device)
            logging.info(f"Finetune Test Set Loss: {test_metrics['loss']}")
            logging.info(f"Finetune Test Set Accuracy: {test_metrics['acc']}")


def main(rank, world_size, port, args):
    # Extract arguments
    DATA_DIR = args.data_dir
    CHECKPOINT_DIR = args.checkpoint_dir
    N_EPOCHS = args.epochs
    LAMBDA = args.strength
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
    train_dl, val_dl, test_dl = dataloaders
    input_shape = (3, 32, 32)

    # Get and build the Model
    model_fn = models.__dict__[args.arch]
    model = model_fn(input_shape, 100, 1)
    model = model.to(rank)

    # Warmup Phase
    if args.warmup:
        # Convert the model to ThermometricNet
        therm_model = ThermometricNet(model, input_shape=input_shape,
                                      cost=args.cost,
                                      init_strategy=args.init_strategy,
                                      warmup_strategy=args.warmup_strategy)
        therm_model = therm_model.to(rank)

        # Move model to DDP
        therm_model = nn.SyncBatchNorm.convert_sync_batchnorm(therm_model)
        therm_model = DDP(therm_model, device_ids=[rank])

        # Save initial state before any training
        if args.finetune_scratch:
            state_dict = therm_model.module.state_dict()
            init_state_dict = copy.deepcopy(state_dict)
            exclude_keys = ['alpha', 'ch_eff']
            for key in state_dict.keys():
                for ex_key in exclude_keys:
                    if ex_key in key:
                        init_state_dict.pop(key)

        warmup_dir = 'warmup_' + str(args.warmup_strategy)
        warmup_dir += '_init_' + str(args.init_strategy)
        logging.info(f'Model Phase: {therm_model.module.phase}')

        # Eventually load previous checkpoint
        if args.resume_ckp_warmup is not None:
            therm_model, optimizer_state_dict, last_epoch = \
                resume_ckp(args.resume_ckp_warmup, therm_model, test_dl, rank)
        else:
            optimizer_state_dict = None
            last_epoch = -1

        # Choose warmup strategy
        if args.warmup_strategy != 'std':
            warmup_loop(therm_model, N_EPOCHS,
                        CHECKPOINT_DIR.parent.parent / warmup_dir,
                        train_dl, val_dl, test_dl, rank,
                        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                        WARMP=WARMP,
                        start_epoch=last_epoch,
                        optimizer_state_dict=optimizer_state_dict)
        else:
            print('Warmup Phase I: Set combiner to always choose 2nd layer and train.')
            therm_model.module.set_init_strategy(strategy='2nd')
            warmup_dir_i = warmup_dir + '_phase_i'
            warmup_loop(therm_model, int(N_EPOCHS),
                        CHECKPOINT_DIR.parent.parent / warmup_dir_i,
                        train_dl, val_dl, test_dl, rank,
                        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                        WARMP=WARMP,
                        start_epoch=last_epoch,
                        optimizer_state_dict=optimizer_state_dict)
            print('Warmup Phase II: Set combiner to always choose 1st layer and train.')
            therm_model.module.set_init_strategy(strategy='1st')
            warmup_dir_ii = warmup_dir + '_phase_ii'
            warmup_loop(therm_model, int(N_EPOCHS),
                        CHECKPOINT_DIR.parent.parent / warmup_dir_ii,
                        train_dl, val_dl, test_dl, rank,
                        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                        WARMP=WARMP,
                        start_epoch=last_epoch,
                        optimizer_state_dict=optimizer_state_dict)

    # Search Phase
    if args.search:
        therm_model.module.set_init_strategy(strategy=args.init_strategy)
        therm_model.module.change_train_phase(phase='search', strategy=args.warmup_strategy)
        logging.info(f'Model Phase: {therm_model.module.phase}')
        logging.info(f'Initial Latency: {therm_model.module.get_real_latency()}')

        # Eventually load previous checkpoint
        logging.info(f'Resuming from {args.resume_ckp_search}')
        if args.resume_ckp_search is not None:
            logging.info(f'Resuming from {args.resume_ckp_search}')
            therm_model, optimizer_state_dict, last_epoch = \
                resume_ckp(args.resume_ckp_search, therm_model, test_dl, rank)
        else:
            optimizer_state_dict = None
            last_epoch = -1

        # Search Loop
        lambda_increment = (LAMBDA * 99/100) / int(N_EPOCHS/2)
        search_loop(therm_model, N_EPOCHS,
                    LAMBDA, lambda_increment,
                    CHECKPOINT_DIR / 'search',
                    train_dl, val_dl, test_dl, rank,
                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                    WARMP=WARMP,
                    start_epoch=last_epoch,
                    optimizer_state_dict=optimizer_state_dict)

        logging.info(f'Final Latency: {therm_model.module.get_real_latency()}')

    # Fine-tuning Phase
    if args.finetune:
        therm_model.module.freeze_alpha()
        if args.finetune_scratch:
            therm_model.module.load_state_dict(init_state_dict, strict=False)

        # Eventually load previous checkpoint
        if args.resume_ckp_finetune is not None:
            therm_model, optimizer_state_dict, last_epoch = \
                resume_ckp(args.resume_ckp_finetune, therm_model, test_dl, rank)
        else:
            optimizer_state_dict = None
            last_epoch = -1

        finetune_loop(therm_model, N_EPOCHS,
                      CHECKPOINT_DIR / 'finetune',
                      train_dl, val_dl, test_dl, rank,
                      lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                      WARMP=WARMP,
                      ft_again=args.finetune_again,
                      ft_scratch=args.finetune_scratch,
                      start_epoch=last_epoch,
                      optimizer_state_dict=optimizer_state_dict)

    destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NAS Search and Fine-Tuning')
    parser.add_argument('--arch', type=str,
                        help=f'Arch name taken from {model_names}')
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
    parser.add_argument('--val-split', default=0.1, type=float,
                        help='Percentage of training data to be used as validation set')
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--init-strategy', type=str, help='One of {"1st", "2nd", "half"}')
    parser.add_argument('--strength', type=float, default=0.0e+0, help='Regularization Strength')
    parser.add_argument('--resume-ckp-warmup', type=str, default=None,
                        help='Resume loading specified model checkpoint')
    parser.add_argument('--resume-ckp-search', type=str, default=None,
                        help='Resume loading specified model checkpoint')
    parser.add_argument('--resume-ckp-finetune', type=str, default=None,
                        help='Resume loading specified model checkpoint')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='Dry run with test-set using passed pretrained model')
    parser.add_argument('--warmup', default=False, action='store_true',
                        help='Whether to perform warmup')
    parser.add_argument('--warmup-strategy', type=str, help='One of {"std", "coarse", "fine"}')
    parser.add_argument('--cost', type=str, default='naive', help='One of {"naive", "darkside"}')
    parser.add_argument('--search', default=False, action='store_true',
                        help='Whether to perform search')
    parser.add_argument('--finetune', default=False, action='store_true',
                        help='Whether to perform finetune')
    parser.add_argument('--finetune-scratch', default=False, action='store_true',
                        help='Whether to perform finetune from scratch')
    parser.add_argument('--finetune-again', default=False, action='store_true',
                        help='Whether to perform again finetune')
    parser.add_argument('--seed', type=int, default=42, help='Random Seed')
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

    if args.timestamp is None:
        args.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Set up logging in the main process
    logging.basicConfig(filename=args.checkpoint_dir / 'log.txt',
                        level=logging.INFO, format='%(asctime)s [%(process)d] %(message)s')

    world_size = args.world_size
    port = get_free_port()
    mp.spawn(main, args=(world_size, port, args), nprocs=world_size)
