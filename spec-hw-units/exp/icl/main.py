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
import pathlib

from torchinfo import summary
import torch

import pytorch_benchmarks.image_classification as icl
from pytorch_benchmarks.utils import seed_all, CheckPoint, EarlyStopping

from odimo.method import ThermometricNet

from exp.common import models
from exp.common.utils import evaluate, train_one_epoch

# Simply parse all models' names contained in model file
model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))


def warmup_loop(model, epochs, checkpoint_dir, train_dl, val_dl, test_dl, device):
    criterion = icl.get_default_criterion()
    # optimizer = icl.get_default_optimizer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(),
    #                              lr=5e-2, momentum=0.9, weight_decay=5e-4)
    scheduler = icl.get_default_scheduler(optimizer)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    warmup_checkpoint = CheckPoint(checkpoint_dir, model, optimizer, 'max')
    skip_warmup = True
    if (checkpoint_dir / 'warmup.ckp').exists():
        warmup_checkpoint.load(checkpoint_dir / 'warmup.ckp')
        print("Skipping warmup")
    else:
        skip_warmup = False
        print("Running warmup")

    if not skip_warmup:
        for epoch in range(epochs):
            metrics = train_one_epoch(
                epoch, False, model, criterion, optimizer, train_dl, val_dl, test_dl, device)
            scheduler.step()
            warmup_checkpoint(epoch, metrics['val_acc'])
        warmup_checkpoint.load_best()
        warmup_checkpoint.save(checkpoint_dir / 'warmup.ckp')
    val_metrics = evaluate(False, model, criterion, val_dl, device)
    test_metrics = evaluate(False, model, criterion, test_dl, device)
    print("Warmup Best Val Set Loss:", val_metrics['loss'])
    print("Warmup Best Val Set Accuracy:", val_metrics['acc'])
    print("Warmup Test Set Loss @ Best on Val:", test_metrics['loss'])
    print("Warmup Test Set Accuracy @ Best on Val:", test_metrics['acc'])


def search_loop(model, epochs, strength, strength_increment, checkpoint_dir,
                train_dl, val_dl, test_dl, device):
    criterion = icl.get_default_criterion()
    param_dicts = [
        {'params': model.nas_parameters(), 'weight_decay': 0},
        {'params': model.net_parameters()}]
    optimizer = torch.optim.Adam(param_dicts, lr=0.001, weight_decay=1e-4)
    # optimizer = optim.SGD(param_dicts,
    #                       lr=5e-2, momentum=0.9, weight_decay=5e-4)
    scheduler = icl.get_default_scheduler(optimizer)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # Set EarlyStop with a patience of 50 epochs and CheckPoint
    earlystop = EarlyStopping(patience=50, mode='max')
    search_checkpoint = CheckPoint(checkpoint_dir / 'search', model, optimizer, 'max')
    skip_search = True
    if (checkpoint_dir / 'search.ckp').exists():
        search_checkpoint.load(checkpoint_dir / 'search.ckp')
        print("Skipping search")
    else:
        skip_search = False
        print("Running search")

    if not skip_search:
        for epoch in range(epochs):
            new_strength = min(strength/100 + strength_increment * epoch, strength)
            metrics = train_one_epoch(
                epoch, True, model, criterion, optimizer, train_dl, val_dl, test_dl, device,
                reg_strength=new_strength)

            if epoch > 5:
                search_checkpoint(epoch, metrics['val_acc'])
                if earlystop(metrics['val_acc']):
                    print(f'Stopping at epoch {epoch}')
                    break

            scheduler.step()
            print("architectural summary:")
            print(model)
            print("model regularization:", model.get_latency())
        print("Load best model")
        search_checkpoint.load_best()
        search_checkpoint.save(checkpoint_dir / 'search.ckp')
    print("final architectural summary:")
    print(model)
    print(f"Final regularization: {model.get_latency()}")
    val_metrics = evaluate(True, model, criterion, val_dl, device)
    test_metrics = evaluate(True, model, criterion, test_dl, device)
    print("Search Best Val Set Loss:", val_metrics['loss'])
    print("Search Best Val Set Accuracy:", val_metrics['acc'])
    print("Search Test Set Loss @ Best on Val:", test_metrics['loss'])
    print("Search Test Set Accuracy @ Best on Val:", test_metrics['acc'])


def finetune_loop(model, epochs, checkpoint_dir, train_dl,
                  val_dl, test_dl, device,
                  ft_again=False, ft_scratch=False):
    criterion = icl.get_default_criterion()
    optimizer = icl.get_default_optimizer(model)
    # optimizer = optim.SGD(model.parameters(),
    #                       lr=5e-2, momentum=0.9, weight_decay=5e-4)
    scheduler = icl.get_default_scheduler(optimizer)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # Set EarlyStop with a patience of 50 epochs and CheckPoint
    earlystop = EarlyStopping(patience=100, mode='max')
    finetune_checkpoint = CheckPoint(checkpoint_dir / 'finetune', model, optimizer, 'max')
    skip_finetune = True
    if (checkpoint_dir / 'finetune.ckp').exists():
        if not ft_scratch:
            finetune_checkpoint.load(checkpoint_dir / 'finetune.ckp')
        if ft_again:
            skip_finetune = False
            print("Running finetune again")
        else:
            print("Skipping finetune")
    else:
        skip_finetune = False
        print("Running finetune")

    if not skip_finetune:
        for epoch in range(epochs):
            metrics = train_one_epoch(
                epoch, False, model, criterion, optimizer, train_dl, val_dl, test_dl, device)

            if epoch > 5:
                finetune_checkpoint(epoch, metrics['val_acc'])
                if earlystop(metrics['val_acc']):
                    print(f'Stopping at epoch {epoch}')
                    break

            scheduler.step()
        finetune_checkpoint.load_best()
        finetune_checkpoint.save(checkpoint_dir / 'finetune.ckp')
    val_metrics = evaluate(False, model, criterion, val_dl, device)
    test_metrics = evaluate(False, model, criterion, test_dl, device)
    print("Finetune Best Val Set Loss:", val_metrics['loss'])
    print("Finetune Best Val Set Accuracy:", val_metrics['acc'])
    print("Finetune Test Set Loss @ Best on Val:", test_metrics['loss'])
    print("Finetune Test Set Accuracy @ Best on Val:", test_metrics['acc'])


def main(args):
    DATA_DIR = args.data_dir
    CHECKPOINT_DIR = pathlib.Path(args.checkpoint_dir)
    N_EPOCHS = args.epochs
    LAMBDA = torch.tensor(args.strength)

    # Check CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)

    # Ensure determinstic execution
    seed_all(seed=args.seed)

    # Get the Data
    data_dir = DATA_DIR
    datasets = icl.get_data(data_dir=data_dir, val_split=0.1)
    dataloaders = icl.build_dataloaders(datasets)
    train_dl, val_dl, test_dl = dataloaders
    input_shape = datasets[0][0][0].numpy().shape

    # Get and build the Model
    model_fn = models.__dict__[args.arch]
    model = model_fn(input_shape, 10)
    model = model.to(device)

    # Model Summary
    stats = summary(model, (1,) + input_shape, mode='eval')
    print(stats)

    # Eventually load pretrained model
    if args.pretrained_model is not None:
        state_dict = torch.load(args.pretrained_model)['model_state_dict']
        model.load_state_dict(state_dict)
        # Eval
        criterion = icl.get_default_criterion()
        pretrained_metrics = evaluate(False, model, criterion, test_dl, device)
        print("Pretrained Test Set Accuracy:", pretrained_metrics['acc'])

    # Warmup Phase
    if args.warmup:
        # Convert the model to ThermometricNet
        therm_model = ThermometricNet(model, input_shape=input_shape,
                                      cost=args.cost,
                                      init_strategy=args.init_strategy,
                                      warmup_strategy=args.warmup_strategy)
        therm_model = therm_model.to(device)
        if args.finetune_scratch:
            state_dict = therm_model.state_dict()
            init_state_dict = copy.deepcopy(state_dict)
            exclude_keys = ['alpha', 'ch_eff']
            for key in state_dict.keys():
                for ex_key in exclude_keys:
                    if ex_key in key:
                        init_state_dict.pop(key)

        warmup_dir = 'warmup_' + str(args.warmup_strategy)
        warmup_dir += '_init_' + str(args.init_strategy)
        print(f'Model Phase: {therm_model.phase}')
        if args.warmup_strategy != 'std':
            warmup_loop(therm_model, N_EPOCHS,
                        CHECKPOINT_DIR.parent.parent / warmup_dir,
                        train_dl, val_dl, test_dl, device)
        else:
            print('Warmup Phase I: Set combiner to always choose 2nd layer and train.')
            therm_model.set_init_strategy(strategy='2nd')
            warmup_dir_i = warmup_dir + '_phase_i'
            warmup_loop(therm_model, int(N_EPOCHS),
                        CHECKPOINT_DIR.parent.parent / warmup_dir_i,
                        train_dl, val_dl, test_dl, device)
            print('Warmup Phase II: Set combiner to always choose 1st layer and train.')
            therm_model.set_init_strategy(strategy='1st')
            warmup_dir_ii = warmup_dir + '_phase_ii'
            warmup_loop(therm_model, int(N_EPOCHS),
                        CHECKPOINT_DIR.parent.parent / warmup_dir_ii,
                        train_dl, val_dl, test_dl, device)

    # Search Phase
    if args.search:
        # Convert the model to ThermometricNet
        # therm_model = ThermometricNet(model, input_shape=input_shape)
        # therm_model = therm_model.to(device)
        # stats = summary(model, (1,) + input_shape, mode='eval')
        # print(stats)
        therm_model.set_init_strategy(strategy=args.init_strategy)
        therm_model.change_train_phase(phase='search', strategy=args.warmup_strategy)
        print(f'Model Phase: {therm_model.phase}')
        print(f'Initial Latency: {therm_model.get_real_latency()}')

        # Search Loop
        lambda_increment = (LAMBDA * 99/100) / int(N_EPOCHS/2)
        search_loop(therm_model, N_EPOCHS, LAMBDA, lambda_increment, CHECKPOINT_DIR, train_dl,
                    val_dl, test_dl, device)

        print(f'Final Latency: {therm_model.get_real_latency()}')

    # Fine-tuning Phase
    if args.finetune:
        # Convert pit model into pytorch model
        # exported_model = therm_model.arch_export()
        # exported_model = exported_model.to(device)
        # print(summary(exported_model, input_example, show_input=False, show_hierarchical=True))
        therm_model.freeze_alpha()
        if args.finetune_scratch:
            # for layer in therm_model.children():
            #     if hasattr(layer, 'reset_parameters'):
            #         layer.reset_parameters()
            therm_model.load_state_dict(init_state_dict, strict=False)
        finetune_loop(therm_model, N_EPOCHS, CHECKPOINT_DIR, train_dl,
                      val_dl, test_dl, device,
                      ft_again=args.finetune_again,
                      ft_scratch=args.finetune_scratch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NAS Search and Fine-Tuning')
    parser.add_argument('--arch', type=str, help=f'Arch name taken from {model_names}')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to Directory with Training Data')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Path to Directory where to save checkpoints')
    parser.add_argument('--epochs', type=int, help='Number of Training Epochs')
    parser.add_argument('--init-strategy', type=str, help='One of {"1st", "2nd", "half"}')
    parser.add_argument('--strength', type=float, default=0.0e+0, help='Regularization Strength')
    parser.add_argument('--pretrained-model', type=str, default=None,
                        help='Path to pretrained model')
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
    parser.add_argument('--seed', type=int, default=14, help='Random Seed')
    args = parser.parse_args()
    main(args)
