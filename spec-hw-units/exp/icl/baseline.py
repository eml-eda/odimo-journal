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

from torchinfo import summary
import torch

import pytorch_benchmarks.image_classification as icl
from pytorch_benchmarks.utils import seed_all, CheckPoint, EarlyStopping


from exp.common import models
from exp.common.utils import evaluate, train_one_epoch

# Simply parse all models' names contained in model file
model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))


def train_loop(model, epochs, checkpoint_dir, train_dl,
               val_dl, test_dl, device,
               train_again=False):
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
    print("Training Best Val Set Loss:", val_metrics['loss'])
    print("Training Best Val Set Accuracy:", val_metrics['acc'])
    print("Training Test Set Loss @ Best on Val:", test_metrics['loss'])
    print("Training Test Set Accuracy @ Best on Val:", test_metrics['acc'])


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

    # Training Phase
    train_loop(model, N_EPOCHS, CHECKPOINT_DIR, train_dl,
               val_dl, test_dl, device)

    if args.train_again:
        train_loop(model, N_EPOCHS, CHECKPOINT_DIR, train_dl,
                   val_dl, test_dl, device,
                   train_again=args.train_again)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline Training')
    parser.add_argument('--arch', type=str, help=f'Arch name taken from {model_names}')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to Directory with Training Data')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Path to Directory where to save checkpoints')
    parser.add_argument('--epochs', type=int, help='Number of Training Epochs')
    parser.add_argument('--pretrained-model', type=str, default=None,
                        help='Path to pretrained model')
    parser.add_argument('--train-again', default=False, action='store_true',
                        help='Whether to perform again training')
    parser.add_argument('--seed', type=int, default=14, help='Random Seed')
    args = parser.parse_args()
    main(args)
