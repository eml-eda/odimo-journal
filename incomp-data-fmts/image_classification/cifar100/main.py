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
import logging
import pathlib
import shutil
import time

import torch
from torch.distributed import destroy_process_group, all_reduce, ReduceOp
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim
from torch.utils.data.distributed import DistributedSampler

from pytorch_benchmarks.utils import seed_all

import models
import cifar100_benchmark as c100
from utils import get_free_port, ddp_setup, DDPCheckPoint, WarmUpLR

# Uncomment to disable debug features and to go faster
# torch.autograd.set_detect_anomaly(False)
# torch.autograd.profiler.profile(False)
# torch.autograd.profiler.emit_nvtx(False)

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

best_acc1 = 0


def resume_ckp(ckp, model, test_dl, rank):
    # Load checkpoint, extract info and load weights
    ckp = torch.load(ckp, map_location='cpu')
    model_state_dict = ckp['model_state_dict']
    optimizer_state_dict = ckp['optimizer_state_dict_0']
    q_optimizer_state_dict = ckp['q_optimizer_state_dict_1']
    last_epoch = ckp['epoch']
    # ckp_test_accuracy = ckp['val']
    model.module.load_state_dict(model_state_dict)

    # Run eval with pretrained model
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    pretrained_acc = validate(test_dl, model, criterion, rank, args)
    logging.info(f"Pretrained Test Set Accuracy: {pretrained_acc}")

    # Add eval consistency check
    # msg = 'Mismatch in test set accuracy'
    # assert ckp_test_accuracy == pretrained_metrics['acc'], msg

    return model, optimizer_state_dict, q_optimizer_state_dict, last_epoch


def main(rank, world_size, port, args):
    # Extract arguments
    DATA_DIR = args.data
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

    best_acc1 = 0
    best_acc1_test = 0

    # Get the Data
    data_dir = DATA_DIR
    datasets = c100.get_data(data_dir=data_dir, val_split=VAL_SPLIT, seed=args.seed)
    dataloaders = c100.build_dataloaders(datasets, seed=args.seed, sampler_fn=DistributedSampler)
    train_dl, val_dl, test_dl = dataloaders

    # create model
    logging.info("=> creating model '{}'".format(args.arch))
    model_fn = models.__dict__[args.arch]
    model = model_fn(
        args.arch_cfg, input_size=32, fine_tune=args.fine_tune, std_head=False)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # Eventually load previous checkpoint
    if args.resume_ckp is not None:
        model, opt_sd, q_opt_sd, last_epoch = \
            resume_ckp(args.resume_ckp, model, test_dl, rank)
    else:
        opt_sd = None
        q_opt_sd = None
        last_epoch = 0

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # group model/quantization parameters
    params, q_params = [], []
    for name, param in model.named_parameters():
        if ('clip_val' in name) or ('scale_param' in name):
            q_params += [param]
        else:
            params += [param]

    optimizer = torch.optim.SGD(params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # TODO: try with `step_size=30`
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[60, 120, 160],
                                                     gamma=0.2)
    iter_per_epoch = len(train_dl)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * WARMP)
    if opt_sd is not None:
        optimizer.load_state_dict(opt_sd)

    if q_params:
        q_optimizer = torch.optim.SGD(q_params, args.lrq)
        q_scheduler = torch.optim.lr_scheduler.StepLR(q_optimizer, 7)
        if q_opt_sd is not None:
            q_optimizer.load_state_dict(q_opt_sd)
    else:
        q_optimizer = None
        q_scheduler = None

    if args.evaluate:
        validate(test_dl, model, criterion, rank, args)
        return

    best_epoch = args.start_epoch
    epoch_wout_improve = 0
    is_best = False
    checkpoint = DDPCheckPoint(CHECKPOINT_DIR, model, [optimizer, q_optimizer],
                               'max', save_best_only=True, save_last_epoch=True)
    earlystop_flag = torch.zeros(1).to(rank)

    for epoch in range(last_epoch, N_EPOCHS):
        if epoch > WARMP-1:
            scheduler.step()
        # train for one epoch
        train(train_dl, model, criterion, optimizer, warmup_scheduler,
              q_optimizer, epoch, rank, args)

        # evaluate on validation set
        if val_dl is not None:
            acc1 = validate(val_dl, model, criterion, epoch, rank, args)
        acc1_test = validate(test_dl, model, criterion, epoch, rank, args)

        if q_scheduler is not None:
            q_scheduler.step()

        # remember best acc@1 and save checkpoint
        if val_dl is not None:
            is_best = acc1 > best_acc1
            if is_best:
                best_epoch = epoch
                best_acc1 = acc1
                best_acc1_test = acc1_test
                epoch_wout_improve = 0
                logging.info(f'New best Acc_val: {best_acc1}')
                logging.info(f'New best Acc_test: {best_acc1_test}')
            else:
                epoch_wout_improve += 1
                logging.info(f'Epoch without improvement: {epoch_wout_improve}')

        if rank == 0:
            if val_dl is not None:
                checkpoint(epoch, acc1)
            else:
                checkpoint(epoch, acc1_test)

        # Early-Stop
        if epoch_wout_improve >= args.patience and rank == 0:
            earlystop_flag += 1
        all_reduce(earlystop_flag, op=ReduceOp.SUM)
        if earlystop_flag > 0:
            logging.info(f"GPU {rank}, early stopping at epoch: {epoch}")
            break

    best_acc1_val = best_acc1
    logging.info('Best Acc_val@1 {0} @ epoch {1}'.format(best_acc1_val, best_epoch))

    test_acc1 = best_acc1_test
    logging.info('Test Acc_val@1 {0} @ epoch {1}'.format(test_acc1, best_epoch))

    destroy_process_group()


def train(train_loader, model, criterion, optimizer, warmup_scheduler, q_optimizer,
          epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    curr_lr = optimizer.param_groups[0]['lr']
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}/{}]\t"
               "LR: {}\t".format(epoch, args.epochs, curr_lr))

    # switch to train mode
    model.train()

    train_loader.sampler.set_epoch(epoch)
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad(set_to_none=True)
        if q_optimizer is not None:
            q_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if q_optimizer is not None:
            q_optimizer.step()

        if epoch <= args.warm-1:
            warmup_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    val_loader.sampler.set_epoch(epoch)

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        logging.info(f' * Acc@1 {top1.avg:.6f} Acc@5 {top5.avg:.6f}')

    return top1.avg


def save_checkpoint(root, state, is_best, epoch, step_epoch, filename='checkpoint.pth.tar'):
    torch.save(state, root / filename)
    if is_best:
        shutil.copyfile(root / filename, root / 'model_best.pth.tar')
    if (epoch + 1) % step_epoch == 0:
        shutil.copyfile(root / filename, root / 'checkpoint_ep{}.pth.tar'.format(epoch + 1))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    initial_learning_rate = 0.001
    decay_per_epoch = 0.99
    lrate = initial_learning_rate * (decay_per_epoch ** epoch)
    for opt in optimizer.param_groups:
        opt['lr'] = lrate


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Imagenet Training')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet8',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Path to Directory where to save checkpoints')
    parser.add_argument('--timestamp', type=str, default=None,
                        help='Timestamp, if not provided will be current time')
    parser.add_argument('--resume-ckp', type=str, default=None,
                        help='Resume loading specified model checkpoint')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
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
    parser.add_argument('--step-epoch', default=50, type=int, metavar='N',
                        help='number of epochs to decay learning rate')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--val-split', default=0.1, type=float,
                        help='Percentage of training data to be used as validation set')
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lrq', '--learning-rate-q', default=1e-5, type=float,
                        metavar='LR', help='initial q learning rate', dest='lrq')
    parser.add_argument('-p', '--print-freq', default=20, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--arch-cfg', '--ac', default=None, type=str, metavar='PATH',
                        help='path to architecture configuration')
    # MR
    parser.add_argument('--complexity-decay', '--cd', default=0, type=float,
                        metavar='W', help='complexity decay (default: 0)')
    parser.add_argument('-ft', '--fine-tune', dest='fine_tune', action='store_true',
                        help='use pre-trained weights from search phase')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', default=False, action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--step-size', default=30, type=int,
                        help='Step size for LR scheduler')
    args = parser.parse_args()
    logging.info(args)

    # Set-up directories
    if args.checkpoint_dir is None:
        args.checkpoint_dir = pathlib.Path().cwd()
        args.checkpoint_dir = args.checkpoint_dir / str(args.arch)[5:]
        args.checkpoint_dir = args.checkpoint_dir / f'model_{args.complexity_decay:.1e}'
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    else:
        args.checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    # if args.timestamp is None:
    #     args.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.checkpoint_dir = args.checkpoint_dir / args.timestamp / 'ft'
    # Maybe create directories
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging in the main process
    logging.basicConfig(filename=args.checkpoint_dir / 'log.txt',
                        level=logging.INFO, format='%(asctime)s [%(process)d] %(message)s')

    world_size = args.world_size
    port = get_free_port()
    mp.spawn(main, args=(world_size, port, args), nprocs=world_size)
