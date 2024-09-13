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

import os
import pathlib
import socket

import torch
from torch.distributed import init_process_group


class DDPCheckPoint():
    """
    save/load a checkpoint based on a metric
    """
    def __init__(self, dir, net, optimizers,
                 mode='min', fmt='ck_{epoch:03d}.pt',
                 save_best_only=False, save_last_epoch=False):
        if mode not in ['min', 'max']:
            raise ValueError("Early-stopping mode not supported")
        self.dir = pathlib.Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.format = fmt
        self.save_best_only = save_best_only
        self.save_last_epoch = save_last_epoch
        self.net = net
        self.optimizers = optimizers
        self.val = None
        self.epoch = None
        self.best_path = None

    def __call__(self, epoch, val):
        val = float(val)
        if self.val is None:
            self.update_and_save(epoch, val)
        elif self.mode == 'min' and val < self.val:
            self.update_and_save(epoch, val)
        elif self.mode == 'max' and val > self.val:
            self.update_and_save(epoch, val)

        if self.save_last_epoch:
            self.save(self.dir / 'last_epoch.pt', val=val)

    def update_and_save(self, epoch, val):
        self.epoch = epoch
        self.val = val
        self.update_best_path()
        self.save()

    def update_best_path(self):
        if not self.save_best_only:
            self.best_path = self.dir / self.format.format(**self.__dict__)
        else:
            self.best_path = self.dir / 'best.pt'

    def save(self, path=None, val=None):
        if path is None:
            path = self.best_path
        if val is None:
            val = self.val
        opt_states = {}
        for idx, opt in enumerate(self.optimizers):
            opt_states[f'optimizer_state_dict_{idx}'] = opt.state_dict()
        torch.save({
                  'epoch': self.epoch,
                  'model_state_dict': self.net.module.state_dict(),
                  **opt_states,
                  'val': val,
                  }, path)

    def load_best(self):
        if self.best_path is None:
            raise FileNotFoundError("Best path not set!")
        self.load(self.best_path)

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.epoch = checkpoint['epoch']
        self.net.module.load_state_dict(checkpoint['model_state_dict'])
        for idx, opt in enumerate(self.optimizers):
            opt.load_state_dict(checkpoint[f'optimizer_state_dict_{idx}'])
        self.val = checkpoint['val']


def ddp_setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def get_free_port():
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port
