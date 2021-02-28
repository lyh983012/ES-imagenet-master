from __future__ import print_function
import torch

import os
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import scipy.io as sio
import h5py


def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=200):

    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    if epoch % lr_decay_epoch == 0 :
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * init_lr

    return optimizer

def assign_optimizer(model, lrs):

    rate = 1
    fc1_params = list(map(id, model.fc.parameters()))
    base_params = filter(lambda p: id(p) not in fc1_params  , model.parameters())

    optimizer = torch.optim.SGD([
        {'params': base_params},
        {'params': model.fc.parameters(), 'lr': lrs * rate},
          ]
        , lr=lrs,momentum=0.9)
    print('successfully reset lr')
    return optimizer

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

