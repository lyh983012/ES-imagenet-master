import time

import numpy as np
import os
import pandas as pd
import torch
import torch.utils.data as data


class ESImagenet_Dataset(data.Dataset):
    def __init__(self, mode, data_set_path='/data/dvsimagenet/'):
        super().__init__()
        self.mode = mode
        self.filenames = []
        self.trainpath = data_set_path + 'train'
        self.testpath = data_set_path + 'val'
        self.traininfotxt = data_set_path + 'trainlabel.csv'
        self.testinfotxt = data_set_path + 'vallabel.csv'
        self.formats = '.npz'
        if mode == 'train':
            data = pd.read_csv(self.traininfotxt, header=None)
            self.filenames = data.values[:, 0]
            self.classnums = data.values[:, 1]
            self.path = self.trainpath

        else:
            data = pd.read_csv(self.testinfotxt, header=None)
            self.filenames = data.values[:, 0]
            self.classnums = data.values[:, 1]
            self.path = self.testpath

    def __getitem__(self, index):

        return np.load(os.path.join(self.path, self.filenames[index]))['arr_0'], np.array([self.classnums[index]])

    def __len__(self):
        return len(self.filenames)


def test():
    start = time.time()

    batch_size = 160
    batch_size_test = batch_size
    drop_last = False
    pip_memory = False
    num_work = 0
    # Data set
    train_dataset = ESImagenet_Dataset(
        data_set_path='/home/huanhuan.gao/data/ES_ImageNet/ES-imagenet-0.18-convert/',
        mode='train',
    )

    test_dataset = ESImagenet_Dataset(
        data_set_path='/home/huanhuan.gao/data/ES_ImageNet/ES-imagenet-0.18-convert/',
        mode='val',
    )
    # Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_work,
        pin_memory=pip_memory)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size_test,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_work,
        pin_memory=pip_memory)
    print(time.time() - start)
    start = time.time()

    for batch_idx, (input, labels) in enumerate(train_loader):
        print(batch_idx)

    print(time.time() - start)
    # start = time.time()
    # for batch_idx, (input, labels) in enumerate(test_loader):
    #     print(batch_idx)
    #
    # print(time.time() - start)


if __name__ == '__main__':
    test()
