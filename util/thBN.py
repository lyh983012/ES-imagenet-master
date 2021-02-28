from __future__ import division
##
#
#author:lyh
#
#
##
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from typing import Optional, Any
import LIAF

class BatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features,k=1, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        self.k=k
        super(BatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.type(LIAF.dtype)

    def forward(self, input,k=1):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2])
            # use biased var in train
            var = input.var([0, 2], unbiased=False) * self.k
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None]) / (torch.sqrt(var[None, :, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None] + self.bias[None, :, None]

        return input

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features,k=1, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        self.k=k
        super(BatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.type(LIAF.dtype)

    def forward(self, input,k=1):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)  * self.k
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input

class BatchNorm3d(nn.BatchNorm3d):
    def __init__(self, num_features,k=1,eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        self.k=k
        super(BatchNorm3d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.type(LIAF.dtype)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3 ,4])
            # use biased var in train
            var = input.var([0, 2, 3 ,4], unbiased=False) * self.k
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None, None]) / (torch.sqrt(var[None, :, None, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]

        return input

if __name__ == '__main__':
    my_bn = BatchNorm3d(3, affine=True)
    bn = nn.BatchNorm3d(3, affine=True)
    
    def compare_bn(bn1, bn2):
        err = False
        if not torch.allclose(bn1.running_mean, bn2.running_mean):
            print('Diff in running_mean: {} vs {}'.format(
                bn1.running_mean, bn2.running_mean))
            err = True

        if not torch.allclose(bn1.running_var, bn2.running_var):
            print('Diff in running_var: {} vs {}'.format(
                bn1.running_var, bn2.running_var))
            err = True

        if bn1.affine and bn2.affine:
            if not torch.allclose(bn1.weight, bn2.weight):
                print('Diff in weight: {} vs {}'.format(
                    bn1.weight, bn2.weight))
                err = True

            if not torch.allclose(bn1.bias, bn2.bias):
                print('Diff in bias: {} vs {}'.format(
                    bn1.bias, bn2.bias))
                err = True

        if not err:
            print('All parameters are equal!')
    
    criterion = nn.MSELoss()
    # Run train
    scale = torch.randint(1, 10, (1,)).float()
    bias = torch.randint(-10, 10, (1,)).float()
    y = torch.randn(10, 3, 100, 100,10) * scale + bias
    for _ in range(10):
        x = torch.randn(10, 3, 100, 100,10) * scale + bias
        my_bn.zero_grad()
        bn.zero_grad()
        out1 = my_bn(x)
        out2 = bn(x)
        compare_bn(my_bn, bn)
        torch.allclose(out1, out2)
        print('Max diff: ', (out1 - out2).abs().max())
        loss1 = criterion(out1,y)
        loss1.backward()
        loss2 = criterion(out2,y)
        loss2.backward()

        

    # Run eval
    my_bn.eval()
    bn.eval()
    for _ in range(10):
        scale = torch.randint(1, 10, (1,)).float()
        bias = torch.randint(-10, 10, (1,)).float()
        x = torch.randn(10, 3, 100, 100, 10) * scale + bias
        out1 = my_bn(x)
        out2 = bn(x)
        compare_bn(my_bn, bn)

        torch.allclose(out1, out2)
        print('Max diff: ', (out1 - out2).abs().max())