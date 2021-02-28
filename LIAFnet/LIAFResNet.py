#data:2021-01-07
#author:linyh
#email: 532109881@qq.com
#note:底层实现

import torch
import torch.nn as nn 
import torch.nn.functional as F
import os
import math
import util.thBN as thBN
from LIAF import *

#ResNet
class LIAFResNet(nn.Module):
    #standard basic Conv network built using LIAFcell
    #update: 2020-08-11
    #author: Linyh
    def _make_layer_50(self, block, inChannels, blocks, stride=1):
        layers = []
        ResBlock = block(cahnnel_now= self.cahnnel_now,
                                inChannels= inChannels,
                                actFun=self.actFun,
                                inputSize=self.dataSize,
                                useBatchNorm= self.useBatchNorm)
        self.dataSize=ResBlock.outputSize
        layers.append(ResBlock)

        self.cahnnel_now = inChannels * block.expansion

        for i in range(1, blocks):
            ResBlock = block(cahnnel_now= self.cahnnel_now,
                                inChannels= inChannels,
                                actFun=self.actFun,
                                inputSize=self.dataSize,
                                useBatchNorm= self.useBatchNorm)
            self.dataSize=ResBlock.outputSize
            layers.append(ResBlock)
        network = nn.Sequential(*layers)
        return network

    def _make_layer(self, block, inChannels, outChannels, blocks, stride=1):
        layers = []
        ResBlock = block(inChannels=inChannels,
                                outChannels=outChannels,
                                actFun=self.actFun,
                                inputSize=self.dataSize,
                                useBatchNorm= self.useBatchNorm)
        self.dataSize=ResBlock.outputSize
        layers.append(ResBlock)
        for i in range(1, blocks):
            ResBlock = block(inChannels=outChannels,
                                outChannels=outChannels,
                                actFun=self.actFun,
                                inputSize=self.dataSize,
                                useBatchNorm= self.useBatchNorm)
            self.dataSize=ResBlock.outputSize
            layers.append(ResBlock)
        network = nn.Sequential(*layers)
        return network

    def __init__(self, config):
        super().__init__()
        self.actFun = config.actFun
        self.dataSize = config.dataSize
        self.batchSize = None
        self.timeWindows = None 
        self.cfgRes = config.cfgRes
        self.cfgFc = config.cfgFc
        self.cfgCnn = config.cfgCnn
        self.block = config.block
        self.num_classes = config.num_classes
        self.useBatchNorm = config.useBatchNorm
        self.useThreshFiring = config.useThreshFiring
        self._data_sparse=config._data_sparse
        self.cahnnel_now = self.cfgCnn[1]

        self.conv1 = LIAFConvCell(inChannels=self.cfgCnn[0],
                                    outChannels=self.cfgCnn[1],
                                    kernelSize=[7,7],
                                    stride= 2,
                                    padding = 3,
                                    actFun=self.actFun,
                                    usePool= True,
                                    useBatchNorm= self.useBatchNorm,
                                    inputSize= self.dataSize,
                                    p_kernelSize = 3,
                                    p_method = 'max',
                                    p_padding = 0,
                                    p_stride = 2)
        self.dataSize = self.conv1.outputSize
        if self.block is LIAFResNeck:
            self.layer1 = self._make_layer_50(self.block, 64, self.cfgRes[0])
            self.layer2 = self._make_layer_50(self.block, 128, self.cfgRes[1], stride=2)
            self.layer3 = self._make_layer_50(self.block, 256, self.cfgRes[2], stride=2)
            self.layer4 = self._make_layer_50(self.block, 512, self.cfgRes[3], stride=2)   # different
        elif self.block is LIAFResBlock:
            self.layer1 = self._make_layer(self.block, 64, 64, self.cfgRes[0])
            self.layer2 = self._make_layer(self.block, 64, 128, self.cfgRes[1], stride=2)
            self.layer3 = self._make_layer(self.block, 128, 256, self.cfgRes[2], stride=2)
            self.layer4 = self._make_layer(self.block, 256, 512, self.cfgRes[3], stride=2)   # different
        else:
            print(self.block)
            print("unddefined/wrongly-defined the type of residual block")
        self.post_pooling_kenerl = list(self.dataSize)
        self.dataSize[0] = math.floor((self.dataSize[0] - self.post_pooling_kenerl[0]) / 2 + 1)#F.avg_pool2d(output,2)#downsampling
        self.dataSize[1] = math.floor((self.dataSize[1] - self.post_pooling_kenerl[1]) / 2 + 1)#F.avg_pool2d(output,2)#downsampling

        self.cfgFc_ = [512 *self.block.expansion * self.dataSize[0] * self.dataSize[1]]
        self.cfgFc_.extend(self.cfgFc)
        self.fc = nn.Linear(self.cfgFc_[0],self.cfgFc_[1])

    def _sparse(self,input):
        if self.useThreshFiring:
                tmp = torch.ones(input.shape, device=self.device).float()
                input = (input >= tmp).to(device=self.device)
        else:  
            input = input > torch.rand(input.size(), device=self.device)
        return input

    def forward(self,input):

        self.device  = self.fc.weight.device
        self.batchSize=input.size()[0]
        self.timeWindows=input.size()[2]
        if input.device != self.device:
            input = input.to(self.device)
        if self._data_sparse:
            self._sparse(input)
        #.......................................#

        output = self.conv1(input)

        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        temp = torch.zeros(self.batchSize,self.timeWindows,self.cfgFc_[0]).to(self.device)
        for time in range(self.timeWindows):
            pool = F.avg_pool2d(output[:,:,time,:,:],self.post_pooling_kenerl)
            temp[:,time,:] = pool.view(self.batchSize,-1)

        output = temp.view(self.batchSize,self.timeWindows,-1)
        output = self.fc(output.mean(dim=1).type(dtype))

        return output.float()
