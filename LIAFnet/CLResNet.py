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

#训练要求
# 20835816
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)# if you are using multi-GPU.
torch.cuda.manual_seed(1)   
torch.backends.cudnn.deterministic = False   #保证每次结果一样
torch.backends.cudnn.benchmark = True 

#这部分超参数不是很重要，可以不用修改
dtype = torch.float
allow_print = False
using_td_batchnorm = False


class Config(object):

    def __init__(self, path=None, dataset=None, embedding=None):
        '''cfg for learning'''
        self.learning_rate = 1e-3                                      # 学习率，最重要的参数，部分demo不是在这里设置
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练，仅LSTM实现
        self.num_epochs =  25                                           # epoch数
        self.batch_size = 14                                         # mini-batch大小，部分demo不是在这里设置
        self.Qbit=0                                                     # 是否使用多阈值函数（>2支持，Qbit的值实际上是阈值个数）
        '''cfg for net'''
        self.block = CLResBlock
        self.num_classes = 1000
        self.cfgCnn = [2,32]
        self.cfgRes = [1,1,1,1]
        self.cfgFc = [self.num_classes]
        self.timeWindows = 8
        self.actFun= torch.nn.LeakyReLU(0.2, inplace=True) #nexttest:selu
        self.useBatchNorm = True
        self.useThreshFiring = True
        self._data_sparse= False
        self.padding= 0
        self.dataSize= [224,224]

class ConvLSTMCell(nn.Module):

    def __init__(self,
                 inChannels,
                 outChannels,
                 kernelSize,
                 stride,
                 padding =0,
                 inputSize=(224,224),
                 ):
        super(ConvLSTMCell, self).__init__()

        self.height, self.width = inputSize
        
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.padding = kernelSize[0] // 2, kernelSize[1] // 2
        self.stride = stride

        self.kernel1 = nn.Conv2d(in_channels = self.inChannels,
                              out_channels = 4 * self.outChannels,
                              kernel_size = self.kernelSize,
                              padding = self.padding,
                              stride = self.stride)

        self.kernel2 = nn.Conv2d(in_channels = self.outChannels,
                              out_channels = 4 * self.outChannels,
                              kernel_size = 3,
                              padding = 1,
                              stride = 1)

        self.CSize = list(inputSize)
        self.CSize[0] = math.floor((inputSize[0] + 2 * self.padding[0]
            - kernelSize[0]) / self.stride + 1)
        self.CSize[1] = math.floor((inputSize[1] + 2 * self.padding[1] 
            - kernelSize[1]) / self.stride + 1)
        self.outputSize = list(self.CSize)

        self.oheight, self.owidth = self.outputSize


    def forward(self, input, cur_state = 0):
        # [b,c,w,h]
        timeWindows = input.size(2)
        batchSize = input.size(0)
        self.device = self.kernel1.weight.device
        if input.device != self.device:
            input = input.to(self.device) 
        C = torch.zeros((batchSize, self.outChannels, self.oheight, self.owidth),device=self.device)
        H = torch.zeros((batchSize, self.outChannels, self.oheight, self.owidth),device=self.device)
        output = torch.zeros((batchSize, self.outChannels, timeWindows, self.oheight, self.owidth),device=self.device)
        for t in range(timeWindows):
            input_t = input[:,:,t,:,:]
            input_conv = self.kernel1(input_t)
            hidden_conv = self.kernel2(H)
            cc_i, cc_f, cc_o, cc_g = torch.split(input_conv+hidden_conv, self.outChannels, dim=1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)
            C = f * C + i * g
            H = o * torch.tanh(C)
            output[:,:,t,:,:] = H
        return output

#ResBlock for ResNet18/34
class CLResBlock(nn.Module):
    # standard LIAF cell based on LIFcell
    # update: 2020-08-10
    # author: Linyh
    expansion = 1
    #简介: 基于LIAF-CNN的残差块
    def __init__(self,
                 inChannels,
                 outChannels,
                 actFun = 0,
                 kernelSize=(3,3),
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 useBatchNorm = True,
                 inputSize=(224,224),
                 name = 'convlstmresblock'
                 ):

        super().__init__()
        self.padding=padding
        self.useBatchNorm = useBatchNorm
        self.kernelSize=kernelSize
        self.timeWindows= None
        self.downSample=False
        self.shortcut = None
        self.actFun = actFun
        self.allow_print = True

        if inChannels!=outChannels:
            #判断残差类型——>输入输出是否具有相同维度
            stride = 2
            self.downSample = True
            self.shortcut = nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=2)
        else:
            stride = 1
        self.cv1 = ConvLSTMCell(inChannels=inChannels,
                                    outChannels=outChannels,
                                    kernelSize=kernelSize,
                                    stride= stride,
                                    inputSize = inputSize)
        self.cv2 = ConvLSTMCell(inChannels=outChannels,
                                    outChannels = outChannels,
                                    kernelSize = kernelSize,
                                    stride = 1,
                                    inputSize = self.cv1.outputSize)
        self.outputSize = self.cv2.outputSize

        self.bn1 = nn.BatchNorm3d(outChannels)
        self.bn2 = nn.BatchNorm3d(outChannels)
        self.shortcut_norm = nn.BatchNorm3d(outChannels)
        if self.allow_print:
            print('the output feature map is'+str(self.outputSize))
            
    def forward(self,input):
        self.timeWindows = input.size(2)
        self.batchSize = input.size(0)

        shortcut_output = input
        cv1_output = self.cv1(input)
        cv1_output = self.bn1(cv1_output)
        cv2_output = self.cv2(cv1_output)
        cv2_output = self.bn2(cv2_output)
        shortcut_output = torch.zeros(cv2_output.size(),device=cv2_output.device,dtype=dtype)
        if self.downSample:
            for time in range(self.timeWindows):
                shortcut_output[:,:,time,:,:] = self.shortcut(input[:,:,time,:,:])
        shortcut_output = self.shortcut_norm(shortcut_output)
        output = self.actFun(cv2_output+shortcut_output)
        return output

#ResNet
class CLResNet(nn.Module):
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
                                inputSize=self.dataSize,
                                actFun = self.actFun)
        self.dataSize=ResBlock.outputSize
        layers.append(ResBlock)
        for i in range(1, blocks):
            ResBlock = block(inChannels=outChannels,
                                outChannels=outChannels,
                                inputSize=self.dataSize,
                                actFun = self.actFun)
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
        self.cahnnel_now = self.cfgCnn[1]

        self.conv1 = ConvLSTMCell(inChannels=self.cfgCnn[0],
                                    outChannels=self.cfgCnn[1],
                                    kernelSize=[7,7],
                                    stride=2,
                                    padding=3,
                                    inputSize=self.dataSize)
        self.dataSize = self.conv1.outputSize

        self.p_kernelSize = 3
        self.p_padding = 0
        self.p_stride = 2
        self.dataSize[0] = math.floor((self.dataSize[0]+ 2 * self.p_padding 
            - self.p_kernelSize) / self.p_stride + 1)
        self.dataSize[1] = math.floor((self.dataSize[1]+ 2 * self.p_padding 
            - self.p_kernelSize) / self.p_stride + 1)

        self.cv1h,self.cv1w = self.dataSize
        self.cv1c = self.cfgCnn[1]

        if self.block is None:
            self.layer1 = self._make_layer_50(self.block, 64, self.cfgRes[0])
            self.layer2 = self._make_layer_50(self.block, 128, self.cfgRes[1], stride=2)
            self.layer3 = self._make_layer_50(self.block, 256, self.cfgRes[2], stride=2)
            #self.layer4 = self._make_layer_50(self.block, 512, self.cfgRes[3], stride=2)   # different
        elif self.block is CLResBlock:
            self.layer1 = self._make_layer(self.block, 32, 64, self.cfgRes[0])
            self.layer2 = self._make_layer(self.block, 64, 128, self.cfgRes[1], stride=2)
            self.layer3 = self._make_layer(self.block, 128, 256, self.cfgRes[2], stride=2)
            #self.layer4 = self._make_layer(self.block, 256, 256, self.cfgRes[3], stride=2)   # different

        self.post_pooling_kenerl = list(self.dataSize)
        self.dataSize[0] = math.floor((self.dataSize[0] - self.post_pooling_kenerl[0]) / 2 + 1)#F.avg_pool2d(output,2)#downsampling
        self.dataSize[1] = math.floor((self.dataSize[1] - self.post_pooling_kenerl[1]) / 2 + 1)#F.avg_pool2d(output,2)#downsampling
        self.avgp = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.cfgFc_ = [256 *self.block.expansion]
        self.cfgFc_.extend(self.cfgFc)
        self.fc = nn.Linear(self.cfgFc_[0],self.cfgFc_[1])

    def forward(self,input):

        self.device  = self.fc.weight.device
        self.batchSize=input.size()[0]
        self.timeWindows=input.size()[2]
        if input.device != self.device:
            input = input.to(self.device)
        #.......................................#

        output = self.conv1(input)

        output_pool = torch.zeros(self.batchSize,self.cv1c, self.timeWindows, 
            self.cv1h, self.cv1w,device=self.device,dtype=dtype)
        for time in range(self.timeWindows):
            output_pool[:,:,time,:,:] = F.max_pool2d(output[:,:,time,:,:], kernel_size=self.p_kernelSize,
                    stride = self.p_stride,padding=self.p_padding)

        output = output_pool
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        #output = self.layer4(output)

        temp = torch.zeros(self.batchSize,self.timeWindows,self.cfgFc_[0]).to(self.device)
        for time in range(self.timeWindows):
            pool = self.avgp(output[:,:,time,:,:])
            temp[:,time,:] = pool.view(self.batchSize,-1)

        output = temp.view(self.batchSize,self.timeWindows,-1)
        output = self.fc(output.mean(dim=1).type(dtype))
        return output.float()
