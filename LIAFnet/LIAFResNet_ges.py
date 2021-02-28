
#accept: LSTM和RNN的LIAF化，支持双向
#test: 随机初始化原始膜电位
#data:2020-08-11
#author:linyh
#email: 532109881@qq.com
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from LIAF import *

#########################################################
'''general_configs'''
#模型demo的初始化在config里统一完成，内含大量默认参数，请认真检查
#参数的修改方式可以在各个example中找到

class Config(object):

    def __init__(self, path=None, dataset=None, embedding=None):
        '''cfg for learning'''
        self.learning_rate = 1e-3                                       # 学习率，最重要的参数，部分demo不是在这里设置
        self.device = device                                      # 不需修改
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练，仅LSTM实现
        self.num_epochs = 75                                           # epoch数
        self.batch_size = 10                                           # mini-batch大小，部分demo不是在这里设置
        self.Qbit=0                                                     # 是否使用多阈值函数（>2支持，Qbit的值实际上是阈值个数）
        '''cfg for net'''
        self.block = LIAFResBlock
        self.num_classes = 1000
        self.cfgCnn = [2,64]
        self.cfgRes = [2,2,2,2]
        self.cfgFc = [self.num_classes]
        self.timeWindows = 60

        self.actFun= torch.nn.LeakyReLU(0.2, inplace=False) #nexttest:selu
        self.useBatchNorm = True
        self.useThreshFiring = True
        self._data_sparse= False
        self.padding= 0
        self.dataSize= [224,224]


