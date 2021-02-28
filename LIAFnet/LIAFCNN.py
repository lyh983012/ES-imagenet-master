#author:linyh
#email: 532109881@qq.com
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import LIAF

print_model = False

#########################################################
'''general_configs'''
#模型demo的初始化在config里统一完成，内含大量默认参数，请认真检查
#参数的修改方式可以在各个example中找到

class Config(object):

    def __init__(self, path=None, dataset=None, embedding=None):

        #LSTMdemo的特殊参数
        if path is not None:
            self.model_name = 'TextCNN'
            self.train_path = path + dataset + '/data/train.txt'                       # 训练集
            self.dev_path = path +dataset + '/data/dev.txt'                            # 验证集
            self.test_path = path +dataset + '/data/test.txt'                          # 测试集
            self.class_list = [x.strip() for x in open(
                path +dataset + '/data/class.txt', encoding='utf-8').readlines()]      # 类别名单
            self.vocab_path = path +dataset + '/data/vocab.pkl'                        # 词表
            self.save_path = path +dataset + '/saved_dict/' + self.model_name + '.ckpt'# 模型训练结果
            self.log_path = path +dataset + '/log/' + self.model_name
            self.embedding_pretrained = torch.tensor(
                np.load(path +dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
                if embedding != 'random' else None                      # 预训练词向量
            self.embed = self.embedding_pretrained.size(1)\
                if self.embedding_pretrained is not None else 300       # 字向量维度, 若使用了预训练词向量，则维度统一
            self.num_classes = len(self.class_list)                     # 类别数
            self.cfgLSTM = [self.embed,512,256,128]
                                                                        # lstm/rnn各层结构参数，
                                                                        # 格式为[输入，隐层,...,输出]
            
        '''cfg for learning'''
        self.learning_rate = 1e-1                                       # 学习率，最重要的参数，部分demo不是在这里设置                                    # 不需修改
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练，仅LSTM实现
        self.num_epochs = 10                                            # epoch数，部分demo不是在这里设置
        self.batch_size = 96                                            # mini-batch大小，部分demo不是在这里设置

        '''cfg for nlp'''
     
        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.pad_size = 32                                              # 等长句子的长度
        self.bidirection = True                                         # 是否使用双向LSTM

        '''cfg for liaf'''                                     
        self.actFun = torch.selu                                        # 连续激活函数（LIAF的A）
        self.decay = 0.3                                                # 时间常数                                                  # 是否使用多阈值函数（>2支持，Qbit的值实际上是阈值个数）
        
        '''cfg for cnn'''

        self.dataSize = [32,32]                                              # 输入图像的大小，用于自动计算特征图大小
        self.padding = 0  
        '''cfg for net'''
        self.dropOut= 0                                                 # 是否使用dropoout
        self.useBatchNorm = False                                     # 是否使用batchnorm（于膜电位）
        self.useLayerNorm = False                                          # 在处理不定长序列数据时可以使用，仅能用于analog模式，否则破坏spike
        self.usetdBatchNorm= False  
        self.timeWindows = 60                                           # 序列长度
        self.useThreshFiring= False                                      # 是否使用阈值预处理输入（LSTM不使用）
        self.onlyLast= False                                            # 是否只用最后一个输出
        self.cfgCnn=[(2,64,3,1,1,False),(64,128,3,2,1,True),(128,128,3,2,1,True)]
                                                                        # CNN各层结构参数，
                                                                        # 格式为[inchannel,outchannel,卷积核大小，pooling核大小，是否pooling]
        self.cfgRes=[(64,64),(64,64),(64,128),(128,128)]
                                                                        # ResNet各层结构参数，
                                                                        # 格式为[inchannel,outchannel]
        self.cfgFc =[256,11]        
                                                                        # fc各层结构参数，
                                                                        # 格式为[输入，隐层,...,输出]

        self._data_sparse=False                                         # 是否做数据预处理，NLP推荐别用
        self.if_static_input = False

# newdemo in example_ges.py
class LIAFCNN(nn.Module):

    #standard basic Conv network built using LIAFcell
    #update: 2020-03-05
    #author: Linyh

    def __init__(self,config):
        #@param config.cfg_cnn: list like [(in_channels, out_channels, kernel_size, pkernel_size, stride ,usepooling),...]
        #@param config.cfg_fc: for FC_layer,  list of each layer's size with form:[(, Num),...] (see LAIFMLP)
        #@param config.actfun: handle of activation function
        #@param config.decay: time coefficient in the model
        #@param config.dropout: 0~1
        #@param config.batchNorm: enable/disable batch-norm layer
        #@param config.timeWindows: equvilant to the number of time step
        #@param config.datasize: the width/height of image, for automaitcally calculating the size of fclayer
        super().__init__()
        self.padding = config.padding
        self.actFun = config.actFun
        self.dropOut = config.dropOut
        self.useBatchNorm = config.useBatchNorm
        self.timeWindows = config.timeWindows
        self.cfgCnn = config.cfgCnn
        self.cfgFc = config.cfgFc
        self.nCnnLayers = len(config.cfgCnn)
        self.network = nn.Sequential()
        self.useThreshFiring = config.useThreshFiring
        self._data_sparse=config._data_sparse
        self.if_static_input = config.if_static_input
        self.onlyLast=config.onlyLast
        self.batchSize = None

        dataSize = config.dataSize

        for dice in range(self.nCnnLayers):
            inChannels, outChannels, kernelSize, p_kernelSize, stride ,usePooling= self.cfgCnn[dice]
            CNNlayer = LIAF.LIAFConvCell(inChannels=inChannels,
                                    outChannels=outChannels,
                                    kernelSize=[kernelSize,kernelSize],
                                    stride=stride,
                                    p_kernelSize=p_kernelSize,
                                    padding=self.padding,
                                    actFun=self.actFun,
                                    timeWindows = self.timeWindows,
                                    usePool=usePooling,
                                    dropOut=self.dropOut,
                                    inputSize=dataSize,
                                    useBatchNorm=self.useBatchNorm)
            dataSize = CNNlayer.outputSize #renew the fearture map size
            self.network.add_module('cnn'+str(dice),CNNlayer)
            
        self.cfgFc_ = [outChannels * dataSize[0] * dataSize[1]]
        self.cfgFc_.extend(self.cfgFc)
        self.nFcLayer=len(self.cfgFc_)-1#special
        for dice2 in range(self.nFcLayer):#DO NOT REUSE LIAFMLP!!!! BIGBUG
            self.network.add_module(str(dice2+dice+1),
                                        LIAF.LIAFCell(self.cfgFc_[dice2],
                                        self.cfgFc_[dice2+1],
                                        actFun=self.actFun,
                                        timeWindows = self.timeWindows,
                                        dropOut=self.dropOut,
                                        useBatchNorm=self.useBatchNorm))
        print(self.network)


    def forward(self,data):

        self.batchSize = data.size()[0]
        frames = data
        if self._data_sparse:
            if self.useThreshFiring:
                tmp = torch.ones(data.shape, device=device).float()
                frames = data >= tmp
            else:  
                frames = data > torch.rand(data.size(), device=device)
        output = frames
        
        for layer in self.network:
            if isinstance(layer, LIAF.LIAFCell):
                output = output.view(self.batchSize,self.timeWindows,-1)
            output = layer(output)

        outputmean = output.mean(dim=1)
        if self.onlyLast:
            outputmean = output[:,-1,:]
        return outputmean
