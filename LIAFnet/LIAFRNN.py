
#accept: LSTM和RNN的LIAF化，支持双向
#test: 随机初始化原始膜电位
#data:2020-08-11
#author:linyh
#email: 532109881@qq.com
import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import LIAF

device=LIAF.device
thresh = LIAF.thresh
lens = LIAF.lens
decay =LIAF.decay
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
        self.learning_rate = 1e-1                                       # 学习率，最重要的参数，部分demo不是在这里设置
        self.device =  LIAF.device                                      # 不需修改
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
        self.decay = 0.3                                                # 时间常数
        self.Qbit=0                                                     # 是否使用多阈值函数（>2支持，Qbit的值实际上是阈值个数）
        
        '''cfg for cnn'''
        self.dataSize = 32                                              # 输入图像的大小，用于自动计算特征图大小
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

# newdemo in example_lstmclass with model/TextLIAFRNN.py
class LIAFRNN(nn.Module):

    #standard basic complex network built using LIAFcell
    #update: 2020-04-04
    #author: Linyh

    def __init__(self,config):
        super().__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        self.actFun = config.actFun
        self.decay = config.decay
        self.dropOut= config.dropOut
        self.useBatchNorm = config.useBatchNorm
        self.useLayerNorm = config.useLayerNorm
        self.timeWindows = config.timeWindows
        self.useThreshFiring=config.useThreshFiring
        self.Qbit = config.Qbit
        self.onlyLast=config.onlyLast
        self._data_sparse=config._data_sparse
        self.network=nn.Sequential()
        self.cfgLSTM=config.cfgLSTM
        self.bidirection=config.bidirection
        if self.bidirection:
            numdim=2
        else :
            numdim=1 

        for dice in range(len(self.cfgLSTM)-1):
            self.network.add_module('liaflstm_forward'+str(dice),
                                    LIAF.LIAFRCell(self.cfgLSTM[dice],
                                    self.cfgLSTM[dice+1],
                                    actFun=self.actFun,
                                    decay=self.decay,
                                    dropOut=self.dropOut,
                                    timeWindows=self.timeWindows,
                                    useBatchNorm=self.useBatchNorm,
                                    useLayerNorm=self.useLayerNorm,
                                    Qbit=self.Qbit)
                                    )
   
            if self.bidirection:
                self.network.add_module('liaflstm_backward'+str(dice),
                                    LIAF.LIAFRCell(self.cfgLSTM[dice],
                                    self.cfgLSTM[dice+1],
                                    actFun=self.actFun,
                                    decay=self.decay,
                                    dropOut=self.dropOut,
                                    timeWindows=self.timeWindows,
                                    useBatchNorm=self.useBatchNorm,
                                    useLayerNorm= self.useLayerNorm,
                                    Qbit=self.Qbit)
                                    )
                
        self.network.add_module('fc'+str(dice),
                        LIAF.LIAFCell(
                        self.cfgLSTM[-1]*numdim,
                        config.num_classes,
                        actFun = self.actFun,
                        decay = self.decay,
                        dropOut= self.dropOut,
                        timeWindows=self.timeWindows,
                        useBatchNorm=True,
                        Qbit = self.Qbit))
        print(self.network,' Is Bidirection?', self.bidirection)

    def forward(self,data):
        data, _ = data
        data = self.embedding(data).detach()
        index_list = [i for i in range(self.timeWindows-1,-1,-1)]
        data_ = data.clone().detach()[:,index_list,:]

        out = [ data, data_ ]
        num = 0
        if self.bidirection:   
            for layer in self.network:
                if not isinstance(layer, LIAF.LIAFCell):
                    out[num%2] = layer(out[num%2])
                    num +=1
                else:
                    output = torch.cat(out,dim=2).to(device=device)
                    output = layer(output)
        else:
            for layer in self.network:
                out[0] = layer(out[0])
            output = out[0]
        
        outputmean = output.mean(dim=1) 
        if self.onlyLast:
            return output[:,-1,:]
        else:
            return outputmean
# demo in example_lstmclass with model/TextLIAFLSTM.py
class LIAFLSTM(nn.Module):

    #standard basic complex network built using LIAFcell
    #update: 2020-04-01
    #author: Linyh

    def __init__(self,config):
        super().__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        self.actFun = config.actFun
        self.decay = config.decay
        self.dropOut= config.dropOut
        self.useBatchNorm = config.useBatchNorm
        self.useLayerNorm = config.useLayerNorm
        self.timeWindows = config.timeWindows
        self.useThreshFiring=config.useThreshFiring
        self.Qbit = config.Qbit
        self.onlyLast=config.onlyLast
        self._data_sparse=config._data_sparse
        self.network=nn.Sequential()
        self.cfgLSTM=config.cfgLSTM
        self.bidirection=config.bidirection
        if self.bidirection:
            numdim=2
        else :
            numdim=1 

        for dice in range(len(self.cfgLSTM)-1):
            self.network.add_module('liaflstm_forward'+str(dice),
                                    LIAF.LIAFLSTMCell(self.cfgLSTM[dice],
                                    self.cfgLSTM[dice+1],
                                    actFun=self.actFun,
                                    spikeActFun = torch.sigmoid,
                                    decay=self.decay,
                                    dropOut=self.dropOut,
                                    timeWindows = self.timeWindows,
                                    useBatchNorm=self.useBatchNorm,
                                    useLayerNorm= self.useLayerNorm,
                                    Qbit=self.Qbit)
                                    )
            if self.bidirection:
                self.network.add_module('liaflstm_backward'+str(dice),
                                    LIAF.LIAFLSTMCell(self.cfgLSTM[dice],
                                    self.cfgLSTM[dice+1],
                                    actFun=self.actFun,
                                    spikeActFun = torch.sigmoid,
                                    decay=self.decay,
                                    dropOut=self.dropOut,
                                    timeWindows = self.timeWindows,
                                    useBatchNorm=self.useBatchNorm,
                                    useLayerNorm= self.useLayerNorm,
                                    Qbit=self.Qbit)
                                    )
                
        self.network.add_module('fc'+str(dice),
                        LIAF.LIAFCell(
                        self.cfgLSTM[-1]*numdim,
                        config.num_classes,
                        actFun = self.actFun,
                        decay = self.decay,
                        dropOut= self.dropOut,
                        timeWindows = self.timeWindows,
                        useBatchNorm=self.useBatchNorm,
                        Qbit = self.Qbit))
        print(self.network,' Is Bidirection?', self.bidirection)

    def forward(self,data):
        data, _ = data
        data = self.embedding(data).detach()
        index_list = [i for i in range(self.timeWindows-1,-1,-1)]
        data_ = data.clone().detach()[:,index_list,:]

        out = [ data, data_ ]
        num = 0
        if self.bidirection:   
            for layer in self.network:
                if not isinstance(layer, LIAF.LIAFCell):
                    out[num%2] = layer(out[num%2])
                    num +=1
                else:
                    output = torch.cat(out,dim=2).to(device=device)
                    output = layer(output)
        else:
            for layer in self.network:
                out[0] = layer(out[0])
            output = out[0]
        
        outputmean = output.mean(dim=1) 
        if self.onlyLast:
            return output[:,-1,:]+1e-10
        else:
            return outputmean+1e-10
# newdemo_LIAF_Bi
class BiLIAFFC(nn.Module):

    #standard basic complex network built using LIAFcell
    #update: 2020-04-01
    #author: Linyh

    def __init__(self,config):
        super().__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        self.actFun = config.actFun
        self.decay = config.decay
        self.dropOut= config.dropOut
        self.useBatchNorm = config.useBatchNorm
        self.useLayerNorm = config.useLayerNorm
        self.timeWindows = config.timeWindows
        self.useThreshFiring=config.useThreshFiring
        self.Qbit = config.Qbit
        self.onlyLast=config.onlyLast
        self._data_sparse=config._data_sparse
        self.network=nn.Sequential()
        self.cfgLSTM=config.cfgLSTM
        self.bidirection=config.bidirection
        if self.bidirection:
            numdim=2
        else :
            numdim=1 

        for dice in range(len(self.cfgLSTM)-1):
                self.network.add_module('liaffc_forward'+str(dice),
                                    LIAF.LIAFCell(self.cfgLSTM[dice],
                                    self.cfgLSTM[dice+1],
                                    actFun=self.actFun,
                                    decay=self.decay,
                                    dropOut=self.dropOut,
                                    timeWindows =self.timeWindows,
                                    useBatchNorm=self.useBatchNorm,
                                    useLayerNorm= self.useLayerNorm,
                                    Qbit=self.Qbit)
                                    )
                if self.bidirection:
                    self.network.add_module('liaffc_backward'+str(dice),
                                    LIAF.LIAFCell(self.cfgLSTM[dice],
                                    self.cfgLSTM[dice+1],
                                    actFun=self.actFun,
                                    decay=self.decay,
                                    dropOut=self.dropOut,
                                    timeWindows =self.timeWindows,
                                    useBatchNorm=self.useBatchNorm,
                                    useLayerNorm= self.useLayerNorm,
                                    Qbit=self.Qbit)
                                    )
                
        self.network.add_module('fc'+str(dice),
                        LIAF.LIAFCell(
                        self.cfgLSTM[-1]*numdim,
                        config.num_classes,
                        actFun = self.actFun,
                        decay = self.decay,
                        dropOut= self.dropOut,
                        timeWindows =self.timeWindows,
                        useBatchNorm=self.useBatchNorm,
                        useLayerNorm= self.useLayerNorm,
                        Qbit = self.Qbit))
        print(self.network,' Is Bidirection?', self.bidirection)

    def forward(self,data):
        data, _ = data
        data = self.embedding(data).detach()
        index_list = [i for i in range(self.timeWindows-1,-1,-1)]
        data_ = data.clone().detach()[:,index_list,:]

        out = [ data, data_ ]
        num = 0
        if self.bidirection:   
            for layer in self.network:
                if num<4:
                    out[num%2] = layer(out[num%2])
                    num +=1
                else:
                    output = torch.cat(out,dim=2).to(device=device)
                    output = layer(output)
        else:
            for layer in self.network:
                out[0] = layer(out[0])
            output = out[0]
        
        outputmean = output.mean(dim=1) 
        if self.onlyLast:
            return output[:,-1,:]
        else:
            return outputmean
