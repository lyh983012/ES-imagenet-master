# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import LIAF

device= LIAF.device  

class Config(object):

    def __init__(self, path, dataset, embedding):
        if path is not None:
            self.model_name = 'TextLIAFLSTM'
            self.train_path = path + dataset + '/data/train.txt'                                # 训练集
            self.dev_path = path +dataset + '/data/dev.txt'                                    # 验证集
            self.test_path = path +dataset + '/data/test.txt'                                  # 测试集
            self.class_list = [x.strip() for x in open(
                path +dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
            self.vocab_path = path +dataset + '/data/vocab.pkl'                                # 词表
            self.save_path = path +dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
            self.log_path = path +dataset + '/log/' + self.model_name
            self.embedding_pretrained = torch.tensor(
                np.load(path +dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
                if embedding != 'random' else None                                       # 预训练词向量

        self.actFun = torch.selu
        self.decay = 0.3
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.dropOut= 0
        self.useBatchNorm = True
        self.timeWindows = 32
        self.useThreshFiring= True
        self._data_sparse=False
        self.Qbit = 0
        self.device =  LIAF.device  
        self.hidden_size = 128                                          # lstm隐藏层
        self.cfgFc =[11]

        self.cfgLSTM = [self.embed,512,256]
        self._data_sparse=True

        self.onlyLast= False
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值

        self.num_epochs = 50                                            # epoch数

        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                     # 学习率
        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数
        self.thresh = 0.5
        self.bidirection = True      


class Model(nn.Module):

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
        self.thresh = config.thresh
        self.useBatchNorm = config.useBatchNorm
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
                                    LIAF.LIAFLSTMCell_test(self.cfgLSTM[dice],
                                    self.cfgLSTM[dice+1],
                                    actFun=self.actFun,
                                    decay=self.decay,
                                    dropOut=self.dropOut,
                                    useBatchNorm=self.useBatchNorm,
                                    Qbit=self.Qbit)
                                    )
   
            if self.bidirection:
                self.network.add_module('liaflstm_backward'+str(dice),
                                    LIAF.LIAFLSTMCell_test(self.cfgLSTM[dice],
                                    self.cfgLSTM[dice+1],
                                    actFun=self.actFun,
                                    decay=self.decay,
                                    dropOut=self.dropOut,
                                    useBatchNorm=self.useBatchNorm,
                                    Qbit=self.Qbit)
                                    )
                
        self.network.add_module('fc'+str(dice),
                        LIAF.LIAFCell(

                        self.cfgLSTM[-1]*numdim,

                        config.num_classes,
                        actFun = self.actFun,
                        decay = self.decay,
                        dropOut= self.dropOut,
                        useBatchNorm=self.useBatchNorm,
                        Qbit = self.Qbit))
        print(self.network,' Is Bidirection?', self.bidirection)

    def forward(self,data):
        torch.cuda.empty_cache()
        data, _ = data
        data = self.embedding(data).detach()
        data_ = data.clone().detach()

        for step in range(self.timeWindows):
            out = [ data[:, step , :] , data_[:, self.timeWindows-step-1, :]]
            num = 0
            if self.bidirection:   
                for layer in self.network:
                    if not isinstance(layer, LIAF.LIAFCell):
                        if num%2 == 0:  
                            out[0] = layer(out[0])
                            #print(output_forward.size())
                        else:
                            out[1] = layer(out[1])
                            #print(output_backward.size())
                        num +=1
                    else:
                        output = torch.cat(out,dim=1).to(device=device)
                        output = layer(output)
            else:
                for layer in self.network:
                    out[0] = layer(out[0])
                output = out[0]

            if step == 0:
                 outputsum = torch.zeros(output.size(), device=device)    
            outputsum += output
                
        if self.onlyLast:
            return output
        else:
            return outputsum / self.timeWindows

