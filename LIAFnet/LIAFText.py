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
            self.model_name = 'TextLIAFFC'
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
        self.learning_rate = 1e-4                                            
        self.actFun = torch.selu
        self.decay = 0.3
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.dropOut= 0
        self.useBatchNorm = False
        self.timeWindows = 32
        self.useThreshFiring= True
        self.Qbit = 0
        self.device =  LIAF.device  
        self.hidden_size = 128                                          # lstm隐藏层
        self.cfgFc =[11]
        self.cfgLSTM = [self.embed,512,256,64]
        self._data_sparse=True
        self.onlyLast= False
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                      # 学习率
        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数

class Model(nn.Module):

    #standard basic complex network built using LIAFcell
    #update: 2020-03-01
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
        self.timeWindows = config.timeWindows
        self.useThreshFiring=config.useThreshFiring
        self.Qbit = config.Qbit
        self.onlyLast=config.onlyLast
        self._data_sparse=config._data_sparse
        self.network=nn.Sequential()
        self.cfgFc=config.cfgFc
        for dice in range(len(self.cfgFc)-1):
            self.network.add_module('liaffc'+str(dice),
                                    LIAF.LIAFCell(self.cfgFc[dice],
                                        self.cfgFc[dice+1],
                                        actFun=self.actFun,
                                        decay=self.decay,
                                        dropOut=self.dropOut,
                                        useBatchNorm=self.useBatchNorm,
                                        Qbit=self.Qbit)
                                    )
        print(self.network)

    def forward(self,data):
        torch.cuda.empty_cache()
        data, _ = data
        data = self.embedding(data)
        layer_num = 0

        for step in range(self.timeWindows):
            frame_t = data[:, step , :]
            output = frame_t
            for layer in self.network:
                output = layer(output)
                if step == 0:
                    outputsum = torch.zeros(output.size(), device=device)
            outputsum += output
        outputsum = outputsum / self.timeWindows
        if self.onlyLast:
            outputsum = output
        return outputsum
