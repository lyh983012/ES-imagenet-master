# -*- coding: utf-8 -*-
#python3 example_mnist_fc.py
from __future__ import print_function
import sys
sys.path.append("..")
from util.util import lr_scheduler

from importlib import import_module
import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import os
import time
import LIAF

from tensorboardX import SummaryWriter

writer = SummaryWriter(comment='../runs/full_connect')

#TODO:input path of mnist dataset[data_path](or it will download one automatically);
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
names = 'spiking_model'
data_path = '/data/MNIST/'

num_classes = 10
batch_size  = 100
learning_rate = 1e-3
num_epochs = 50 # max epoch

train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

loss_train_list = []
loss_test_list = []
acc_train_list = []
acc_test_list = []
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
timeWindows=10

modules = import_module('LIAFnet.LIAFMLP')          
config = modules.Config()   
config.cfgFc =[28*28,800,num_classes]
config.decay = 0.5
config.dropOut= 0
config.timeWindows = timeWindows
config.actFun=torch.selu
config.useBatchNorm=True
config.useLayerNorm=False
config.useThreshFiring = False
snn = modules.LIAFMLP(config).to(device)
criterion = nn.CrossEntropyLoss()
######################################################################################
#note:
#CorssEntrophyLoss适用于分类问题（其为Max函数的连续近似）
#它的输入是output（每一类别的概率）和label（第几个类别）
######################################################################################
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    running_loss = 0
    start_time = time.time()

    #training
    snn.train(mode=True)
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(batch_size, 28*28)
        snn.zero_grad()
        optimizer.zero_grad()
        images = images.float().to(device)
        Is = images.size()

        input = torch.zeros(Is[0], timeWindows, Is[1])
        for j in range(timeWindows):
            input[:, j, :] = images
        #exp版本中输入的数据是图片序列，需要提前准备好
        outputs = snn(input)
        loss = criterion(outputs.cpu(), labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if (i+1)%100 == 0:
             print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                    %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size,running_loss ))
             writer.add_scalar('running_loss', running_loss, epoch)
             running_loss = 0
             print('Time elasped:', time.time()-start_time)
            
    correct = 0
    total = 0
    optimizer = lr_scheduler(optimizer, epoch, learning_rate)

    #evaluation
    snn.eval()
    with torch.no_grad():
        for name,parameters in snn.named_parameters():
            #print(name,':',parameters.size())
            writer.add_histogram(name, parameters.detach().cpu().numpy(), epoch)

        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            inputs = inputs.view(batch_size, 28 * 28)
            Is = inputs.size()
            input = torch.zeros(Is[0], timeWindows, Is[1])
            for j in range(timeWindows):
                input[:, j, :] = inputs
            optimizer.zero_grad()
            outputs = snn(input)
            loss = criterion(outputs.cpu(), labels)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            if batch_idx %100 ==0:
                acc = 100. * float(correct) / float(total)
                print(batch_idx, len(test_loader),' Acc: %.5f' % acc)
         #save & load
        print('Iters:', epoch,'\n\n\n')
        print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
        acc = 100. * float(correct) / float(total)
        if acc > best_acc:
            best_acc = acc
        print("best:",best_acc)
        writer.add_scalar('accuracy', acc, epoch)

   

    