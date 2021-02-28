# Author: lyh 
# Date  : 2020-09-19
# 使用了分布式学习的ImageNet训练代码
# 使用以下命令直接执行
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m torch.distributed.launch --nproc_per_node=7 example_ES_res34.py
from __future__ import print_function
import sys
sys.path.append("..")
from util.util import lr_scheduler
from datasets.es_imagenet import ESImagenet_Dataset
import LIAF
from LIAFnet.LIAFResNet import *

import torch.distributed as dist 
import torch.nn as nn
import argparse, pickle, torch, time, os,sys
from importlib import import_module
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

##################### Step1. Env Preparation #####################

writer = None #仅在master进程上输出
master = False 
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
save_folder = 'ES_exp_34_ES'

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank',type = int,default=0)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group('nccl',init_method='env://')
local_rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(dist.get_rank(),' is ready')

if local_rank == 0 :
    writer = SummaryWriter('runs/'+save_folder)
    master = True
    print('start recording')

##################### Step2. load in dataset #####################

modules = import_module('LIAFnet.LIAFResNet_34')
config  = modules.Config()
workpath = os.path.abspath(os.getcwd())

num_epochs = config.num_epochs
batch_size = config.batch_size
timeWindows = config.timeWindows
epoch = 0
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
training_iter = 0
start_epoch = 0
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])

train_dataset = ESImagenet_Dataset(mode='train',data_set_path='/data/ES-imagenet-0.18/')
test_dataset = ESImagenet_Dataset(mode='test',data_set_path='/data/ES-imagenet-0.18/')

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
test_sampler  = torch.utils.data.distributed.DistributedSampler(test_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1,pin_memory=True,drop_last=True,sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1,pin_memory=True)

##################### Step3. establish module #####################

snn = LIAFResNet(config)
print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in snn.parameters())))

snn=torch.nn.SyncBatchNorm.convert_sync_batchnorm(snn)
snn.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(snn.parameters(),
            lr=config.learning_rate)
#防止进程冲突
with torch.cuda.device(local_rank):
    snn = DDP(snn,device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)

################step4. training and validation ################

def val(optimizer,snn,test_loader,test_dataset,batch_size,epoch):
    if master:
        print('===> evaluating models...')
    snn.eval()
    correct = 0
    total = 0
    predicted = 0
    with torch.no_grad():
        if master:
            for name,parameters in snn.module.named_parameters():
                writer.add_histogram(name, parameters.detach().cpu().numpy(),epoch)
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if ((batch_idx+1)<=len(test_dataset)//batch_size):
                optimizer.zero_grad()
                try:
                    targets=targets.view(batch_size)#tiny bug
                    outputs = snn(inputs.type(LIAF.dtype))
                    _ , predicted = outputs.cpu().max(1)
                    total += float(targets.size(0))
                    correct += float(predicted.eq(targets).sum())
                except:
                    print('sth. wrong')
                    print('val_error:',batch_idx, end='')
                    print('taret_size:',targets.size())
    acc = 100. * float(correct) / float(total)
    acc_record.append(acc)
    if master:
        writer.add_scalar('acc', acc,epoch)
    return optimizer


state = {
        'state': snn.module.state_dict(),
        'epoch': epoch,                   # 将epoch一并保存
        'best_acc' : best_acc,            # best test accuracy
        'acc_record': acc_record, 
        'loss_train_record': loss_train_record,
        'loss_test_record': loss_test_record ,
        'optimizer':optimizer.state_dict()
}

#val(optimizer,snn,test_loader,test_dataset,batch_size,epoch)

for epoch in range(num_epochs):
    #training
    running_loss = 0
    snn.train()
    start_time = time.time() 
    if master:
        print('===> training models...')
    correct = 0.0
    total = 0.0
    # 新增2：设置sampler的epoch，DistributedSampler需要这个来维持各个进程之间的相同随机数种子
    train_loader.sampler.set_epoch(epoch)
    for i, (images, labels) in enumerate(train_loader):
        if ((i+1)<=len(train_dataset)//batch_size):
            snn.zero_grad()
            optimizer.zero_grad()

            outputs = snn(images.type(LIAF.dtype)).cpu()
            labels = labels.view(batch_size)
            loss = criterion(outputs, labels)

            _ , predict = outputs.max(1)
            correct += predict.eq(labels).sum()
            total += float(predict.size(0))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (i+1)%10 == 0:
                if master : 
                    if not os.path.isdir(save_folder):
                        os.mkdir(save_folder)
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f \n'
                    %(epoch+start_epoch, num_epochs+start_epoch, i+1, len(train_dataset)//(world_size*batch_size),running_loss ))
                    print('Time elasped: %d \n'  %(time.time()-start_time))
                    writer.add_scalar('Loss_th', running_loss, training_iter)
                    train_acc =  correct / total
                    print('Epoch [%d/%d], Step [%d/%d], acc: %.5f \n'
                        %(epoch+start_epoch, num_epochs+start_epoch, i+1, len(train_dataset)//(world_size*batch_size), train_acc)) 
                    writer.add_scalar('train_acc', train_acc*100, training_iter)
                correct = 0.0
                total = 0.0
                running_loss = 0
        training_iter +=1 
    torch.cuda.empty_cache()
    #evaluation
    val(optimizer,snn,test_loader,test_dataset,batch_size,epoch)
    state = {
        'state': snn.module.state_dict(),
        'epoch': epoch+start_epoch,       # 将epoch一并保存
        'best_acc' : best_acc,            # best test accuracy
        'acc_record': acc_record, 
        'loss_train_record': loss_train_record,
        'loss_test_record': loss_test_record ,
        'optimizer':optimizer.state_dict()
    }
    if master:
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        print('===> Saving models...')
        torch.save(state, workpath+'/'+save_folder+'/'+str(epoch+start_epoch)+'modelsaved.t7')
        print('===> Saved')
    