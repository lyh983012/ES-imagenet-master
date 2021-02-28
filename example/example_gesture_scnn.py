# Please install dcll pkgs from below
# https://github.com/nmi-lab/dcll
# and then enjoy yourself.
# If there is any question please mail me.
# CUDA_VISIBLE_DEVICES=6 python3 example_gesture_scnn.py

import dcll
from dcll.load_dvsgestures_sparse import *
import argparse, pickle, torch, time, os
from importlib import import_module
import torch.nn as nn
import LIAF
import numpy as np
import pandas as pd
import util

from tensorboardX import SummaryWriter
writer = SummaryWriter(comment='runs/ges_scnn_flexest2')

LIAF.using_syn_batchnorm = False
torch.backends.cudnn.deterministic = True   #保证每次结果一样
device = LIAF.device
#TODO 1: enter ur path (for result)
#TODO 2: put your dataset(unziped) in dcll-maseter/data
#TODO 3: if any error plz mail me(dcll pakage has some bugs..)

resultPath = '/home/lyh/dvs_gestrue_model/result_LIAF.csv'
modelPath =  '/home/lyh/dvs_gestrue_model'

#################################
#Arg for network
batch_size = 32
thresh = LIAF.thresh
lens = LIAF.lens

learning_rate =1e-2
time_window = 60


#################################
#Arg for dataset
# For how many ms do we present a sample during classification
n_iters =  time_window
n_iters_test =  time_window
# How many epochs to run before testing
n_test_interval = 20
dt = 25000  # us, time of event accumulation for 1 frame
ds = 4      # size scale (1/ds)
target_size = 11 # num_classes
n_epochs = 4000  # in fact number of batches(no classical epoch)
in_channels = 2  # Green and Red
im_dims = im_width, im_height = (128 // ds, 128 // ds)
names = 'dvsGesture_stbp_cnn_test1'

#################################

parser = argparse.ArgumentParser(description='STBP for DVS gestures')
loss_train_list = []
loss_test_list = []
acc_train_list = []
acc_test_list = []
train_correct = 0
test_epoch = 20

# Load data
gen_train, _ = create_data(
    batch_size=batch_size,
    chunk_size=n_iters,
    size=[in_channels, im_width, im_height],
    ds=ds,
    dt=dt)

_, gen_test = create_data(
    batch_size=batch_size,
    chunk_size=n_iters_test,
    size=[in_channels, im_width, im_height],
    ds=ds,
    dt=dt)

def generate_test(gen_test, n_test: int, offset=0):
    input_test, labels_test = gen_test.next(offset=offset)
    input_tests = []
    labels1h_tests = []
    n_test = min(n_test, int(np.ceil(input_test.shape[0] / batch_size)))
    for i in range(n_test):
        input_tests.append(
            torch.Tensor(input_test.swapaxes(0, 1))[:, i * batch_size:(i + 1) * batch_size].reshape(n_iters_test, -1,
                                                                                                    in_channels,
                                                                                                    im_width,
                                                                                                    im_height))
        labels1h_tests.append(torch.Tensor(labels_test[:, i * batch_size:(i + 1) * batch_size]))
    return n_test, input_tests, labels1h_tests

n_test, input_tests, labels1h_tests = generate_test(gen_test, n_test=100, offset=0)

modules = import_module('models.LIAFCNN')
config = modules.Config()
config.cfgCnn = [(2, 64, 3, 1, 1, False),(64, 128, 3, 2, 1, True)]
config.cfgFc = [256, 11]
config.decay = 0.3
config.dropOut= 0
config.dataSize=[im_width,im_height]
config.padding=1
config.timeWindows=time_window
config.dropOut=0
config.useBatchNorm=True
config.usetdBatchNorm=False
config.useThreshFiring=True
config.actFun=torch.selu
snn = modules.LIAFCNN(config).to(config.device)
device_ids=[0,1,2,3]
snn = nn.DataParallel(snn,device_ids=device_ids)

best_acc = 0
acc = 0
criterion = nn.CrossEntropyLoss()
######################################################################################
#note:
#CorssEntrophyLoss适用于分类问题（其为Max函数的连续近似）
#它的输入是output（每一类别的概率）和label（第几个类别）
######################################################################################
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)

running_loss = 0

for epoch in range(n_epochs):

    #training
    snn.train(mode=True)
    snn.zero_grad()
    optimizer.zero_grad()
    start_time = time.time()
    input, labels = gen_train.next()
    input = torch.Tensor(input.swapaxes(1,2)).float()
    labels = torch.from_numpy(labels).float()
    _ , labels = labels[1, :, :].max(dim=1)
    outputs = snn(input)
    loss = criterion(outputs.cpu(), labels)
    running_loss = running_loss + loss.item()
    loss.backward()
    optimizer.step()
    writer.add_scalar('loss', loss.item(), epoch)
    print('Epoch [%d/%d], Loss:%.5f' % (epoch + 1, n_epochs, loss.item()))

    torch.cuda.empty_cache()
    
    #evaluation
    snn.eval()
    with torch.no_grad():
        for name,parameters in snn.named_parameters():
            writer.add_histogram(name, parameters.detach().cpu().numpy(), epoch)
        if (epoch + 1) % test_epoch == 0:
            correct = 0
            total = 0
            optimizer = util.lr_scheduler(optimizer, epoch, learning_rate, 200)
            test_loss = 0
            for i in range(len(input_tests)):
                
                inputTest = input_tests[i].transpose(0,2).transpose(0,1).float()

                outputs = snn(inputTest)
                
                _ , predicted = torch.max(outputs.data, 1)
                _ , labels = labels1h_tests[i][0, :, :].max(dim=1)

                total = total + predicted.size(0)
                correct = correct + (predicted.cpu() == labels.cpu()).sum()
                loss = criterion(outputs.data.cpu(), labels.data.cpu())
                test_loss += loss.item()

            test_loss = test_loss / len(input_tests)
            running_loss = running_loss / test_epoch
            print('test loss:%.5f' % (test_loss))
            print('tarin loss:%.5f' % (running_loss))
            loss_test_list.append(test_loss)
            loss_train_list.append(running_loss)
            running_loss = 0
            acc = 100. * float(correct) / float(total)
            acc_test_list.append(acc)
            acc_train_list.append(float(train_correct) / (batch_size * test_epoch))
            print('test acc: %.3f' % (100 * correct.float() / total))
            train_correct = 0
            writer.add_scalar('accuracy', acc, epoch)

