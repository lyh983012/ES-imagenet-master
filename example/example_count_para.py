# Author: lyh 
# Date  : 2020-09-19
# run it use foloowing comand
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m torch.distributed.launch --nproc_per_node=7 example_ES_3DCNN.py
from __future__ import print_function
import sys
sys.path.append("..")
from util.util import lr_scheduler
from datasets.es_imagenet_new import ESImagenet_Dataset
import LIAF

from LIAFnet.ResNet3D import *

import torch.distributed as dist 
import torch.nn as nn
import argparse, pickle, torch, time, os,sys
from importlib import import_module
from torch.nn.parallel import DistributedDataParallel as DDP
from LIAFnet.LIAFResNet import *


modules = import_module('LIAFnet.LIAFResNet_34')
config  = modules.Config()
snn = LIAFResNet(config)
print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in snn.parameters())))
