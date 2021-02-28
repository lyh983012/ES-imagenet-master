# es-imagenet-master
code for generating data set ES-ImageNet with corresponding training code

## dataset generator 
  - some codes of ODG algorithm
  - The variables to be modified include datapath (data storage path after transformation, which needs to be created before transformation) and root_Path (root directory of training set before transformation)
  | file name | function |
  | -- | -- |
  | traconvert.py        | converting training set of ISLVRC 2012 into event stream using ODG |
  | trainlabel_dir.txt   | It stores the corresponding relationship between the class name and label of the original Imagenet file |
  | trainlabel.txt       | It is generated during transformation and stores the label of training set |
  | valconvert.py        | Transformation code for test set. |
  | valorigin.txt        | Original test label, need and valconvert.py Put it in the same folder |
  | vallabel.txt         | It is generated during transformation and stores the label of training set. |

## dataset usage

  - codes are in ./datasets
  - some traing examples are provided for ES-imagenet in ./example
  An example code for easily using this dataset based on **Pytorch**
  ```python
  from __future__ import print_function
  import sys
  sys.path.append("..")
  from datasets.es_imagenet_new import ESImagenet_Dataset
  import torch.nn as nn
  import torch

  data_path = None #TODO:modify 
  train_dataset = ESImagenet_Dataset(mode='train',data_set_path=data_path)
  test_dataset = ESImagenet_Dataset(mode='test',data_set_path=data_path)

  train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
  test_sampler  = torch.utils.data.distributed.DistributedSampler(test_dataset)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1,pin_memory=True,drop_last=True,sampler=train_sampler)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1,pin_memory=True)
  
  for batch_idx, (inputs, targets) in enumerate(train_loader)
    pass
    # input = [batchsize,time,channel,width,height]
    
  for batch_idx, (inputs, targets) in enumerate(test_loader):
    pass
    # input = [batchsize,time,channel,width,height]
  ```
  
  
  ## training example and benchmarks
  
    