# es-imagenet-master

![image](./viz.gif)

code for generating data set ES-ImageNet with corresponding training code

## dataset generator 
  - some codes of ODG algorithm
  - The variables to be modified include datapath (data storage path after transformation, which needs to be created before transformation) and root_Path (root directory of training set before transformation)
  
  | file name | function |
  | ---- | ---- |
  | traconvert.py        | converting training set of ISLVRC 2012 into event stream using ODG |
  | trainlabel_dir.txt   | It stores the corresponding relationship between the class name and label of the original Imagenet file |
  | trainlabel.txt       | It is generated during transformation and stores the label of training set |
  | valconvert.py        | Transformation code for test set. |
  | valorigin.txt        | Original test label of ImageNet-1K.  Put it in the same folder with valconvert.py if you need. |
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
  
  ### Requirements
  -   Python >= 3.5
  -   Pytorch >= 1.7
  -   CUDA >=10.0
  -   TenosrBoradX(optional)

  ### Train the baseline models
  
  ```bash
  
  $ cd example
  
  $ CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 example_ES_res18.py #LIAF/LIF-ResNet-18
  $ CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 example_ES_res34.py #LIAF/LIF-ResNet-34
  $ CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 compare_ES_3DCNN34.py #3DCNN-ResNet-34
  $ CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 compare_ES_3DCNN18.py #3DCNN-ResNet-18
  $ CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 compare_ES_2DCNN34.py #2DCNN-ResNet-34 
  $ CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 compare_ES_2DCNN18.py #2DCNN-ResNet-18
  $ CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 compare_CONVLSTM.py #ConvLSTM (no used in paper)
  $ CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 example_ES_res50.py #LIAF/LIF-ResNet-50 (no used in paper)
  ```

** note:** To select LIF mode, change the config files under /LIAFnet :
``` self.actFun= torch.nn.LeakyReLU(0.2, inplace=False) #nexttest:selu```
to
``` self.actFun= LIAF.LIFactFun.apply```


  ### baseline / Benchmark
  |Network|layer| Type Test Acc/%| # of Para| FP32+/GFLOPs|FP32x/GFLOPs|
  | ---- | ---- | ---- | ---- |---- |---- |
  | ResNet18 |2D-CNN |41.030 |11.68M|1.575|1.770 |
  | ResNet18|3D-CNN |38.050 |28.56M|12.082|12.493 |
  | ResNet18|LIF |39.894 |11.69M|12.668|0.269 |
  | ResNet18|LIAF |42.544| 11.69M|12.668|14.159 |
  | ResNet34|2D-CNN |42.736| 21.79M|3.211|3.611 |
  | ResNet34|3D-CNN |39.410 |48.22M|20.671|21.411 |
  | ResNet34|LIF| 43.424 |21.80M|25.783|0.288 |
  | ResNet18+imagenet-pretrain (a)|LIF |**43.74** |11.69M|12.668|0.269 |
  | ResNet34|LIAF| 47.466 |21.80M|25.783|28.901 |
  | ResNet18+self-pretrain|LIAF |50.54| 11.69M|12.668|14.159 |
  | ResNet18+imagenet-pretrain (b)|LIAF |**52.25**| 11.69M|12.668|14.159 |
  | ResNet34+imagenet-pretrain (c)|LIAF| 51.83 |21.80M|25.783|28.901 |
  
  Note: model (a), (b) and (c) are stored in ./pretrained_model

## Download

- The datasets ES-ImageNet (100GB) for this study can be download in the [Tsinghua Cloud1](https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/) or [Openl](https://git.openi.org.cn/xjtu_ym/ES-ImageNet/datasets?type=0)

- The converted event-frame version (40GB) can be found in [Tsinghua Cloud2](https://cloud.tsinghua.edu.cn/d/ee07f304fb3a498d9f0f/)

- If you only need the validation set, you can download it in [Tsinghua Cloud3](https://cloud.tsinghua.edu.cn/f/5e32c9fdd8094f3994df/) separately

## Citation
If you use this for research, please cite. Here is an example BibTeX entry:

```
@ARTICLE{ES_ImageNet2021,
AUTHOR={Lin, Yihan and Ding, Wei and Qiang, Shaohua and Deng, Lei and Li, Guoqi},   
TITLE={ES-ImageNet: A Million Event-Stream Classification Dataset for Spiking Neural Networks},      
JOURNAL={Frontiers in Neuroscience},      
VOLUME={15},      
PAGES={1546},     
YEAR={2021},      	  
URL={https://www.frontiersin.org/article/10.3389/fnins.2021.726582},       	
DOI={10.3389/fnins.2021.726582},      
ISSN={1662-453X},   
}
```
