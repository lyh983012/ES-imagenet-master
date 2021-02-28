#data:2021-01-07
#author:linyh
#email: 532109881@qq.com
#note:底层实现

import torch
import torch.nn as nn 
import torch.nn.functional as F
import os
import math
import util.thBN as thBN

#训练要求
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)# if you are using multi-GPU.
torch.cuda.manual_seed(1)   
torch.backends.cudnn.deterministic = False   #保证每次结果一样
torch.backends.cudnn.benchmark = True 

#这部分超参数不是很重要，可以不用修改
dtype = torch.float
allow_print = False
using_td_batchnorm = False

#######################################################
#activation
class LIFactFun(torch.autograd.Function):
    thresh = 0.25                                #LIF激活函数的阈值参数
    lens = 0.5                                  #LIF激活函数的梯度近似参数，越小则梯度激活区域越窄
    bias = -0.2                                 #多阈值激活函数的值域平移参数
    sigma = 1
    use_rect_approx = True
    use_gause_approx = False
    # LIFactFun : approximation firing function
    # For 2 value-quantified approximation of δ(x)
    # LIF激活函数
    def __init__(self):
        super(LIFactFun, self).__init__()

    @staticmethod
    def forward(ctx, input, thresh=None):
        if thresh is None:
            thresh = LIFactFun.thresh
        fire = input.gt(thresh.to(input.device)).float() 
        ctx.save_for_backward(input)
        ctx.thresh = thresh.to(input.device)
        return fire # 阈值激活

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = 0
        thresh = ctx.thresh
        if LIFactFun.use_rect_approx:
            temp = abs(input - LIFactFun.thresh) < LIFactFun.lens  
            grad = grad_input * temp.float() / (2 * LIFactFun.lens)
        if LIFactFun.use_gause_approx:
            temp = 0.3989422804014327 / LIFactFun.sigma * torch.exp(- 0.5 / (LIFactFun.sigma ** 2) * (input - LIFactFun.thresh + LIFactFun.bias) ** 2)
            grad = grad_input * temp.float()
        return grad, None

#######################################################
#init method
def paramInit(model,method='kaiming'):
    scale = 0.05
    if isinstance(model, nn.BatchNorm3d) or isinstance(model, nn.BatchNorm3d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    else:
        for name, w in model.named_parameters():
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.orthogonal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, LIFactFun.thresh)
            else:
                pass

#######################################################
# cells 
# keneral-newv-norm-liaf-activation-pooling

# LIF神经元
class baseNeuron(nn.Module):
    # 所有参数都可训练的LIF神经元
    # 阈值按ARLIF的训练方法
    # decay使用sigmoid归一化
    # FC处理2维输入，Conv处理4维输入
    fire_rate_upperbound = 0.8
    fire_rate_lowebound  = 0.2
    thresh_trainable = False
    decay_trainable = False
    def __init__(self):
        super().__init__()
        self.norm = 0.3
        self.decay = torch.nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)
        self.thresh = torch.ones(1).cuda() * 0.5
        
    def forward(self,input):
        raise NotImplementedError("Input neurons must implement `forward`")

    def norm_decay(self,decay):
        return torch.sigmoid(decay)
    
    def thresh_update(self,newV,spike):
        inputsize = newV.size()
        count = 1.0
        for dim in inputsize:
            count = count*dim
        firerate = spike.sum()/count
        if firerate> baseNeuron.fire_rate_upperbound:
            self.thresh += 0.1
        if firerate< baseNeuron.fire_rate_lowebound:
            self.thresh -= 0.1
            
    def mem_update_fc(self,x,init_mem=None,spikeAct = LIFactFun.apply):
        # 输入突触连接为fc层 输入维度为
        # [B,T,N]
        
        time_window = x.size()[1]
        spike = torch.zeros_like(x[:,0,:])
        output = torch.zeros_like(x)
        if init_mem is None:
            mem = x[:,0,:]
        else:
            mem = init_v  
        mem_old= 0
        for i in range(time_window):
            if i>=1 :
                mem = mem_old * self.norm_decay(self.decay)*(1 - spike.detach()) + x[:,i,:]
            mem_old = mem.clone()
            spike = spikeAct(mem_old,self.thresh)
            if baseNeuron.thresh_trainable: 
                self.thresh_update(mem_old,spike)
            output[:,i,:] = self.actFun(mem_old) 
        return output

    def mem_update(self,x,init_mem=None,spikeAct = LIFactFun.apply):
        # 输入突触连接为conv层 输入维度为
        # [B,C,T,W,H]
                
        time_window = x.size()[2]
        spike = torch.zeros_like(x[:,:,0,:,:])
        output = torch.zeros_like(x)
        if init_mem is None:
            mem = x[:,:,0,:,:]
        else:
            mem = init_v 
        mem_old= 0
        for i in range(time_window):
            if i>=1 :
                mem = mem_old * self.norm_decay(self.decay)*(1 - spike.detach()) + x[:,:,i,:,:]
            mem_old = mem.clone()
            spike = spikeAct(mem_old,self.thresh) 
            if baseNeuron.thresh_trainable: 
                self.thresh_update(mem_old,spike)
            output[:,:,i,:,:] = self.actFun(mem_old) 
        return output

# 复合的LIAF神经元，突触连接：linear
class LIAFCell(baseNeuron):

    #standard LIAF cell based on LIFcell
    #简介: 最简单的LIAF cell，由LIF模型的演变而来
    #记号：v为膜电位，f为脉冲，x为输入，w为权重，t是时间常数
    #         v_t' = v_{t-1} + w * x_n
    #         f = spikefun(v_t')
    #         x_{n+1} = analogfun(v_t')
    #         v_t = v_t' * (1-f) * t
    #用法：torch_network.add_module(name,LIAF.LIAFCell())
    #update: 2020-02-29
    #author: Linyh

    def __init__(self,
        inputSize,
        hiddenSize,
        actFun = torch.selu,
        timeWindows = 60,
        dropOut= 0,
        useBatchNorm = False,
        init_method='kaiming',
        act = True
        ):
        '''
        @param input_size: (Num) number of input
        @param hidden_size: (Num) number of output
        @param actfun: handle of activation function to hidden layer(analog fire)
        @param decay: time coefficient in the model 
        @param Qbit: >=2， # of thrsholds of activation 
        @param dropout: 0~1
        @param useBatchNorm: if use batch-norm
        '''
        super().__init__()
        self.inputSize = inputSize              
        self.hiddenSize = hiddenSize 
        self.timeWindows = timeWindows 
        self.act = act          
        self.actFun = actFun                
        self.spikeActFun = LIFactFun.apply  
        self.useBatchNorm = useBatchNorm    
        self.batchSize = None

        # block 1：add synaptic inputs: Wx+b=y
        self.kernel=nn.Linear(inputSize, hiddenSize)    
        paramInit(self.kernel,init_method)

        # block 2： add a BN layer
        if self.useBatchNorm:
            self.NormLayer = nn.BatchNorm1d(self.timeWindows)#???
        
        # block 3： use dropout
        self.UseDropOut = False
        self.DPLayer = nn.Dropout(dropOut)             
        if 0 < dropOut < 1: 
            self.UseDropOut = True

    def forward(self,
        input,
        init_v=None):
        """
        @param input: a tensor of of shape (Batch, time, insize)
        @param init_v: a tensor with size of (Batch, time, outsize) denoting the mambrane v.
        """
        #step 0: init
        self.batchSize = input.size()[0]#adaptive for mul-gpu training
        self.device = self.kernel.weight.device
        self.timeWindows = input.size()[1]
        if input.device != self.device:
            input = input.to(self.device)

        synaptic_input = torch.zeros(self.batchSize,self.timeWindows,self.hiddenSize,device=self.device,dtype=dtype)
       
        # Step 1: accumulate 
        for time in range(self.timeWindows):
            synaptic_input[:,time,:] = self.kernel(input[:,time,:].view(self.batchSize, -1))
        # Step 2: Normalization 
        if self.useBatchNorm:
            synaptic_input = self.NormLayer(synaptic_input)

        # Step 3: update membrane
        output = self.mem_update_fc(synaptic_input)
        
        # step 4: DropOut
        if self.UseDropOut:
            output = self.DPLayer(output)
            
        return output

# 复合的LIAF神经元，突触连接：linear+wighted-memory
class LIAFRCell(baseNeuron):

    #standard LIAF-RNN cell based on LIAFcell
    #简介: 仿照RNN对LIAF模型进行轻度修改
    #记号：v为膜电位，f为脉冲，x为输入，w1w2为权重，t是时间常数
    #         v_t' = w1 *v_{t-1} + w2 * x_n
    #         f = spikefun(v_t')
    #         x_{n+1} = analogfun(v_t')
    #         v_t = v_t' * (1-f) * t
    #update: 2020-03-21
    #author: Linyh

    def __init__(self,
        inputSize,
        hiddenSize,
        actFun = torch.selu,
        dropOut= 0,
        timeWindows = 32,
        useBatchNorm = False,
        useLayerNorm = False,
        init_method='kaiming'
        ):
        '''
        @param input_size: (Num) number of input
        @param hidden_size: (Num) number of output
        @param actfun: handle of activation function 
        @param decay: time coefficient in the model 
        @param dropout: 0~1
        @param useBatchNorm: if use
        '''

        super().__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        

        self.actFun = actFun
        self.timeWindows = timeWindows
        self.spikeActFun = LIFactFun.apply
        self.useBatchNorm = useBatchNorm
        self.useLayerNorm = useLayerNorm
        self.UseDropOut = True
        self.batchSize = None
        # block 1. add synaptic inputs: Wx+b=y
        self.kernel=nn.Linear(inputSize, hiddenSize)
        paramInit(self.kernel,init_method)
        self.kernel_v = nn.Linear(hiddenSize, hiddenSize)
        paramInit(self.kernel_v,init_method)
        # note: random initialization weights ->[0,scale),bias->[0,scale)
        # block 2. add a BN layer
        if self.useBatchNorm:
            if using_td_batchnorm:
                self.NormLayer = thBN.BatchNorm1d(hiddenSize)
            else:
                self.NormLayer = nn.BatchNorm1d(hiddenSize)

        if self.useLayerNorm:
            self.NormLayer = nn.LayerNorm(hiddenSize)
        # block 3. use dropout
        self.UseDropOut = False
        self.DPLayer = nn.Dropout(dropOut)
        if 0 < dropOut < 1:  # enable drop_out in cell
            self.UseDropOut = True

        # Choose QLIAF mode
        if Qbit>0 :
            MultiLIFactFun.Nbit = Qbit
            print("warning: assert ActFun = MultiLIFactFun.apply")
            self.actFun = MultiLIFactFun.apply
            print(MultiLIFactFun.Nbit)


    def forward(self,
        input,
        init_v=None):
        """
        @param input: a tensor of of shape (Batch, time, insize)
        @param init_v: a tensor with size of (Batch, time, outsize) denoting the mambrane v.
        """
        #step 0: init
        self.device = self.kernel.weight.device
        if self.timeWindows != input.size()[1]:
            print('wrong num of time intervals')
        if input.device != self.device:
            input = input.to(self.device)
        
        self.batchSize = input.size()[0]#adaptive for mul-gpu training
        output_init = torch.zeros(self.batchSize,self.timeWindows,self.hiddenSize,device=self.device,dtype=dtype)
        output_fired = torch.zeros(self.batchSize,self.timeWindows,self.hiddenSize,device=self.device,dtype=dtype)
        # Step 1: accumulate 
        for time in range(self.timeWindows):
            event_frame_t = input[:,time,:].float().to(self.device)
            event_frame_t = event_frame_t.view(self.batchSize, -1)
            output_init[:,time,:] = self.kernel(event_frame_t)
        # Step 2: Normalization 
        normed_v = output_init
        if self.useBatchNorm:
            normed_v = self.NormLayer(output_init)

        if v is None:#initialization of V
            if init_v is None:
                v = torch.zeros(self.batchSize, self.hiddenSize, device=event_frame_t.device,dtype=dtype)
            else:
                v = init_v   

        for time in range(self.timeWindows):
            v = normed_v[:,time,:] + self.kernel_v(v)
            # Step 2: Fire and leaky 
            fire = self.spikeActFun(v)
            output = self.actFun(v)
            output_fired[:,time,:] = output
            v = self.decay * (1 - fire) * v
        
        if self.useLayerNorm:
            output = self.NormLayer(output)

        # step 4: DropOut
        if self.UseDropOut:
            output_fired = self.DPLayer(output_fired)
        return output_fired

# 复合的LIAF神经元，突触连接：2dconv
class LIAFConvCell(baseNeuron):
    # standard LIAF cell based on LIFcell
    #简介: 替换线性为卷积的 cell，由LIF模型的演变而来
    #记号：v为膜电位，f为脉冲，x为输入，w为权重，t是时间常数
    #         v_t' = v_{t-1} + w conv x_n
    #         f = spikefun(v_t')
    #         x_{n+1} = analogfun(v_t')
    #         v_t = v_t' * (1-f) * t
    # update: 2020-02-29
    # author: Linyh
    def __init__(self,
                 inChannels,
                 outChannels,
                 kernelSize,
                 stride,
                 padding =0,
                 dilation=1,
                 groups=1,
                 timeWindows = 60,
                 actFun=LIFactFun.apply,
                 dropOut =0,
                 useBatchNorm = False,
                 inputSize=(224,224),
                 init_method='kaiming',
                 act = True,
                 usePool= True,
                 p_method = 'avg',
                 p_kernelSize = 2,
                 p_stride = 2,
                 p_padding = 0
                 ):
        '''
        @param inChannels: (Num) number of input Channels
        @param outChannels: (Num) number of output Channels
        @param kernelSize: size of convolutional kernel
        @param p_kernelSize: size of pooling kernel
        @param stride，padding，dilation，groups -> for convolutional input connections
        @param outChannels: (Num) number of output
        @param actfun: handle of activation function to hidden layer(analog fire)
        @param decay: time coefficient in the model 
        @param Qbit: >=2， # of thrsholds of activation 
        @param dropout: 0~1
        @param useBatchNorm: if use batch-norm
        @param p_method: 'max' or 'avg'
        @param act: designed for resnet
        '''
        super().__init__()
        self.kernelSize = kernelSize

        self.decay =  torch.nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)

        self.actFun = actFun
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.spikeActFun = LIFactFun.apply
        self.useBatchNorm= useBatchNorm
        self.usePool= usePool
        self.inputSize=inputSize
        self.batchSize= None
        self.outChannel=outChannels
        self.layer_index= 0
        self.p_method  = p_method
        self.p_stride = p_stride
        self.p_kernelSize = p_kernelSize
        self.p_padding  = p_padding
        self.act=act
        self.timeWindows=timeWindows
        # block 1. conv layer:
        self.kernel = nn.Conv2d(inChannels,
                                outChannels,
                                self.kernelSize,
                                stride=self.stride,
                                padding= self.padding,
                                dilation=self.dilation,
                                groups=self.groups,
                                bias=True,
                                padding_mode='zeros')
        paramInit(self.kernel,init_method)
        # block 2. add a pooling layer
        
        # block 3. use dropout
        self.UseDropOut=False
        self.DPLayer = nn.Dropout(dropOut)
        if 0 < dropOut < 1: 
            self.UseDropOut = True# enable drop_out in cell
        
        # Automatically calulating the size of feature maps
        self.CSize = list(self.inputSize)
        self.CSize[0] = math.floor((self.inputSize[0] + 2 * self.padding 
            - kernelSize[0]) / stride + 1)
        self.CSize[1] = math.floor((self.inputSize[1] + 2 * self.padding 
            - kernelSize[1]) / stride + 1)
        self.outputSize = list(self.CSize)

        if self.useBatchNorm:
            self.NormLayer = nn.BatchNorm3d(outChannels)
            
        if self.usePool:
            self.outputSize[0] = math.floor((self.outputSize[0]+ 2 * self.p_padding 
            - self.p_kernelSize) / self.p_stride + 1)
            self.outputSize[1] = math.floor((self.outputSize[1]+ 2 * self.p_padding 
            - self.p_kernelSize) / self.p_stride + 1)
        
    def forward(self,
                input,
                init_v=None):
        """
        @param input: a tensor of of shape (B, T, C，H，W)
        @param init_v: an tensor denoting initial mambrane v with shape of  (B,T, C，H，W). set None use 0
        @return: new state of cell and output
        note : only batch-first mode can be supported 
        """
        #step 0: init
        self.timeWindows = input.size()[2]
        self.batchSize=input.size()[0]
        self.device = self.kernel.weight.device
        if input.device != self.device:
            input = input.to(self.device) 
        synaptic_input = torch.zeros(self.batchSize,self.outChannel, self.timeWindows, 
                    self.CSize[0], self.CSize[1],device=self.device,dtype=dtype)
        
        # Step 1: accumulate 
        for time in range(self.timeWindows):
            synaptic_input[:,:,time,:,:] = self.kernel(input[:,:,time,:,:].to(self.device))
        
        # Step 2: Normalization 
        if self.useBatchNorm:
            synaptic_input = self.NormLayer(synaptic_input)
        
        # Step 3: LIF
        output = self.mem_update(synaptic_input)
        
        # Step 4: Pooling
        if  self.usePool:
            output_pool = torch.zeros(self.batchSize,self.outChannel, self.timeWindows, 
                    self.outputSize[0], self.outputSize[1],device=self.device,dtype=dtype)
            for time in range(self.timeWindows):
                if self.p_method == 'max':
                    output_pool[:,:,time,:,:] = F.max_pool2d(output[:,:,time,:,:], kernel_size=self.p_kernelSize,
                    stride = self.p_stride,padding=self.p_padding)
                else:
                    output_pool[:,:,time,:,:] = F.avg_pool2d(output[:,:,time,:,:], kernel_size=self.p_kernelSize,
                    stride = self.p_stride,padding=self.p_padding)
        else:
            output_pool = output
        
        # step 5: DropOut
        if self.UseDropOut:
            output_pool = self.DPLayer(output_pool)
        return output_pool

# 时间维分别卷积的卷积层
class Temporal_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,inputSize,
            stride=1,padding=1, dilation=1, groups=1,
            bias=True, padding_mode='zeros', marker='b'):
        super(Temporal_Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                stride, padding, dilation, groups, bias, padding_mode)
        self.marker = marker
        self.h = (inputSize[0]-self.kernel_size[0]+2*self.padding[0])//self.stride[0]+1
        self.w = (inputSize[1]-self.kernel_size[0]+2*self.padding[0])//self.stride[0]+1
        self.outputSize = [self.h,self.w]

    def forward(self, input):
        timeWindows = input.size()[2]
        batchSize=input.size()[0]
        C = torch.zeros(batchSize,self.out_channels,timeWindows , self.h, self.w ,device=input.device)
        for i in range(timeWindows):
            C[:,:,i,:,:] = F.conv2d(input[:,:,i,:,:], self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return C

#ResBlock for ResNet18/34
class LIAFResBlock(baseNeuron):
    # standard LIAF cell based on LIFcell
    # update: 2020-08-10
    # author: Linyh
    expansion = 1
    #简介: 基于LIAF-CNN的残差块
    def __init__(self,
                 inChannels,
                 outChannels,
                 kernelSize=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 actFun=LIFactFun.apply,
                 useBatchNorm = False,
                 inputSize=(224,224),
                 name = 'liafres'
                 ):

        super().__init__()
        self.padding=padding
        self.actFun=actFun
        self.useBatchNorm = useBatchNorm
        self.kernelSize=kernelSize
        self.timeWindows= None
        self.downSample=False
        self.shortcut = None

        if inChannels!=outChannels:
            #判断残差类型——>输入输出是否具有相同维度
            stride = 2
            self.downSample = True
            #print(name +' dimension changed')  
            self.shortcut = nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=2)
        else:
            stride = 1

        self.cv1 = Temporal_Conv2d(in_channels=inChannels,
                                    out_channels=outChannels,
                                    kernel_size=kernelSize,
                                    stride= stride,
                                    inputSize = inputSize)
        self.cv2 = Temporal_Conv2d(in_channels=outChannels,
                                    out_channels = outChannels,
                                    kernel_size = kernelSize,
                                    stride = 1,
                                    inputSize = self.cv1.outputSize)
        self.outputSize = self.cv2.outputSize

        if self.useBatchNorm:
            if using_td_batchnorm:
                self.bn1 = thBN.BatchNorm3d(outChannels)
                self.bn1.k = 4
                self.bn2 = thBN.BatchNorm3d(outChannels)
                self.bn2.k = 8
                self.shortcut_norm = thBN.BatchNorm3d(outChannels)
                self.shortcut_norm.k=8
            else:
                self.bn1 = nn.BatchNorm3d(outChannels)
                self.bn2 = nn.BatchNorm3d(outChannels)
                self.shortcut_norm = nn.BatchNorm3d(outChannels)
        if allow_print:
            print('the output feature map is'+str(self.outputSize))
            
    def forward(self,input):
        '''
        设计网络的规则：
        1.对于输出feature map大小相同的层，有相同数量的filters，即channel数相同；
        2. 当feature map大小减半时（池化），filters(channel)数量翻倍。
            对于残差网络，维度匹配的shortcut连接为实线，反之为虚线。维度不匹配时，同等映射有两种可选方案：  
            直接通过zero padding 来增加维度（channel）。
            乘以W矩阵投影到新的空间。实现是用1x1卷积实现的，直接改变1x1卷积的filters数目。这种会增加参数。
        '''
        self.timeWindows = input.size()[2]
        self.batchSize = input.size()[0]

        shortcut_output = input
        cv1_output = self.cv1(input)
        cv1_output = self.bn1(cv1_output)
        cv1_output = self.mem_update(cv1_output)
        cv2_output = self.cv2(cv1_output)
        cv2_output = self.bn2(cv2_output)

        shortcut_output = torch.zeros(cv2_output.size(),device=cv2_output.device,dtype=dtype)
        if self.downSample:
            for time in range(self.timeWindows):
                shortcut_output[:,:,time,:,:] = self.shortcut(input[:,:,time,:,:])
        # 3dnorm require[batch channel depth width height]
        # input is [batch channel time  width height]
        shortcut_output = self.shortcut_norm(shortcut_output)
        output = self.actFun(cv2_output+shortcut_output)
        output = self.mem_update(output)
        return output

#ResNeck for ResNet50+
class LIAFResNeck(baseNeuron):
    # standard LIAF cell based on LIFcell
    # update: 2020-11-22
    # author: Linyh
    #简介: 基于LIAF-CNN的残差块

    expansion = 4
    
    def __init__(self,
                 cahnnel_now,
                 inChannels,
                 kernelSize=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 actFun=LIFactFun.apply,
                 useBatchNorm = False,
                 useLayerNorm = False,
                 inputSize=(224,224),
                 name = 'liafres'
                 ):

        super().__init__()
        self.padding=padding

        self.actFun=actFun
        self.useBatchNorm = useBatchNorm
        self.useLayerNorm = useLayerNorm
        self.kernelSize=kernelSize
        self.timeWindows= None
        self.downSample=False
        self.shortcut = None

        if (inChannels* LIAFResNeck.expansion)!=cahnnel_now:
            #判断残差类型——>输入输出是否具有相同维度
            stride = 2
            self.downSample = True
            #print(name +' dimension changed')  
            self.shortcut = nn.Conv2d(cahnnel_now, inChannels* LIAFResNeck.expansion, kernel_size=1, stride=2)
        else:
            stride = 1

        self.cv1 = Temporal_Conv2d(in_channels=cahnnel_now,
                                    out_channels=inChannels,
                                    kernel_size= 1,
                                    padding= 0,
                                    stride= stride,
                                    inputSize = inputSize)
        self.cv2 = Temporal_Conv2d(in_channels=inChannels,
                                    out_channels = inChannels,
                                    kernel_size = kernelSize,
                                    stride = 1,
                                    padding= 1,
                                    inputSize = self.cv1.outputSize)
        self.cv3 = Temporal_Conv2d(in_channels=inChannels,
                                    out_channels = inChannels* LIAFResNeck.expansion,
                                    kernel_size = 1,
                                    stride = 1,
                                    padding= 0,
                                    inputSize = self.cv2.outputSize)
        self.outputSize = self.cv3.outputSize

        if self.useBatchNorm:
            self.bn1 = nn.BatchNorm3d(inChannels)
            self.bn2 = nn.BatchNorm3d(inChannels)
            self.bn3 = nn.BatchNorm3d(inChannels* LIAFResNeck.expansion )
            self.shortcut_norm = nn.BatchNorm3d(inChannels* LIAFResNeck.expansion)
        if self.useLayerNorm:
            self.bn1 = nn.BatchNorm3d(inChannels)
            self.bn2 = nn.BatchNorm3d(inChannels)
            self.bn3 = nn.BatchNorm3d(inChannels)
            self.shortcut_norm = nn.BatchNorm3d(inChannels* LIAFResNeck.expansion)

        if allow_print:
            print('the output feature map is'+str(self.outputSize))
            
    def forward(self,input):
        '''
        设计网络的规则：
        1.对于输出feature map大小相同的层，有相同数量的filters，即channel数相同；
        2. 当feature map大小减半时（池化），filters(channel)数量翻倍。
            对于残差网络，维度匹配的shortcut连接为实线，反之为虚线。维度不匹配时，同等映射有两种可选方案：  
            直接通过zero padding 来增加维度（channel）。
            乘以W矩阵投影到新的空间。实现是用1x1卷积实现的，直接改变1x1卷积的filters数目。这种会增加参数。
        '''
        self.timeWindows = input.size()[2]
        self.batchSize = input.size()[0]

        shortcut_output = input
        cv1_output = self.cv1(input)
        cv1_output = self.bn1(cv1_output)
        cv1_output = self.mem_update(cv1_output)

        cv2_output = self.cv2(cv1_output)
        cv2_output = self.bn2(cv2_output)
        cv2_output = self.mem_update(cv2_output)

        cv3_output = self.cv3(cv2_output)
        cv3_output = self.bn3(cv3_output)

        shortcut_output = torch.zeros(cv3_output.size(),device=cv3_output.device,dtype=dtype)
        if self.downSample:
            for time in range(self.timeWindows):
                shortcut_output[:,:,time,:,:] = self.shortcut(input[:,:,time,:,:])
            shortcut_output = self.shortcut_norm(shortcut_output)
        output = self.actFun(cv3_output+shortcut_output)
        output = self.mem_update(output)
        return output

# 复合的LIAF神经元，以门控形式组合
class LIAFLSTMCell(nn.Module):
    #LSTMCell
    #update: 2020-03-21
    #author: Linyh
    #简介: 仿照RNN对LIAF模型进行轻度修改，每个门维护一个膜电位
    #方案1: 全仿LSTM，当decay=0，spikefire为sigmoid时完全是LSTM
    #记号：类似
    #         v_t' = v_{t-1} + w2 * x_n
    #         f = spikefun(v_t')
    #         v_t = v_t' * (1-f) * t
    #
    #         fi = \sigma(vi) 
    #         ff = \sigma(vf) 
    #         fg = \tanh(vg)
    #         fo = \sigma(vo) 
    #         c' = ff * fc + fi * fg 
    #         x_{n+1} = o * \tanh(c') 

    def __init__(self,
        inputSize,
        hiddenSize,
        actFun = torch.selu,
        spikeActFun = LIFactFun.apply,
        decay = 0.3,
        dropOut= 0,
        useBatchNorm= False,
        useLayerNorm= False,
        timeWindows = 5,
        Qbit = 0,
        init_method='kaiming',
        sgFun = torch.relu
        ):
        '''
        @param input_size: (Num) number of input
        @param hidden_size: (Num) number of output
        @param actfun: handle of activation function 
        @param actfun: handle of recurrent spike firing function
        @param decay: time coefficient in the model 
        @param dropout: 0~1 unused
        @param useBatchNorm: unused
        '''

        super().__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.decay = decay
        self.actFun = actFun
        self.sgFun = sgFun #default:sigmoid
        self.spikeActFun = spikeActFun
        self.useBatchNorm = useBatchNorm
        self.useLayerNorm = useLayerNorm
        self.timeWindows = timeWindows
        self.UseDropOut = True
        self.batchSize = None
        # block 1. add synaptic inputs: Wx+b=y
        self.kernel_i=nn.Linear(inputSize, hiddenSize)
        paramInit(self.kernel_i,init_method)
        self.kernel_f = nn.Linear(inputSize, hiddenSize)
        paramInit(self.kernel_f,init_method)
        self.kernel_g = nn.Linear(inputSize, hiddenSize)
        paramInit(self.kernel_g,init_method)
        self.kernel_o = nn.Linear(inputSize, hiddenSize)
        paramInit(self.kernel_o,init_method)

        # block 2. add a Norm layer
        if self.useBatchNorm:
            self.BNLayerx = nn.BatchNorm1d(self.timeWindows)
        if self.useLayerNorm:
            self.Lnormx = nn.LayerNorm([self.timeWindows,hiddenSize])
        # block 3. use dropout
        self.UseDropOut = False
        self.DPLayer = nn.Dropout(dropOut)
        if 0 < dropOut < 1:  # enable drop_out in cell
            self.UseDropOut = True

        # Choose QLIAF mode
        if Qbit>0 :
            MultiLIFactFun.Nbit = Qbit
            print("warning: assert ActFun = MultiLIFactFun.apply")
            self.actFun = MultiLIFactFun.apply
            print('# of threshold = ', MultiLIFactFun.Nbit)

        self.c =None

    def forward(self,
        input,
        init_v=None):
        """
        @param input: a tensor of of shape (Batch, N)
        @param state: a pair of a tensor including previous output and cell's potential with size (Batch,3, N).
        @return: new state of cell and output for hidden layer
        Dense Layer: linear kernel
        \begin{array}{ll}
        i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
        o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        x' = o * \tanh(c') \\
        \end{array}
        """
        #step 0: init
        self.batchSize = input.size()[0]#adaptive for mul-gpu training
        self.device = self.kernel_i.weight.device
        if self.timeWindows != input.size()[1]:
            print('wrong num of time intervals')
        if input.device != self.device:
            input = input.to(self.device)

        if init_v is None:
            vi = torch.zeros(self.batchSize, self.hiddenSize, device=input.device,dtype=dtype)
            vf = torch.zeros(self.batchSize, self.hiddenSize, device=input.device,dtype=dtype)
            vg = torch.zeros(self.batchSize, self.hiddenSize, device=input.device,dtype=dtype)
            vo= torch.zeros(self.batchSize, self.hiddenSize, device=input.device,dtype=dtype)
            c = torch.zeros(self.batchSize, self.hiddenSize, device=input.device,dtype=dtype)
        else:
            vi = init_v.clone()
            vf = init_v.clone()
            vg = init_v.clone()
            vo= init_v.clone()
            c = init_v.clone()
        
        output = torch.zeros(self.batchSize,self.timeWindows,self.hiddenSize,device=self.device,dtype=dtype)

        for time in range(self.timeWindows):
        # Step 1: accumulate and reset,spike used as forgetting gate
            vi = self.kernel_i(input[:,time,:].float()) + vi
            vf = self.kernel_f(input[:,time,:].float()) + vf 
            vg = self.kernel_g(input[:,time,:].float()) + vg
            vo= self.kernel_o(input[:,time,:].float()) + vo

            fi = self.spikeActFun(vi)
            ff = self.spikeActFun(vf)
            fg = self.spikeActFun(vg)
            fo = self.spikeActFun(vo)

            # Step 2: figo
            i = self.sgFun(vi)
            f = self.sgFun(vf)
            o = self.sgFun(vo)
            g = self.actFun(vg)

            # step 3: Learn 
            c = c * f  + i * g
            
            # step 4: renew
            output[:,time,:] = self.actFun(c) * o

            #step 5: leaky
            vi = self.decay * (1 - fi ) * vi
            vf = self.decay * (1 - ff ) * vf
            vg = self.decay * (1 - fg ) * vg
            vo= self.decay * (1 - fo ) * vo

        # step 6: Norms
        if self.useBatchNorm:
            output = self.BNLayerx(output)
        if  self.useLayerNorm:
            output = self.Lnormx(output)

        return output

# 复合的LIAF神经元，以GRU形式组合
class LIAFGRUCell(nn.Module):
    #########
    #author：Lin-Gao
    #in 2020-07
    #########
    def __init__(self, inputSize, hiddenSize, spikeActFun, actFun=torch.selu, dropOut=0,
                 useBatchNorm=False, useLayerNorm=False, init_method='kaiming', gFun=torch.tanh, decay=0.3):
        """
        :param inputSize:(Num) number of input
        :param hiddenSize:(Num) number of output
        :param actFun:handle of activation function
        :param spikeAcFun:handle of recurrent spike firing function
        :param dropOut: 0~1 unused
        :param useBatchNorm:
        :param useLayerNorm:
        :param init_method:
        :param gFun:
        """
        super().__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.decay = decay
        self.actFun = actFun
        self.gFun = gFun  # default:tanh
        self.spikeActFun = spikeActFun
        self.useBatchNorm = useBatchNorm
        self.useLayerNorm = useLayerNorm
        self.UseDropOut = True
        self.batchSize = None
        # block 1. add synaptic inputs:Wx+b
        self.kernel_r = nn.Linear(inputSize, hiddenSize)
        paramInit(self.kernel_r, init_method)  # 作用
        self.kernel_z = nn.Linear(inputSize, hiddenSize)
        paramInit(self.kernel_z, init_method)
        self.kernel_h = nn.Linear(inputSize, hiddenSize)
        paramInit(self.kernel_h, init_method)
        # block 2. add synaptic inputs:Hx+b
        self.kernel_r_h = nn.Linear(hiddenSize, hiddenSize)
        paramInit(self.kernel_r_h, init_method)
        self.kernel_z_h = nn.Linear(hiddenSize, hiddenSize)
        paramInit(self.kernel_z_h, init_method)
        self.kernel_h_h = nn.Linear(hiddenSize, hiddenSize, bias=False)
        paramInit(self.kernel_h_h, init_method)
        # block 3. add a Norm layer
        if self.useBatchNorm:
            self.BNLayerx = nn.BatchNorm1d(hiddenSize)
            self.BNLayerc = nn.BatchNorm1d(hiddenSize)
        if self.useLayerNorm:
            self.Lnormx = nn.LayerNorm(hiddenSize)
            self.Lnormc = nn.LayerNorm(hiddenSize)
        # block 4. use dropout
        self.UseDropOut = False
        self.DPLayer = nn.Dropout(dropOut)
        if 0 < dropOut < 1:  # enable drop_out in cell
            self.UseDropOut = True

    def forward(self, input, init_v=None):
        """
        :param input:
        :param init_v:
        :return:
        """
        self.batchSize = input.size()[0]
        input = input.reshape([self.batchSize, -1])
        if input.device != self.kernel_r.weight.device:
            input = input.to(self.kernel_r.weight.device)
        if self.h is None:
            if init_v is None:
                self.h = torch.zeros(self.batchSize, self.hiddenSize, device=input.device,dtype=dtype)
                self.u = torch.zeros(self.batchSize, self.hiddenSize, device=input.device,dtype=dtype)
            else:
                self.h = init_v.clone()
                self.u = init_v.clone()
        # Step 1: accumulate and reset,spike used as forgetting gate
        r = self.kernel_r(input.float()) + self.kernel_r_h(self.h)
        z = self.kernel_z(input.float()) + self.kernel_z_h(self.h)
        r = self.actFun(r)
        z = self.actFun(z)
        h = self.kernel_h(input.float()) + self.kernel_h_h(self.h) * r
        h = self.gFun(h)
        # Step 2: renew
        h_ = self.h.clone()
        self.h = self.decay * self.u * (1 - self.spikeActFun(self.u))
        self.u = (1 - z) * h_ + z * h
        x = self.spikeActFun(self.u)

        # step 3: Norms
        if self.useBatchNorm:
            self.h = self.BNLayerc(self.h)
            self.u = self.BNLayerc(self.u)
            x = self.BNLayerx(x)
        if self.useLayerNorm:
            self.h = self.Lnormc(self.h)
            self.u = self.Lnormc(self.u)
            x = self.Lnormx(x)
        return x

