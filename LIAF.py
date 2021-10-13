import torch
import torch.nn as nn 
import torch.nn.functional as F
import os
import math
import util.thBN as thBN

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)# if you are using multi-GPU.
torch.cuda.manual_seed(1)   
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True 

dtype = torch.float
allow_print = False
using_td_batchnorm = False

#######################################################
#activation
class LIFactFun(torch.autograd.Function):
    thresh = 0.5                                #LIF激活函数的阈值参数
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
        fire = input.gt(thresh).float() 
        ctx.save_for_backward(input)
        ctx.thresh = thresh
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
class baseNeuron(nn.Module):
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

class LIAFCell(baseNeuron):
    #standard LIAF cell based on LIFcell
    #         v_t' = v_{t-1} + w * x_n
    #         f = spikefun(v_t')
    #         x_{n+1} = analogfun(v_t')
    #         v_t = v_t' * (1-f) * t
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

class LIAFConvCell(baseNeuron):
    # standard LIAF cell based on LIFcell
    #         v_t' = v_{t-1} + w conv x_n
    #         f = spikefun(v_t')
    #         x_{n+1} = analogfun(v_t')
    #         v_t = v_t' * (1-f) * t
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
    expansion = 1
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
        self.timeWindows = input.size()[2]
        self.batchSize = input.size()[0]

        shortcut_output = input
        cv1_output = self.cv1(input)
        cv1_output = self.bn1(cv1_output)
        cv1_output = self.mem_update(cv1_output)
        cv2_output = self.cv2(cv1_output)
        cv2_output = self.bn2(cv2_output)
        if self.downSample:
            shortcut_output = torch.zeros(cv2_output.size(),device=cv2_output.device,dtype=dtype)
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
        if self.downSample:
            shortcut_output = torch.zeros(cv3_output.size(),device=cv3_output.device,dtype=dtype)
            for time in range(self.timeWindows):
                shortcut_output[:,:,time,:,:] = self.shortcut(input[:,:,time,:,:])
            shortcut_output = self.shortcut_norm(shortcut_output)
        output = self.actFun(cv3_output+shortcut_output)
        output = self.mem_update(output)
        return output