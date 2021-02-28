import numpy as np
import torch
import linecache
import torch.utils.data as data

class ESImagenet_Dataset(data.Dataset):
    def __init__(self, mode, data_set_path='/data/ES-imagenet-release1.0/'):
        super().__init__()
        self.mode = mode
        self.filenames = []
        self.trainpath = data_set_path+'train'
        self.testpath = data_set_path+'val'
        self.traininfotxt = data_set_path+'trainlabel.txt'
        self.testinfotxt = data_set_path+'vallabel.txt'
        if mode == 'train':
            self.path = self.trainpath
            trainfile = open(self.traininfotxt, 'r')
            for line in trainfile:
                filename, classnum, a, b = line.split()
                self.filenames.append(filename)
        else:
            self.path = self.testpath
            testfile = open(self.testinfotxt, 'r')
            for line in testfile:
                filename, classnum, a, b = line.split()
                self.filenames.append(filename)

    def __getitem__(self, index):
        if self.mode == 'train':
            info = linecache.getline(self.traininfotxt, index+1)
        else:
            info = linecache.getline(self.testinfotxt, index+1)
        filename, classnum, a, b = info.split()
        filename = self.path + r'/' + filename
        classnum = int(classnum)
        a = int(a)
        b = int(b)
        data = np.load(filename)
        dy = (254 - b) // 2
        dx = (254 - a) // 2
        input = np.zeros([8, 2, 256, 256])
        #print(data)#data[x,y,t,p]
        p = data[:,3]
        ones = np.where(p == 1)
        mones = np.where(p == 255)
        xyt_ones = data[ones]
        xyt_mones = data[mones]
        
        x = xyt_ones[:,0]
        y = xyt_ones[:,1]
        t = xyt_ones[:,2]
        plus= input[:,0,:,:]
        plus[t - 1,x + dx,y + dy] = 1

        x = xyt_mones[:,0]
        y = xyt_mones[:,1]
        t = xyt_mones[:,2]
        minus= input[:,1,:,:]
        minus[t - 1,x + dx,y + dy] = 1

        reshape = torch.tensor(input[:, :, 16:240, 16:240]).transpose(0,1)
        label = torch.tensor([classnum])
        return reshape, label

    def __len__(self):
        return len(self.filenames)
