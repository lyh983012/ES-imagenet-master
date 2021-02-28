import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image
import time,os
import torch
import cv2
import os
from torch.nn import functional as F


from torchvision import transforms, datasets as ds
import torchvision as tv
from torch.utils.data import DataLoader
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES=True
thresh = '0.18'
datasetpath = "/data/ES-imagenet-"+thresh+"/"
datapath = "/data/ES-imagenet-"+thresh+"/val/"
root_path = "/data/imagenet2012_png/val"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
thresh = float(thresh)
f = open(datasetpath+'vallabel.txt','w')

def get_file_path(root_path,file_list,dir_list):
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(root_path,dir_file)
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            get_file_path(dir_file_path,file_list,dir_list)
        else:
            file_list.append(dir_file_path)
    
file_list = []
dir_list = []
get_file_path(root_path,file_list,dir_list)

tracex = [1,0,2,1,0,2,1,1,2]
tracey = [0,2,1,0,1,2,0,1,1]

def traceX(t):
    x = tracex[t]
    return x

def traceY(t):
    y = tracey[t]
    return y

count = 0
start = time.time()

aimsize = 254
event_rate = 0
error_sample = 0
pulse = list([])
timelist = list([])

fi_t = dict([])
for line in open("valorigin.txt"):   
    x,y = line.split('.')
    s,t = line.split()
    fi_t[x] = t

for file in file_list:
    time0 = time.time()
    imagetest = cv2.imread(file)
    imagetest = imagetest / 255.0
    imagetest = np.transpose(imagetest, (2, 0, 1))
    imagetorch = torch.from_numpy(imagetest)
    imagetorch = imagetorch.unsqueeze(0)
    a = imagetorch.shape[2]
    b = imagetorch.shape[3]
    rootname,subfilename = os.path.split(file)
    filename,_ = subfilename.split(".") 

    label = fi_t[filename]

    if (a>b):
        image = F.interpolate(imagetorch, size = (aimsize,(int)(b*aimsize/a)), mode = 'nearest')
        b = (int)(b*aimsize/a)
        a = aimsize
    else:
        image = F.interpolate(imagetorch, size = ((int)(a*aimsize/b),aimsize), mode = 'nearest')
        a = (int)(a*aimsize/b)
        b = aimsize

    image = image[0].to(device)
    imageInfo ,_ = torch.max(image,dim=0)
    imageShape0 = (imageInfo.size()[0], imageInfo.size()[1])

    imageInfo1 = torch.ones(imageShape0[0],imageShape0[1],dtype = torch.uint8,device=device)
    imageShape = (imageInfo.size()[0]+4,imageInfo.size()[1]+4)

    lastImage = torch.zeros(imageShape,device=device)
    newImage = torch.zeros(imageShape,device=device)
    torch1 = torch.ones([imageShape[0],imageShape[1]],dtype = torch.uint8,device=device)
    torch0 = torch.zeros_like(torch1,device=device)
    imageStorep = torch.ones([0,3],dtype = torch.uint8,device=device)
    imageStoren = torch.ones([0,3],dtype = torch.uint8,device=device)
    elen = 0
    for t in range(9):
        newImage = torch.zeros(imageShape,device=device)
        x = traceX(t)
        y = traceY(t)
        x0 =traceX(t-1)
        y0 = traceY(t-1)
        newImage[x:x+imageShape0[0],y:y+imageShape0[1]] = imageInfo      
        if t != 0:
            diffImage = newImage - lastImage
            diffImageInfo1 = torch.where(diffImage > thresh, torch1 , torch0)
            diffImageInfo2 = torch.where(diffImage < -thresh, torch1 , torch0)
            diffImageInfo1[x:x0,:] = 0
            diffImageInfo1[x0:x,:] = 0
            diffImageInfo1[x+imageShape0[0]:x0+imageShape0[0],:] = 0
            diffImageInfo1[x0+imageShape0[0]:x+imageShape0[0],:] = 0
            diffImageInfo1[:,y:y0] = 0
            diffImageInfo1[:,y0:y] = 0
            diffImageInfo1[:,y+imageShape0[1]:y0+imageShape0[1]] = 0
            diffImageInfo1[:,y0+imageShape0[1]:y+imageShape0[1]] = 0

            diffImageInfo2[x:x0,:] = 0
            diffImageInfo2[x0:x,:] = 0
            diffImageInfo2[x+imageShape0[0]:x0+imageShape0[0],:] = 0
            diffImageInfo2[x0+imageShape0[0]:x+imageShape0[0],:] = 0
            diffImageInfo2[:,y:y0] = 0
            diffImageInfo2[:,y0:y] = 0
            diffImageInfo2[:,y+imageShape0[1]:y0+imageShape0[1]] = 0
            diffImageInfo2[:,y0+imageShape0[1]:y+imageShape0[1]] = 0
            
            imageStore1 = torch.nonzero(diffImageInfo1,as_tuple=False)    
            x = imageStore1[:,0]
            y = imageStore1[:,1]
            good_xy = (x < 240) & (x > 16) & (y < 240) & (y > 16)
            imageStore1 = imageStore1[good_xy,:]
            timeStore1 = torch.zeros([imageStore1.size()[0],1],device=device).fill_(t)
            imageStore1 = torch.cat((imageStore1,timeStore1),1)
            imageStorep = torch.cat((imageStorep,imageStore1),0) 

            imageStore2 = torch.nonzero(diffImageInfo2,as_tuple=False)
            x = imageStore2[:,0]
            y = imageStore2[:,1]
            good_xy = (x < 240) & (x > 16) & (y < 240) & (y > 16)
            imageStore2 = imageStore2[good_xy,:]
            timeStore2 = torch.zeros([imageStore2.size()[0],1],device=device).fill_(t)
            imageStore2 = torch.cat((imageStore2,timeStore2),1)
            imageStoren = torch.cat((imageStoren,imageStore2),0) 
            
            elen+= (imageStore1.size()[0]+imageStore2.size()[0])

        lastImage = newImage

    if(elen<1000):#0.2%
        error_sample += 1
        continue        
    else:
        event_rate += elen
        f.write(filename +'.npz'+' '+ label + ' ' + str(a) + ' ' + str(b) +'\n')

        imageStorep = imageStorep.cpu().numpy().astype(np.uint8)
        imageStoren = imageStoren.cpu().numpy().astype(np.uint8)
        count += 1
        documentName = datapath + filename
        np.savez_compressed(documentName,pos = imageStorep, neg = imageStoren)

    if (count % 1000 == 0):
        print(count)
        print(time.time()-start) 
        print(error_sample) 
        timelist.append(time.time()-start)
        print('event_rate = ', event_rate/1000/(224*224*8*2)*100,'%')
        event_rate = 0
    
print(count)
print('total missing = ', error_sample)
end = time.time()
print("time=",end-start)
print('th=',thresh)
f.close()
print(timelist)