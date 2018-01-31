
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from __future__ import print_function
import argparse
import csv
import os.path
import torch.nn.parallel
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image


# In[2]:


traindir = '../data/hymenoptera_data/train'
valdir = '../data/hymenoptera_data/val'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_loader = data.DataLoader(
        datasets.ImageFolder(traindir,
                         transforms.Compose([
                             transforms.RandomResizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             normalize,
                         ])),
                        batch_size=4,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True)

val_loader = data.DataLoader(
    datasets.ImageFolder(valdir,
                    transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ])),
                    batch_size=1,
                    shuffle=False,
                    num_workers=1,
                    pin_memory=False)


# In[3]:


# class resnet(nn.Module):
#     def __init__(self, resnet_model):
#         super(resnet, self).__init__()
#         self.resnet_model = resnet_model
#         self.num_ftrs = self.resnet_model.fc.in_features
#         print(self.num_ftrs)
#         self.last_layer = nn.Linear(self.num_ftrs,2)
        
#     def forward(self,x):
#         return self.last_layer(self.resnet_model(x))

    
net = models.resnet18(pretrained = True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs,2)
# net = resnet(model)


# In[4]:


# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     net = nn.DataParallel(net)

# if torch.cuda.is_available():
# net = nn.DataParallel(net).cuda()
net.cuda()


# In[5]:


import torch.optim as optim

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr = 0.002)


# In[6]:


# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     #print(inp)
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     #print(label)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated


# In[7]:


# dataiter = iter(train_loader)
# images, labels = dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images))
# plt.show()
# print(labels)


# In[8]:



for epoch in range(40):
    running_loss = 0.
    correct = 0
    length = 0
    uncorrect = 0
    for i, data in enumerate(train_loader,0) :
        images, label = data
        
        images, label = Variable(images.cuda()), Variable(label.cuda())
        
        optimizer.zero_grad()
        
        output = net(images)
        
        loss = criterion(output, label)
        
        loss.backward()
        
        optimizer.step()

        running_loss += loss.data[0]
        
        prediction = torch.max(output.data,1)[1]

        correct += torch.sum(label.data==prediction)
        uncorrect += torch.sum(label.data!=prediction)
    
    print('[epoch : %d] loss: %.6f acc : %.6f' % (epoch + 1, running_loss/200, 100*correct/(correct+uncorrect)))
    running_loss = 0.0


# In[9]:


net.eval()
correct = 0
uncorrect = 0
for i, data in enumerate(val_loader,0) :
    images, label = data
        
    images, label = Variable(images.cuda()), Variable(label.cuda())
        
    optimizer.zero_grad()
        
    output = net(images)
        
    loss = criterion(output, label)
        
    loss.backward()
        
    optimizer.step()
        
    prediction = torch.max(output.data,1)[1]

    correct += torch.sum(label.data==prediction)
    uncorrect += torch.sum(label.data!=prediction)
    
print('validation acc : %.6f' % (100*correct/(correct+uncorrect)))


# In[ ]:




