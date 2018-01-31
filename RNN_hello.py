
# coding: utf-8

# In[10]:


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
import torch.optim as optim
import sys


# In[11]:


torch.manual_seed(777)  # reproducibility

idx2char = ['h', 'i', 'e' , 'l' , 'o']
x_data = [0,1,0,2,3,3] # hihell
# x_one_hot = [[[1,0,0,0,0],
#              [0,1,0,0,0],
#              [1,0,0,0,0],
#              [0,0,0,1,0],
#              [0,0,0,0,1],
#              [0,0,0,0,1]]]
one_hot = [[1,0,0,0,0],
           [0,1,0,0,0],
           [0,0,1,0,0],
           [0,0,0,1,0],
           [0,0,0,0,1]]

x_one_hot = [one_hot[i] for i in x_data]

y_data = [1,0,2,3,3,4] # ihello

inputs = Variable(torch.Tensor(x_one_hot))
labels = Variable(torch.LongTensor(y_data))


# In[12]:


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.rnn = nn.RNN(input_size = 5, hidden_size = 5, batch_first = True)
    def forward(self,hidden, x):
        x = x.view(1,1,5) # batch_size, sequence_length, input_size
        
        out , hidden = self.rnn(x,hidden)
        out = out.view(-1, 5) # (-1, num_classes)
        return hidden , out
    
    def init_hidden(self):
        #num_layers * num_directions, batch, hidden_size
        return Variable(torch.zeros(1,1,5))


# In[13]:


model = Model()
print(model)

criterion= nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.1)

for epoch in range(100):
    optimizer.zero_grad()
    loss = 0
    hidden = model.init_hidden()
    sys.stdout.write("Predicted String : ")
    
    for inp, label in zip(inputs, labels):
        hidden, output = model(hidden, inp)
        val, idx = output.max(1)
        sys.stdout.write(idx2char[idx.data[0]])
#         print(output, label)
        loss += criterion(output, label)
    
    print(", epoch : %d, loss : %1.3f" % (epoch +1, loss.data[0]))
    
    loss.backward()
    optimizer.step()


# In[ ]:




