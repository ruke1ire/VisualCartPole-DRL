############ NETWORK ##############
'''
For policy gradient, the network should input the raw pixels, and output the probabilities of choosing either 0 or 1
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self,h=60,w=135,device='cuda:0'):
        super().__init__()
        
        self.device = device
        
        input_channel = 2
        hidden_channel = 32#64
        hidden_channel2 = 32
        kernel_size = 5
        stride = 2
        
        self.base = nn.Sequential(
            nn.Conv2d(input_channel, hidden_channel, kernel_size = kernel_size, stride = stride),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(),
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size = kernel_size, stride = stride),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(),
            nn.Conv2d(hidden_channel, hidden_channel2, kernel_size = kernel_size, stride = stride),
            nn.BatchNorm2d(hidden_channel2),
            nn.ReLU()
        )
        
        def conv2d_size_out(size, kernel_size = kernel_size, stride = stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * hidden_channel2
        
        self.head = nn.Sequential(
            nn.Linear(linear_input_size, 24),
            nn.ReLU(),
            nn.Linear(24, 2)
        )

    def forward(self, x):
        #x = x.to(self.device)
        x = self.base(x)
        x = x.flatten(1)
        x = self.head(x)
        return F.softmax(x,dim = 1)
    
class QNetwork(nn.Module):
    def __init__(self,h=60,w=135,device='cuda:0'):
        super().__init__()
        
        self.device = device
        
        input_channel = 2
        hidden_channel = 128#64
        hidden_channel2 = 32
        kernel_size = 5
        stride = 2
        
        self.base = nn.Sequential(
            nn.Conv2d(input_channel, hidden_channel, kernel_size = kernel_size, stride = stride),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(),
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size = kernel_size, stride = stride),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(),
            nn.Conv2d(hidden_channel, hidden_channel2, kernel_size = kernel_size, stride = stride),
            nn.BatchNorm2d(hidden_channel2),
            nn.ReLU()
        )
        
        def conv2d_size_out(size, kernel_size = kernel_size, stride = stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * hidden_channel2
        
        self.head = nn.Sequential(
            nn.Linear(linear_input_size, 24),
            nn.ReLU(),
            nn.Linear(24, 2)
        )

    def forward(self, x):
        #x = x.to(self.device)
        x = self.base(x)
        x = x.flatten(1)
        x = self.head(x)
