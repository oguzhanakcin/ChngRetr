from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 :
        nn.init.xavier_uniform_(m.weight)
    elif classname.find("BatchNorm") != -1 :
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data, 0.01)

class Encoder(nn.Module):
    def __init__(self,ngpu):
        super(Encoder,self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input channel size = 3
            nn.Conv2d(3,32,5,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # state size 512,512,32
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size 256,256,64
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size 128,128,128
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size 64,64,256
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 512, 5, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size 32,32,512
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(512, 1024, 5, padding=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # state size 16,16,1024
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(1024, 2048, 5, padding=2),
            nn.BatchNorm2d(2048),
            nn.ReLU(True),
            # state size 8,8,2048

        )
    def forward(self,input):
        return self.main(input)

class Decoder(nn.Module):
    def __init__(self,ngpu):
        super(Decoder,self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input = 8,8,2048
            nn.Upsample(scale_factor=2),
            nn.Conv2d(2048,1024,5,padding=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # state size 16,16,1024
            nn.Upsample(scale_factor=2),
            nn.Conv2d(1024, 512, 5, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size 32,32,512
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size 64,64,256
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size 128,128,128
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size 256,256,64
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # state size 512,512,32
            nn.Conv2d(32, 3, 5,padding=2),
            nn.BatchNorm2d(3),
            nn.Tanh()
            # state size 512,512,3
        )
    def forward(self,input):
        return self.main(input)