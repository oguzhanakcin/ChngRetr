from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import time
import os
import copy
import cv2
import time
from utils.dataload import create_dataloader, read_augment_images
from utils.model import Encoder, Decoder, init_weights
from tqdm import tqdm

bsize = 2
ngpu = 1
num_epochs = 1
lr = 0.001
beta1 = 0.5
k = 0
t0 = time.time()

dataloader = create_dataloader("./test.json", bsize)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

p = torch.cuda.get_device_properties(device)
print("(Device Name:",p.name,", Total Memory:",p.total_memory/1024**2,"MB)")

netE = Encoder(ngpu).to(device)
netE.apply(init_weights)
# netE.load_state_dict(torch.load("Encoder.pt"))

netD = Decoder(ngpu).to(device)
netD.apply(init_weights)
# netD.load_state_dict(torch.load("Decoder.pt"))


criterion1 = nn.MSELoss()
criterion2 = nn.L1Loss()
optimizer = optim.Adam(list(netE.parameters()) + list(netD.parameters()), lr=lr, betas=(beta1, 0.99))

imgs1 = torch.zeros((bsize, 3, 512, 512), dtype=torch.float).to(device)
imgs2 = torch.zeros((bsize, 3, 512, 512), dtype=torch.float).to(device)

print("Starting Training Loop...")
cudnn.benchmark = True

for epoch in range(num_epochs):

    pbar = enumerate(dataloader)
    pbar = tqdm(pbar,total=len(dataloader))

    for i,img_paths in pbar:
        if len(img_paths) != bsize:
            continue
        imgs1, imgs2 = read_augment_images(img_paths, bsize)
        imgs1 = torch.from_numpy(imgs1).to(dtype=torch.float).to(device).permute(3, 2, 0, 1)
        imgs2 = torch.from_numpy(imgs2).to(dtype=torch.float).to(device).permute(3, 2, 0, 1)

        netD.zero_grad()
        netE.zero_grad()
        oimgs2 = netD(netE(imgs1))
        loss1 = criterion1(oimgs2, imgs2)
        loss2 = criterion2(oimgs2, imgs2)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        mem = "%.3gG" % (torch.cuda.memory_reserved()/1E9 if torch.cuda.is_available() else 0)
        s = ("%10s"*2 + "%10.4g"*2) % ("%g/%g" % (epoch,num_epochs-1),mem,loss,imgs1.shape[-1])
        pbar.set_description(s)

torch.save(netE.state_dict(), "Encoder.pt")
torch.save(netD.state_dict(), "Decoder.pt")