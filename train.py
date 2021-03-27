import torch
import torch.nn as nn
import torch.optim as optim
import argparse,json
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

if __name__ = "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size" ,type=int,default=4)
    parser.add_argument("--num-epoch",type=int,default=50)
    parser.add_argument("--out-loc",type=str,default="./")
    parser.add_argument("--weights",type=str,default="./")
    parser.add_argument("--learn-rate",type=float,default=0.001)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--ngpu", type=int, default=1)
    parser.add_argument("--data-loc",type=str,default="./train.json")
    parser.add_argument("--init-weight",type=bool,default=True)
    parser.add_argument("--checkpoint",type=int,default=100)
    opt = parser.parse_args()
    print(opt)

    bsize = opt.batch_size
    ngpu = opt.ngpu
    num_epochs = opt.num_epoch
    lr = opt.learn_rate
    beta1 = opt.beta1
    data_loc = opt.data_loc
    weights_loc = opt.weights
    init_w = opt.init_weight
    c_numb = opt.checkpoint
    out_loc = opt.out_loc

    dataloader = create_dataloader(data_loc, bsize)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    p = torch.cuda.get_device_properties(device)
    print("(Device Name:", p.name, ", Total Memory:", p.total_memory / 1024 ** 2, "MB)")


    t1netE = Encoder(ngpu).to(device)
    t1netD = Decoder(ngpu).to(device)
    t2netE = Encoder(ngpu).to(device)
    t2netD = Decoder(ngpu).to(device)

    if init_w:
        t1netE.apply(init_weights)
        t1netD.apply(init_weights)
        t2netE.apply(init_weights)
        t2netD.apply(init_weights)
    else :
        t1netE.load_state_dict(torch.load(weights_loc+"t1Encoder.pt"))
        t1netD.load_state_dict(torch.load(weights_loc+"t1Decoder.pt"))
        t2netE.load_state_dict(torch.load(weights_loc + "t2Encoder.pt"))
        t2netD.load_state_dict(torch.load(weights_loc + "t2Decoder.pt"))

    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()
    optimizert1 = optim.Adam(list(t1netE.parameters()) + list(t1netD.parameters()), lr=lr, betas=(beta1, 0.99))
    optimizert2 = optim.Adam(list(t2netE.parameters()) + list(t2netD.parameters()), lr=lr, betas=(beta1, 0.99))
    imgs1 = torch.zeros((bsize, 3, 512, 512), dtype=torch.float).to(device)
    imgs2 = torch.zeros((bsize, 3, 512, 512), dtype=torch.float).to(device)
    print("Starting Training Loop...")
    cudnn.benchmark = True
    lossest1 = []
    lossest2 = []

    for epoch in range(num_epochs):
        pbar = enumerate(dataloader)
        pbar = tqdm(pbar, total=len(dataloader))

        for i, img_paths in pbar:
            if len(img_paths) != bsize:
                continue
            imgs1, imgs2 = read_augment_images(img_paths, bsize)
            imgs1 = torch.from_numpy(imgs1).to(dtype=torch.float).to(device).permute(3, 2, 0, 1)
            imgs2 = torch.from_numpy(imgs2).to(dtype=torch.float).to(device).permute(3, 2, 0, 1)


            t1netD.zero_grad()
            t1netE.zero_grad()
            t2netD.zero_grad()
            t2netE.zero_grad()

            ot1 = t1netD(t1netE(imgs1))
            ot2 = t2netD(t2netE(imgs1))

            t1loss1 = criterion1(ot1, imgs2)
            t1loss2 = criterion2(ot1, imgs2)
            t1loss = t1loss1 + t1loss2

            t2loss1 = criterion1(ot2, imgs1)
            t2loss2 = criterion2(ot2, imgs1)
            t2loss = t2loss1 + t2loss2

            lossest1.append("%10.4g"%t1loss)
            lossest2.append("%10.4g" % t2loss)

            t1loss.backward()
            t2loss.backward()

            optimizert1.step()
            optimizert2.step()

            mem = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ("%10s" * 2 + "%10.4g" * 3) % ("%g/%g" % (epoch, num_epochs - 1), mem, t1loss,t2loss, imgs1.shape[-1])
            pbar.set_description(s)

            if (i+1) % c_numb  == 0:
                torch.save(t1netE.state_dict(),out_loc+ "t1Encoder.pt")
                torch.save(t1netD.state_dict(),out_loc+ "t1Decoder.pt")
                torch.save(t2netE.state_dict(),out_loc+  "t1Encoder.pt")
                torch.save(t2netD.state_dict(),out_loc+  "t1Decoder.pt")
                with open(out_loc + "t1losses.json", "w") as losst1file:
                    json.dump(lossest1, losst1file, indent=4)
                with open(out_loc + "t2losses.json", "w") as losst2file:
                    json.dump(lossest2, losst2file, indent=4)


    print("Training Has Ended")
    torch.save(t1netE.state_dict(), out_loc + "t1Encoder.pt")
    torch.save(t1netD.state_dict(), out_loc + "t1Decoder.pt")
    torch.save(t2netE.state_dict(), out_loc + "t1Encoder.pt")
    torch.save(t2netD.state_dict(), out_loc + "t1Decoder.pt")
    with open(out_loc + "t1losses.json", "w") as losst1file:
        json.dump(lossest1, losst1file, indent=4)
    with open(out_loc + "t2losses.json", "w") as losst2file:
        json.dump(lossest2, losst2file, indent=4)
    fig,axs = plt.subplots(2,1)
    axs[0,0].plot(lossest1)
    axs[1,0].plot(lossest1)
    plt.show()