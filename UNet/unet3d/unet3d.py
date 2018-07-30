import torch.nn as nn
import torch
import numpy as np
from unet_down_block import DownBlock
from unet3d_parts import ConvBnRelu,n_conv
from unet_up_block import UpBlock

class net(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(net,self).__init__()
        self.n_convs=1
        self.inconv1=ConvBnRelu(in_channels,64)
        self.inconv2=ConvBnRelu(64,64)
        self.down1=DownBlock(64,128,self.n_convs)
        self.down2=DownBlock(128,256,self.n_convs)
        self.down3=DownBlock(256,512,self.n_convs)
        self.down4=DownBlock(512,512,self.n_convs)
        self.up4=UpBlock(1024,256,self.n_convs)
        self.up3=UpBlock(512, 128, self.n_convs)
        self.up2=UpBlock(256, 64, self.n_convs)
        self.up1=UpBlock(128, 64, self.n_convs)
        self.out=nn.Conv3d(64,out_channels,kernel_size=1,stride=1)

    def forward(self,input):
        x1=self.inconv1(input)
        x2=self.inconv2(x1)
        x3=self.down1(x2)
        x4=self.down2(x3)
        x5=self.down3(x4)
        x6=self.down4(x5)
        x=self.up4(x6,x5)
        x=self.up3(x,x4)
        x=self.up2(x,x3)
        x=self.up1(x,x2)
        x=self.out(x)
        return x

input_num=np.zeros((1,1,16,96,96))
input=torch.from_numpy(input_num).float()
unet=net(1,1)
output=unet.forward(input)
print output.size()