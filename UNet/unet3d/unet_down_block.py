import torch.nn as nn
from unet3d_parts import ConvBnRelu,n_conv

class DownBlock(nn.Module):

    def __init__(self,in_channels,out_channels,num_conv):
        super(DownBlock,self).__init__()
        self.pool=nn.MaxPool3d(kernel_size=2)
        self.convs=n_conv(in_channels,out_channels,num_conv)

    def forward(self,input):
        x1=self.pool(input)
        x2=self.convs(x1)
        return x2
