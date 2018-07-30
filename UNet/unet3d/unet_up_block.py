import torch.nn as nn
from unet3d_parts import ConvBnRelu,n_conv
import torch.nn.functional as F
import torch

class UpBlock(nn.Module):

    def __init__(self,in_channels,out_channels,num_conv):
        super(UpBlock,self).__init__()
        self.up = nn.ConvTranspose3d(in_channels/2, in_channels/2,kernel_size=2,stride=2)
        self.convs=n_conv(in_channels,out_channels,num_conv)

    def forward(self,input1,input2):
        x1=self.up(input1)
        dx = x1.size()[3] - input2.size()[3]
        dy = x1.size()[4] - input2.size()[4]
        x2 = F.pad(input2, (dx // 2, int(dx / 2), dy // 2, int(dy / 2)))
        x3 = torch.cat([x2, x1], dim=1)
        x4 = self.convs(x3)
        return x4