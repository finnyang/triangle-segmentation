import torch.nn as nn

class ConvBnRelu(nn.Module):

    def __init__(self, in_channels,out_channels):
        super(ConvBnRelu,self).__init__()
        self.conv=nn.Conv3d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.bn=nn.BatchNorm3d(out_channels)
        self.act=nn.ReLU(inplace=True)
    def forward(self,input):
        return self.act(self.bn(self.conv(input)))

class n_conv(nn.Module):
    def __init__(self,in_channels,out_channels,num_conv):
        super(n_conv,self).__init__()
        self.conv=nn.Conv3d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.bn=nn.BatchNorm3d(out_channels)
        self.act=nn.ReLU(inplace=True)
        layers=[]
        for i in range(num_conv):
            layers.append(ConvBnRelu(out_channels,out_channels))
        self.convs=nn.Sequential(*layers)

    def forward(self,input):
        x1=self.act(self.bn(self.conv(input)))
        x2=self.convs(x1)
        return x2
