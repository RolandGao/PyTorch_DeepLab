from torch import nn
import torch
from torch.nn import functional as F

class XBlock(nn.Module): # From figure 4
    def __init__(self, in_channels, out_channels, bottleneck_ratio, group_width, stride):
        super(XBlock, self).__init__()
        inter_channels = out_channels // bottleneck_ratio
        groups = inter_channels // group_width

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None
        self.rl = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut=self.shortcut(x) if self.shortcut else x
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.rl(x + shortcut)
        return x


class VBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(VBlock, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None
        self.rl = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut=self.shortcut(x) if self.shortcut else x
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.rl(x + shortcut)
        return x

class AtrousConcat(nn.Module):
    def __init__(self, in_channels, out_channels, group_width,rates):
        super(AtrousConcat, self).__init__()
        modules = []
        groups = out_channels // group_width
        for rate in rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False,groups=groups),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.convs = nn.ModuleList(modules)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return res
class SpatialConcat(nn.Module):
    def __init__(self, in_channels, out_channels,group_width, bin_sizes):
        super(SpatialConcat, self).__init__()
        modules = []
        groups = out_channels // group_width
        self.reg_conv=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False,groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        for size in bin_sizes:
            modules.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(size,size)),
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False,groups=groups),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ))
        self.convs = nn.ModuleList(modules)

    def forward(self, x):
        input_shape = x.shape[-2:]
        res = [self.reg_conv(x)]
        for conv in self.convs:
            res.append(F.interpolate(conv(x), size=input_shape, mode='bilinear',align_corners=False))
        res = torch.cat(res, dim=1)
        return res
class AtrousSum(nn.Module):
    def __init__(self, in_channels, out_channels,group_width, rates):
        super(AtrousSum, self).__init__()
        modules = []
        groups = out_channels // group_width
        for rate in rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False,groups=groups),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.convs = nn.ModuleList(modules)

    def forward(self, x):
        out=self.convs[0](x)
        for conv in self.convs[1:]:
            out=out+conv(x)
        return out

class ABlock(nn.Module):
    def __init__(self, in_channels, conv1_channels,conv2_channels, out_channels,group_width,rates=(1,6,12,18),mode="concat"):
        super(ABlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, conv1_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(conv1_channels),
            nn.ReLU(inplace=True)
        )
        if mode=="concat":
            self.conv2=AtrousConcat(conv1_channels, conv2_channels, group_width,rates)
            conv2_out_channels=conv2_channels * len(rates)
        elif mode=="sum":
            self.conv2=AtrousSum(conv1_channels, conv2_channels, group_width,rates)
            conv2_out_channels=conv2_channels
        else:
            raise NotImplementedError()
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv2_out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut=None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.rl = nn.Sequential(
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        shortcut=self.shortcut(x) if self.shortcut else x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.rl(x + shortcut)
        return x

class SBlock(nn.Module):
    def __init__(self, in_channels, conv1_channels,conv2_channels, out_channels,group_width,bin_sizes=(1,2,3,6),mode="concat"):
        super(SBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, conv1_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(conv1_channels),
            nn.ReLU(inplace=True)
        )
        if mode=="concat":
            self.conv2=SpatialConcat(conv1_channels, conv2_channels, group_width,bin_sizes)
            conv2_out_channels=conv2_channels * (len(bin_sizes)+1)
        else:
            raise NotImplementedError()
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv2_out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut=None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.rl = nn.Sequential(
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        shortcut=self.shortcut(x) if self.shortcut else x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.rl(x + shortcut)
        return x

def profile(model,x,device,num_iter=15):
    import time
    model=model.to(device)
    model.eval()
    t1=time.time()
    for i in range(num_iter):
        y=model(x)
    t2=time.time()
    return (t2-t1)/num_iter

def profiler(models,x):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("warming up")
    profile(models[0],x, device,num_iter=15)
    for model in models:
        seconds=profile(model,x,device,num_iter=30)
        print(round(seconds,3))

def try_models():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    x=torch.randn(2,128,128,128)
    group_width=128
    w=128
    model1=ABlock(w,w,w,w,group_width,rates=(1,6))
    model2=SBlock(w,w,w,w,group_width,bin_sizes=(4,16))
    model3=XBlock(w,w,1,group_width,1)
    models=[model1,model2,model3]
    profiler(models,x)


if __name__=="__main__":
    try_models()
