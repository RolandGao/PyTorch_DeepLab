import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy

class AtrousSeparableConvolution(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True,add_norm=True):
        modules=[]
        modules.append(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation,
                      bias=(not add_norm), groups=in_channels))
        if add_norm:
            modules.append(nn.BatchNorm2d(in_channels))
            modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                      padding=0, bias=bias))
        super().__init__(*modules)


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes,output_stride):
        base_rates=[3,6,9]
        mul=32//output_stride
        rates=[x*mul for x in base_rates]
        super().__init__(
            ASPP(in_channels, rates),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
class DeepLabHeadNoASSP(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super().__init__(
            ASPP(in_channels, []),
            nn.Conv2d(256, num_classes, 1)
        )

def get_ASSP(in_channels,output_stride,output_channels=256):
    base_rates = [3, 6, 9]
    mul = 32 // output_stride
    rates = [x * mul for x in base_rates]
    return ASPP(in_channels, rates,output_channels)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        num_branches=2+len(rates)
        self.project = nn.Sequential(
            nn.Conv2d(num_branches * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

def convert_to_separable_conv(module,deep_copy=True):
    new_module=module
    if deep_copy:
        new_module = deepcopy(module)
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1 and module.groups == 1:
        new_module = AtrousSeparableConvolution(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.bias is not None)
    for name, child in new_module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child,deep_copy=False))
    return new_module

if __name__=='__main__':
    module=nn.Conv2d(5,5,1,bias=False)
    print(module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.bias is not None)
