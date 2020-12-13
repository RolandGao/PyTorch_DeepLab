import torchvision
from torch import nn
import torch
from torch.nn import functional as F
from math import log2
from Deeplabv3 import DeepLabHead,DeepLabHeadNoASSP,get_ASSP,convert_to_separable_conv
import timm

def replace(output_stride=8):
    d=[False,False,False]
    n=int(log2(32/output_stride))
    assert n<=3,'output_stride too small'
    for i in range(n):
        d[2-i]=True
    return d

class Deeplab3P(nn.Module):
    def __init__(self, name="mobilenetv2_100",num_classes=21,pretrained="",pretrained_backbone=True):
        super(Deeplab3P,self).__init__()
        output_stride = 16
        num_filters = 256
        num_low_filters = 48
        self.backbone=timm.create_model(name, features_only=True,
                          output_stride=output_stride, out_indices=(1, 4),pretrained=pretrained_backbone)
        channels=self.backbone.feature_info.channels()
        self.head16=get_ASSP(channels[1], output_stride,num_filters)
        self.head4=torch.nn.Sequential(
        nn.Conv2d(channels[0], num_low_filters, 1, bias=False),
        nn.BatchNorm2d(num_low_filters),
        nn.ReLU(inplace=True))
        self.decoder= nn.Sequential(
        nn.Conv2d(num_low_filters+num_filters, num_filters, 3, padding=1, bias=False),
        nn.BatchNorm2d(num_filters),
        nn.ReLU(inplace=True),
        nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False),
        nn.BatchNorm2d(num_filters),
        nn.ReLU(inplace=True),
        nn.Conv2d(num_filters, num_classes, 1)
        )
        self.decoder = convert_to_separable_conv(self.decoder)
        if pretrained != "":
            dic = torch.load(pretrained, map_location='cpu')
            self.load_state_dict(dic['model'])

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x=self.head16(features[1])
        x2=self.head4(features[0])
        intermediate_shape=x2.shape[-2:]
        x = F.interpolate(x, size=intermediate_shape, mode='bilinear',align_corners=False)
        x=torch.cat((x,x2),dim=1)
        x=self.decoder(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear',align_corners=False)
        return x

def profile(model,device):
    import time
    num_iter=30
    model=model.to(device)
    model.eval()
    x=torch.randn(1,3,480,480).to(device)
    t1=time.time()
    for i in range(num_iter):
        y=model(x)
    t2=time.time()
    return (t2-t1)/num_iter

def profiler(models):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torchvision.models.resnet101()
    print("warming up")
    profile(model, device)
    for model in models:
        seconds=profile(model,device)
        print(round(seconds,3))

def total_params(models):
    for model in models:
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total=round(total/1000000,2)
        print(f"{model.__class__.__name__}: {total}M")

# def profiler2(models):
#     from ptflops import get_model_complexity_info
#     for model in models:
#         macs, params = get_model_complexity_info(model, (3, 480, 480),
#                                                  as_strings=True,
#                                                  print_per_layer_stat=False,
#                                                  verbose=False)
#         print(f"{model.__class__.__name__}: {macs}, {params}")
if __name__=='__main__':
    #50d checkpoint 77.1 mIOU
    num_classes=21
    print(timm.list_models())
    model = Deeplab3P(name='regnetx_040', num_classes=19, pretrained="",pretrained_backbone=False)
    names = [
        'resnet50',
        'resnet50d',
        'resnest50d',
        'mobilenetv2_100',
        'regnetx_032',
        'regnetx_040'
    ]
    models = []
    for name in names:
        models.append(Deeplab3P(name=name, num_classes=21, pretrained="",pretrained_backbone=False))
        #models.append(timm.create_model(name, features_only=True))
    profiler(models)
