# import torchvision
# from torch import nn
# import torch
# from torch.nn import functional as F
# from math import log2
# from Deeplabv3 import DeepLabHead,DeepLabHeadNoASSP,get_ASSP,convert_to_separable_conv
# import timm
from model import *

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
        if self.se is not None:
            x = x * self.se(x)
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
            nn.ReLU(inplace=True)
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

class LightASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=(1,6,12,18)):
        super(LightASPP, self).__init__()
        modules = []
        for rate in rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
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

class ASPP2(nn.Module):
    def __init__(self, in_channels, intermediate_channels=512, out_channels=256,rates=(1,6,12,18),dropout=0.5):
        super(ASPP2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True)
        )
        conv2_out_channels=intermediate_channels//len(rates)
        self.conv2=LightASPP(intermediate_channels,conv2_out_channels,rates)

        self.conv3 = nn.Sequential(
            nn.Conv2d(conv2_out_channels * len(rates), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.rl = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        shortcut=self.shortcut(x) if self.shortcut else x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.rl(x + shortcut)
        return x

class Decoder(nn.Module):
    def __init__(self,in_channels,out_channels,version=1,group_width=16):
        super(Decoder,self).__init__()
        if version==0:
            self.decoder= nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        elif version==1:
            self.decoder= nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        elif version==2:
            self.decoder= nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        elif version==3:
            self.decoder=XBlock(in_channels,out_channels,1,group_width,1,None)
        elif version==4:
            self.decoder= nn.Sequential(
                XBlock(in_channels,out_channels,1,group_width,1,None),
                XBlock(out_channels,out_channels,1,group_width,1,None)
            )
        elif version==5:
            self.decoder=VBlock(in_channels,out_channels,1)
        else:
            raise NotImplementedError()
    def forward(self,x):
        return self.decoder(x)
def nearest_multiple(x,m):
    return int(round(x/m)*m)
class ExperimentalModel1(nn.Module):
    def __init__(self, name="mobilenetv2_100",num_classes=21,pretrained="",
                 pretrained_backbone=True,filter_multiplier=1.0,version=1):
        super(ExperimentalModel1,self).__init__()
        output_stride = 16
        m=16
        head_filters16 = 512#nearest_multiple(512*filter_multiplier,m)
        head_filters8 = int(64*filter_multiplier)
        head_filters4=int(32*filter_multiplier)
        decoder_filters16=nearest_multiple(256*filter_multiplier,m)
        decoder_filters8=nearest_multiple(192*filter_multiplier,m)
        decoder_filters4=nearest_multiple(128*filter_multiplier,m)
        try:
            self.backbone=timm.create_model(name, features_only=True,
                                            output_stride=output_stride, out_indices=(1,2,4),pretrained=pretrained_backbone and pretrained =="")
        except RuntimeError:
            print("no model")
            print(timm.list_models())
            raise RuntimeError()
        channels=self.backbone.feature_info.channels()
        self.head16=None#Decoder(channels[2],head_filters16,version=0)
        self.head8=torch.nn.Sequential(
            nn.Conv2d(channels[1], head_filters8, 1, bias=False),
            nn.BatchNorm2d(head_filters8),
            nn.ReLU(inplace=True))
        self.head4=torch.nn.Sequential(
            nn.Conv2d(channels[0], head_filters4, 1, bias=False),
            nn.BatchNorm2d(head_filters4),
            nn.ReLU(inplace=True))
        if self.head16 is None:
            head_filters16=channels[2]
        #self.decoder16=ASPP(head_filters16, [6,12,18],decoder_filters16,128,0.5)
        self.decoder16=ASPP2(head_filters16,512,decoder_filters16,(1,6,12,18),dropout=0.5)
        self.decoder8=Decoder(decoder_filters16+head_filters8,decoder_filters8,version=version)
        self.decoder4=Decoder(decoder_filters8+head_filters4,decoder_filters4,version=version)
        self.classifier=nn.Conv2d(decoder_filters4, num_classes, 1)
        if pretrained != "":
            dic = torch.load(pretrained, map_location='cpu')
            if type(dic)==dict:
                self.load_state_dict(dic['model'])
            else:
                self.load_state_dict(dic)

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        if self.head16:
            print("hello")
            x=self.head16(features[2])
        else:
            x=features[2]
        x=self.decoder16(x)
        x2=self.head8(features[1])
        size8=x2.shape[-2:]
        x = F.interpolate(x, size=size8, mode='bilinear',align_corners=False)
        x=torch.cat((x,x2),dim=1)
        x=self.decoder8(x)
        x2=self.head4(features[0])
        size4=x2.shape[-2:]
        x = F.interpolate(x, size=size4, mode='bilinear',align_corners=False)
        x=torch.cat((x,x2),dim=1)
        x=self.decoder4(x)
        x=self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear',align_corners=False)
        return x

class ExperimentalModel2(nn.Module):
    def __init__(self, name="mobilenetv2_100",num_classes=21,pretrained="",
                 pretrained_backbone=True,version=1):
        super(ExperimentalModel2,self).__init__()
        output_stride = 16
        #m=16
        #decoder_filters16=nearest_multiple(256*filter_multiplier,m)
        try:
            self.backbone=timm.create_model(name, features_only=True,
                                            output_stride=output_stride, out_indices=(4,),pretrained=pretrained_backbone and pretrained =="")
        except RuntimeError:
            print("no model")
            print(timm.list_models())
            raise RuntimeError()
        channels=self.backbone.feature_info.channels()

        if version==1:
            self.decoder16=ASPP2(channels[0],512,256,(1,6,12,18),dropout=0.5)
            self.classifier=nn.Conv2d(256, num_classes, 1)
        elif version==2:
            self.decoder16=ASPP2(channels[0],512,512,(1,6,12,18),dropout=0.5)
            self.classifier=nn.Conv2d(512, num_classes, 1)
        elif version==3:
            self.decoder16=ASPP2(channels[0],1024,256,(1,6,12,18),dropout=0.5)
            self.classifier=nn.Conv2d(256, num_classes, 1)
        elif version==4:
            self.decoder16=get_ASSP(channels[0], output_stride,256)
            self.classifier=nn.Conv2d(256, num_classes, 1)
        else:
            raise NotImplementedError()
        if pretrained != "":
            dic = torch.load(pretrained, map_location='cpu')
            if type(dic)==dict:
                self.load_state_dict(dic['model'])
            else:
                self.load_state_dict(dic)

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x=self.decoder16(features[0])
        x=self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear',align_corners=False)
        return x

def test_fast2():
    name='resnet50d'
    models=[
        Deeplab3P(name=name, num_classes=21,pretrained_backbone=False),
        ExperimentalModel1(name=name,num_classes=21,pretrained="",
                           pretrained_backbone=False,filter_multiplier=1.0,version=1),
        ExperimentalModel1(name=name,num_classes=21,pretrained="",
                           pretrained_backbone=False,filter_multiplier=1.0,version=2),
        ExperimentalModel1(name=name,num_classes=21,pretrained="",
                           pretrained_backbone=False,filter_multiplier=1.0,version=3),
        ExperimentalModel1(name=name,num_classes=21,pretrained="",
                           pretrained_backbone=False,filter_multiplier=1.0,version=5),
        Deeplab3P(name=name, num_classes=21,pretrained_backbone=False),
    ]
    profiler(models)

def experiment1():
    model=ExperimentalModel1(name="mobilenetv2_100",num_classes=21,pretrained="",
                             pretrained_backbone=False,filter_multiplier=1.0)
    x=torch.randn((2,3,128,128))
    y=model(x)
    print(y.shape)
def experiment2():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torchvision.models.resnet50()
    profile_warmup(model, device)

def memory_experiment():
    #name='resnet50d'
    name="regnety_040"
    models=[
        # ExperimentalModel2(name=name,num_classes=21,pretrained="",pretrained_backbone=False,version=1),
        # ExperimentalModel2(name=name,num_classes=21,pretrained="",pretrained_backbone=False,version=2),
        # ExperimentalModel2(name=name,num_classes=21,pretrained="",pretrained_backbone=False,version=3),
        # ExperimentalModel2(name=name,num_classes=21,pretrained="",pretrained_backbone=False,version=4),
        ExperimentalModel1(name=name,num_classes=21,pretrained="",
                           pretrained_backbone=False,filter_multiplier=1.0,version=1),
        ExperimentalModel1(name=name,num_classes=21,pretrained="",
                           pretrained_backbone=False,filter_multiplier=1.0,version=2),
        ExperimentalModel1(name=name,num_classes=21,pretrained="",
                           pretrained_backbone=False,filter_multiplier=1.0,version=3),
        ExperimentalModel1(name=name,num_classes=21,pretrained="",
                           pretrained_backbone=False,filter_multiplier=1.0,version=5),
        #Deeplab3(name=name, num_classes=21,pretrained_backbone=False,aspp=False),
        Deeplab3P(name=name, num_classes=21,pretrained_backbone=False),
    ]
    memory_test(models,device)

if __name__=='__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes=21
    #experiment1()
    #test_fast2()
    memory_experiment()

    # print(timm.list_models())
    # model=timm.create_model("regnety_040")
    # print(model)


# def gradient():
#     x=torch.ones((1,1,4,4),requires_grad=True)
#     y=F.adaptive_avg_pool2d(x,(2,2))
#     y=F.interpolate(x, size=(8,8), mode='bilinear',align_corners=False)
#     print(y)
#     loss=torch.sum(y)
#     loss.backward()
#     print(x.grad)
#
# class Yoho(nn.Module):
#     def __init__(self):
#         super(Yoho, self).__init__()
#     def forward(self,x):
#         return torch.abs(x)*0.5 #*1.83
#
# class LittleNet(nn.Module):
#     def __init__(self):
#         super(LittleNet, self).__init__()
#         k=3
#         p=(k-1)//2
#         n=20
#         for i in range(n):
#             self.add_module(f"{i}_conv",nn.Conv2d(10,10,k,padding=p,bias=False))
#             self.add_module(f"{i}_bn",nn.BatchNorm2d(10))
#             self.add_module(f"{i}_relu",nn.ReLU())
#             #self.add_module(f"{i}_relu",Yoho())
#     def forward(self,x):
#         for name,m in self.named_children():
#             if "relu" in name:
#                 a=torch.sum(x>=0)
#                 b=torch.sum(x<0)
#                 print(a,b)
#             x=m(x)
#             print(name,torch.mean(x),torch.std(x))
#         return x
# def gradient2():
#     #model=timm.create_model('mobilenetv2_100',features_only=True,out_indices=(4,),output_stride=16)
#     #model=Deeplab3P(name='mobilenetv2_100', num_classes=21,pretrained_backbone=False)
#     # pretrained_path='checkpoints/voc_mobilenetv2'
#     # model=Deeplab3P(name='mobilenetv2_100',num_classes=num_classes,pretrained=pretrained_path,sc=False).to(
#     #     device)
#     #model=torchvision.models.vgg16()
#     model=torchvision.models.resnet50()
#     model=LittleNet()
#     model.train()
#     print(model)
#     x=torch.randn((2,10,128,128))
#     y=model(x)
#     loss=torch.mean(y**2)
#     loss.backward()
#     for name,p in model.named_parameters():
#         if p.grad is not None:
#             if "conv" in name:
#                 print(name,float(torch.sum(torch.abs(p.grad))))
