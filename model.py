import torchvision
from torch import nn
import torch
from torch.nn import functional as F
from math import log2
from Deeplabv3 import DeepLabHead,DeepLabHeadNoASSP,get_ASSP,convert_to_separable_conv, ASPP
import timm

def replace(output_stride=8):
    d=[False,False,False]
    n=int(log2(32/output_stride))
    assert n<=3,'output_stride too small'
    for i in range(n):
        d[2-i]=True
    return d

class Deeplab3(nn.Module):
    def __init__(self,name="mobilenetv2_100",num_classes=21,pretrained="",
                 pretrained_backbone=True,aspp=True):
        super(Deeplab3,self).__init__()
        output_stride = 16
        self.backbone=timm.create_model(name, features_only=True,
                                        output_stride=output_stride, out_indices=(4,),pretrained=pretrained_backbone and pretrained =="")
        channels=self.backbone.feature_info.channels()
        if aspp:
            self.head=DeepLabHead(channels[0], num_classes,output_stride)
        else:
            self.head=DeepLabHeadNoASSP(channels[0], num_classes)
        if pretrained != "":
            dic = torch.load(pretrained, map_location='cpu')
            if type(dic)==dict:
                self.load_state_dict(dic['model'])
            else:
                self.load_state_dict(dic)
    def forward(self,x):
        input_shape = x.shape[-2:]
        x=self.backbone(x)
        x=self.head(x[0])
        x = F.interpolate(x, size=input_shape, mode='bilinear',align_corners=False)
        return x

class Deeplab3P(nn.Module):
    def __init__(self, name="mobilenetv2_100",num_classes=21,pretrained="",
                 pretrained_backbone=True,sc=False,filter_multiplier=1.0):
        super(Deeplab3P,self).__init__()
        output_stride = 16
        num_filters = int(256*filter_multiplier)
        num_low_filters = int(48*filter_multiplier)
        try:
            self.backbone=timm.create_model(name, features_only=True,
                                            output_stride=output_stride, out_indices=(1, 4),pretrained=pretrained_backbone and pretrained =="")
        except RuntimeError:
            print("no model")
            print(timm.list_models())
            raise RuntimeError()
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
        if sc:
            self.decoder = convert_to_separable_conv(self.decoder)
        if pretrained != "":
            dic = torch.load(pretrained, map_location='cpu')
            if type(dic)==dict:
                self.load_state_dict(dic['model'])
            else:
                self.load_state_dict(dic)

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

def profile(model,device,num_iter=15):
    import time
    model=model.to(device)
    model.eval()
    x=torch.randn(2,3,321,321).to(device)
    t1=time.time()
    for i in range(num_iter):
        y=model(x)
    t2=time.time()
    return (t2-t1)/num_iter
def profile_warmup(model,device):
    import time
    num_iter=0
    model=model.to(device)
    model.eval()
    x=torch.randn(2,3,321,321).to(device)
    t1=time.time()
    for _ in range(100):
        num_iter+=1
        y=model(x)
        t2=time.time()
        average_time=(t2-t1)/num_iter
        print(num_iter,average_time)

def profiler(models):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torchvision.models.resnet50()
    print("warming up")
    profile(model, device,num_iter=15)
    for model in models:
        seconds=profile(model,device,num_iter=30)
        print(round(seconds,3))

def total_params(models):
    for model in models:
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total=round(total/1000000,2)
        print(f"{model.__class__.__name__}: {total}M")

def profiler2(models):
    from ptflops import get_model_complexity_info
    for model in models:
        macs, params = get_model_complexity_info(model, (3, 480, 480),
                                                 as_strings=True,
                                                 print_per_layer_stat=False,
                                                 verbose=False)
        print(f"{model.__class__.__name__}: {macs}, {params}")

def memory_used(device):
    x=torch.cuda.memory_allocated(device)
    return round(x/1024/1024,4)
def max_memory_used(device):
    x=torch.cuda.max_memory_allocated(device)
    return round(x/1024/1024,4)
def memory_test_helper(model,device):
    model=model.to(device)
    model.train()
    N=16
    x=torch.randn(N, 3, 481, 481).to(device)
    target=torch.randint(0,21,(N, 481, 481)).to(device)
    t1=memory_used(device)
    out=model(x)
    loss=nn.functional.cross_entropy(out,target,ignore_index=255)
    loss.backward()
    t2=max_memory_used(device)
    print(t2-t1)
    torch.cuda.reset_peak_memory_stats(device)

def memory_test(models,device):
    for i in range(len(models)):
        try:
            memory_test_helper(models[0],device)
            print()
        except:
            print("out of memory")
        for p in models[0].parameters():
            p.grad=None
        del models[0]

def test_separable():
    model1=nn.Sequential(nn.Conv2d(3,256, 1, padding=0, bias=False),nn.BatchNorm2d(256),nn.ReLU(inplace=True),
                         nn.Conv2d(256,256, 3, padding=1, bias=False),nn.BatchNorm2d(256),nn.ReLU(inplace=True))
    model2=nn.Sequential(nn.Conv2d(3,256, 1, padding=0, bias=False),nn.BatchNorm2d(256),nn.ReLU(inplace=True),
                         nn.Conv2d(256,256, 3, padding=1, bias=False),nn.BatchNorm2d(256),nn.ReLU(inplace=True))
    model2=convert_to_separable_conv(model2)
    model3=nn.Sequential(nn.Conv2d(3,256, 1, padding=0, bias=False),nn.BatchNorm2d(256),nn.ReLU(inplace=True),
                         nn.Conv2d(256,256, 3, padding=2, bias=False,dilation=2),nn.BatchNorm2d(256),nn.ReLU(inplace=True))
    model4=nn.Sequential(nn.Conv2d(3,256, 1, padding=0, bias=False),nn.BatchNorm2d(256),nn.ReLU(inplace=True),
                         nn.Conv2d(256,256, 3, padding=2, bias=False,dilation=2),nn.BatchNorm2d(256),nn.ReLU(inplace=True))
    model4=convert_to_separable_conv(model4)
    models=[
        model1,model2,model3,model4
    ]
    profiler(models)
def test_fast():
    models=[
        Deeplab3P(name='mobilenetv2_100', num_classes=21,pretrained_backbone=False),
        #Deeplab3P(name='mobilenetv2_100', num_classes=21,pretrained_backbone=False,filter_multiplier=0.5),
        Deeplab3(name='mobilenetv2_100', num_classes=21,pretrained_backbone=False),
        Deeplab3(name='mobilenetv2_100', num_classes=21,pretrained_backbone=False,aspp=False),
        Deeplab3P(name='resnet50d', num_classes=21,pretrained_backbone=False),
        #Deeplab3P(name='resnet50d', num_classes=21,pretrained_backbone=False,filter_multiplier=0.5),
        Deeplab3(name='resnet50d', num_classes=21,pretrained_backbone=False),
        Deeplab3(name='resnet50d', num_classes=21,pretrained_backbone=False,aspp=False),
    ]
    profiler(models)

def test_models():
    # names = [
    #     'mnasnet_a1',
    #     'mobilenetv2_100',
    #     'mobilenetv3_large_100',
    #     'mobilenetv3_small_100',
    #     'regnety_006',
    #     'regnety_004'
    # ]
    names=[
        'resnet50d',
        'regnety_040',
        'ecaresnet50d',
        'ecaresnet50',
        'seresnet50tn'
    ]
    models = []
    for name in names:
        #models.append(Deeplab3P(name=name, num_classes=21,pretrained_backbone=False))
        models.append(timm.create_model(name, features_only=True,
                                        output_stride=16, out_indices=(4,)))
    profiler(models)

if __name__=='__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #resnet50d 77.1 mIOU
    #regnetx_040 77.0 mIOU
    #regnety_040 78.6 mIOU
    #mobilenetv2_100 72.8 mIOU
    #regnetx_080 77.3 mIOU

    num_classes=21
    print(timm.list_models())
    #experiment1()
    test_fast()




# 0.897
# 0.752
# 0.614
# 0.371
# 0.251

# regnety4G
# 0.764
# 0.691
# 0.572
# 0.289
# 0.131

# regnet
#     1.016
# 0.946
# 0.853
# 0.49
# 0.181


# 'mobilenetv2_100'
#     0.34
# 0.178
# 0.167
# 0.168
# 0.167
