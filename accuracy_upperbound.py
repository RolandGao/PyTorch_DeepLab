import torchvision
from torch import nn
import torch
from torch.nn import functional as F
from train import ConfusionMatrix
from data import get_cityscapes,get_pascal_voc

class DoNothingNet(nn.Module):
    def __init__(self,output_stride=16,mode="bilinear"):
        super(DoNothingNet,self).__init__()
        self.os=output_stride
        self.mode=mode
    def forward(self,y):
        shape = y.shape[-2:]
        downsample_shape=((shape[0]-1)//self.os+1,(shape[1]-1)//self.os+1)
        if self.mode=="bilinear":
            y=F.interpolate(y,size=downsample_shape,mode='bilinear',align_corners=False)
        elif self.mode=="adaptive_avg":
            y=F.adaptive_avg_pool2d(y,downsample_shape)
        elif self.mode=="adaptive_max":
            y=F.adaptive_max_pool2d(y,downsample_shape)
        elif self.mode=="max3x3":
            y=F.max_pool2d(y,kernel_size=3,stride=2,padding=1)
        elif self.mode=="avg3x3":
            y=F.avg_pool2d(y,kernel_size=3,stride=2,padding=1)
        else:
            raise NotImplementedError()
        y=F.interpolate(y,size=shape,mode='bilinear',align_corners=False)
        return y

def evaluate(model, data_loader, device, num_classes,eval_steps,print_every=100):
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    with torch.no_grad():
        for i,(image, target) in enumerate(data_loader):
            if (i+1)%print_every==0:
                print(i+1)
            if i==eval_steps:
                break
            target = target.to(device)
            logits=torch.zeros(target.shape[0],num_classes,target.shape[1],target.shape[2],device=device)
            target2=torch.unsqueeze(target,1)
            target2[target2==255]=0
            logits.scatter_(1,target2,1)
            output = model(logits)
            confmat.update(target.flatten(), output.argmax(1).flatten())
    return confmat

def f(os,mode,device,num_classes):
    net=DoNothingNet(output_stride=os,mode=mode).to(device)
    #data_loader, data_loader_test=get_pascal_voc("pascal_voc_dataset",16,train_size=481,val_size=513)
    data_loader, data_loader_test=get_cityscapes("cityscapes_dataset",16,train_size=480,val_size=1024,num_workers=0)
    confmat=evaluate(net,data_loader_test,device,num_classes,eval_steps=100,print_every=20)
    return confmat

def experiment1():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes=19
    print()
    for mode in ["adaptive_avg"]:#"bilinear","adaptive_max","max3x3",
        for os in [2,4,8,16,32]:
            confmat=f(os,mode,device,num_classes)
            # net=DoNothingNet(output_stride=os,mode=mode).to(device)
            # #data_loader, data_loader_test=get_pascal_voc("pascal_voc_dataset",16,train_size=481,val_size=513)
            # data_loader, data_loader_test=get_cityscapes("cityscapes_dataset",16,train_size=480,val_size=1024)
            # confmat=evaluate(net,data_loader_test,device,num_classes,eval_steps=300,print_every=100)
            print(mode,os)
            print(confmat)

if __name__=="__main__":
    experiment1()
