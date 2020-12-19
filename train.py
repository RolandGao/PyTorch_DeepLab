from model import Deeplab3P
from data import get_cityscapes,get_pascal_voc
import datetime
import time

import torch
import torch.utils.data
from torch import nn
import numpy as np
import torch.nn.functional

class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu
    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
            acc_global.item() * 100,
            ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
            ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100)


def criterion2(inputs, target, w):
    return nn.functional.cross_entropy(inputs,target,ignore_index=255)
# def criterion(inputs, target, w):
#     losses = {}
#     for name, x in inputs.items():
#         losses[name] = nn.functional.cross_entropy(x, target, weight=w,ignore_index=255)
#
#     if len(losses) == 1:
#         return losses['out']
#
#     return losses['out'] + 0.5 * losses['aux']

# def mixup(x,device):
#     assert len(x.size())==4
#     batch_size = x.size()[0]
#     index = torch.randperm(batch_size).to(device)
#     lam=0.9
#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     return mixed_x


def evaluate(model, data_loader, device, num_classes,eval_steps):
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    with torch.no_grad():
        for i,(image, target) in enumerate(data_loader):
            if (i+1)%100==0:
                print(i+1)
            if i==eval_steps:
                break
            image, target = image.to(device), target.to(device)
            output = model(image)
            confmat.update(target.flatten(), output.argmax(1).flatten())

    return confmat

def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, max_iter,w):
    model.train()
    #np.random.seed()
    print(f"epoch: {epoch}")
    losses=0
    for t, x in enumerate(data_loader):
        image, target=x
        image, target = image.to(device), target.to(device)
        #image=mixup(image,device)
        output = model(image)
        loss = criterion(output, target, w)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        losses+=loss.item()
        if t % print_freq==0:
            print(t,loss.item())
        if t == max_iter-1:
            break
    num_iter=min(max_iter,len(data_loader))
    print(losses/num_iter)

def save(model,optimizer,scheduler,epoch,path):
    dic={
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': scheduler.state_dict(),
        'epoch': epoch
    }
    torch.save(dic,path)

def train(model, path, epochs,optimizer, data_loader, data_loader_test, lr_scheduler, device,num_classes, max_iter,w,eval_steps):
    start_time = time.time()
    best_mIU=0
    for epoch in epochs:
        train_one_epoch(model, criterion2, optimizer, data_loader, lr_scheduler,
                        device, epoch, print_freq=50, max_iter=max_iter,w=w)
        confmat = evaluate(model, data_loader_test, device=device,
                           num_classes=num_classes,eval_steps=eval_steps)
        print(confmat)
        acc_global, acc, iu = confmat.compute()
        mIU=iu.mean().item() * 100
        if mIU > best_mIU:
            best_mIU=mIU
            save(model, optimizer, lr_scheduler, epoch, path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def main():
    num_classes = 21
    batch_size=16
    epochs=30
    resume = True
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    data_loader, data_loader_test=get_pascal_voc("pascal_voc_dataset",batch_size)
    eval_steps=300
    resume_path = '/content/voc_regnetx40'
    save_path = '/content/drive/My Drive/voc_regnetx40'

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    epoch_start=0
    max_iter=len(data_loader)*1.0
    model=Deeplab3P(name='regnetx_040',num_classes=num_classes,pretrained_backbone=True,sc=False).to(device)
    params_to_optimize=model.parameters()
    optimizer = torch.optim.SGD(params_to_optimize, lr=lr,
                                momentum=momentum, weight_decay=weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,lambda x: (1 - x / (max_iter * epochs)) ** 0.9)
    if resume:
        dic=torch.load(resume_path,map_location='cpu')
        model.load_state_dict(dic['model'])
        optimizer.load_state_dict(dic['optimizer'])
        lr_scheduler.load_state_dict(dic['lr_scheduler'])
        epoch_start = dic['epoch'] + 1

    train(model, save_path, range(epoch_start,epochs),optimizer, data_loader,
          data_loader_test, lr_scheduler, device,num_classes, max_iter,None,eval_steps)

def check():
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 21
    pretrained_path='/content/drive/My Drive/Colab Notebooks/voc_50d'
    data_loader, data_loader_test=get_pascal_voc("pascal_voc_dataset",16)
    eval_steps = len(data_loader_test)
    model=Deeplab3P(name="resnet50d",num_classes=num_classes,pretrained=pretrained_path).to(
        device)
    print("evaluating")
    confmat = evaluate(model, data_loader_test, device=device,
                       num_classes=num_classes,eval_steps=eval_steps)
    print(confmat)
if __name__=='__main__':
    #check()
    main()
