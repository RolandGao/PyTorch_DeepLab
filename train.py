from model import Deeplab3P
from benchmark import compute_time_full
from data import get_cityscapes,get_pascal_voc,get_coco
import datetime
import time

import torch
import torch.utils.data
from torch import nn
import torch.nn.functional
import yaml
import torch.cuda.amp as amp

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
    return nn.functional.cross_entropy(inputs,target,ignore_index=255,weight=w)

def get_loss_fun(weight):
    return nn.CrossEntropyLoss(weight=weight,ignore_index=255)

def evaluate(model, data_loader, device, num_classes,eval_steps,mixed_precision,print_every=100):
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    with torch.no_grad():
        for i,(image, target) in enumerate(data_loader):
            if (i+1)%print_every==0:
                print(i+1)
            if i==eval_steps:
                break
            image, target = image.to(device), target.to(device)
            with amp.autocast(enabled=mixed_precision):
                output = model(image)
            confmat.update(target.flatten(), output.argmax(1).flatten())

    return confmat

def train_one_epoch(model, loss_fun, optimizer, loader, lr_scheduler, device, print_freq,mixed_precision,scaler):
    model.train()
    losses=0
    for t, x in enumerate(loader):
        image, target=x
        image, target = image.to(device), target.to(device)
        with amp.autocast(enabled=mixed_precision):
            output = model(image)
            loss = loss_fun(output, target)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        losses+=loss.item()
        if t % print_freq==0:
            print(t,loss.item())
    num_iter=len(loader)
    print(losses/num_iter)

def save(model,optimizer,scheduler,epoch,path,best_mIU,scaler):
    dic={
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': scheduler.state_dict(),
        'scaler':scaler.state_dict(),
        'epoch': epoch,
        'best_mIU':best_mIU
    }
    torch.save(dic,path)

def train(model, save_best_path,save_latest_path, epochs,optimizer, data_loader, data_loader_test, lr_scheduler, device,num_classes,save_best_on_epochs,loss_fun,mixed_precision,scaler,best_mIU):
    start_time = time.time()
    for epoch in epochs:
        print(f"epoch: {epoch}")
        train_one_epoch(model, loss_fun, optimizer, data_loader, lr_scheduler,
                        device, print_freq=50,mixed_precision=mixed_precision,scaler=scaler)
        if epoch in save_best_on_epochs:
            confmat = evaluate(model, data_loader_test, device=device,
                               num_classes=num_classes,eval_steps=len(data_loader_test),mixed_precision=mixed_precision,print_every=100)
            print(confmat)
            acc_global, acc, iu = confmat.compute()
            mIU=iu.mean().item() * 100
            if mIU > best_mIU:
                best_mIU=mIU
                save(model, optimizer, lr_scheduler, epoch, save_best_path,best_mIU,scaler)
        if save_latest_path != "":
            save(model, optimizer, lr_scheduler, epoch, save_latest_path,best_mIU,scaler)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

# def main():
#     num_classes = 21
#     batch_size=16
#     epochs=10
#     resume = False
#     lr = 0.004
#     momentum = 0.9
#     weight_decay = 1e-4
#     data_loader, data_loader_test=get_pascal_voc("pascal_voc_dataset",batch_size,train_size=481,val_size=513)
#     eval_steps=1000
#     class_weight=None
#     mixed_precision=False
#     resume_path = '/content/drive/My Drive/Colab Notebooks/SemanticSegmentation/checkpoints/voc_50d'
#     save_path = '/content/drive/My Drive/Colab Notebooks/SemanticSegmentation/checkpoints/voc_resnet50d_noise2'
#
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     epoch_start=0
#     model=Deeplab3P(name='resnet50d',num_classes=num_classes,pretrained_backbone=True,sc=True,pretrained=resume_path).to(device)
#     params_to_optimize=model.parameters()
#     optimizer = torch.optim.SGD(params_to_optimize, lr=lr,
#                                 momentum=momentum, weight_decay=weight_decay)
#     scaler = amp.GradScaler(enabled=mixed_precision)
#     loss_fun=get_loss_fun(class_weight).to(device)
#     lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
#         optimizer,lambda x: (1 - x / (len(data_loader) * epochs)) ** 0.9)
#     if resume:
#         dic=torch.load(resume_path,map_location='cpu')
#         model.load_state_dict(dic['model'])
#         optimizer.load_state_dict(dic['optimizer'])
#         lr_scheduler.load_state_dict(dic['lr_scheduler'])
#         epoch_start = dic['epoch'] + 1
#
#     train(model, save_path, range(epoch_start,epochs),optimizer, data_loader,
#           data_loader_test, lr_scheduler, device,num_classes,eval_steps,loss_fun,mixed_precision,scaler)

def get_dataset_loaders(config):
    name=config["dataset_name"]
    if name=="pascal_voc":
        f=get_pascal_voc
    elif name=="cityscapes":
        f=get_cityscapes
    elif name=="coco":
        f=get_coco
    else:
        raise NotImplementedError()
    mode="baseline"
    if "aug_mode" in config:
        mode=config["aug_mode"]
    data_loader, data_loader_test=f(config["dataset_dir"],config["batch_size"],train_size=config["train_size"],val_size=config["val_size"],mode=mode)
    return data_loader, data_loader_test

def get_model(config):
    return Deeplab3P(name=config["model_name"],
                     num_classes=config["num_classes"],
                     pretrained_backbone=config["pretrained_backbone"],
                     sc=config["separable_convolution"],
                     pretrained=config["pretrained_path"])

def main2(config_filename):
    with open(config_filename) as file:
        config=yaml.full_load(file)
        print(config)
    save_best_path=config["save_best_path"]
    save_latest_path=config["save_latest_path"]
    epochs=config["epochs"]
    num_classes=config["num_classes"]
    # eval_steps=config["eval_steps"]
    class_weight=config["class_weight"]
    mixed_precision=config["mixed_precision"]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data_loader, data_loader_test=get_dataset_loaders(config)
    model=get_model(config).to(device)
    params_to_optimize=model.parameters()
    optimizer = torch.optim.SGD(params_to_optimize, lr=config["lr"],
                            momentum=config["momentum"], weight_decay=config["weight_decay"])
    scaler = amp.GradScaler(enabled=mixed_precision)
    loss_fun=get_loss_fun(class_weight)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,lambda x: (1 - x / (len(data_loader) * epochs)) ** 0.9)

    epoch_start=0
    best_mIU=0
    save_every_k_epochs=config["save_every_k_epochs"]
    save_best_on_epochs=[i*save_every_k_epochs-1 for i in range(1,epochs//save_every_k_epochs+1)]
    if epochs-1 not in save_best_on_epochs:
        save_best_on_epochs.append(epochs-1)
    print("save on epochs: ",save_best_on_epochs)

    if config["resume"]:
        dic=torch.load(config["resume_path"],map_location='cpu')
        model.load_state_dict(dic['model'])
        optimizer.load_state_dict(dic['optimizer'])
        lr_scheduler.load_state_dict(dic['lr_scheduler'])
        epoch_start = dic['epoch'] + 1
        if "best_mIU" in dic:
            best_mIU=dic["best_mIU"]
        if "scaler" in dic:
            scaler.load_state_dict(dic[scaler])

    train(model, save_best_path,save_latest_path, range(epoch_start,epochs),optimizer, data_loader,
          data_loader_test, lr_scheduler, device,num_classes,save_best_on_epochs,loss_fun,mixed_precision,scaler,best_mIU)
def check3(config_filename):
    with open(config_filename) as file:
        config=yaml.full_load(file)
        print(config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data_loader, data_loader_test=get_dataset_loaders(config)
    model=get_model(config).to(device)
    num_classes=config["num_classes"]
    eval_steps = len(data_loader_test)
    mixed_precision=config["mixed_precision"]
    print("evaluating")
    confmat = evaluate(model, data_loader_test, device=device,
                       num_classes=num_classes,eval_steps=eval_steps,mixed_precision=mixed_precision)
    print(confmat)

def check():
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 21
    pretrained_path='/content/drive/My Drive/Colab Notebooks/SemanticSegmentation/checkpoints/voc_resnet50d_noise2'
    #voc_resnet50d_noise
    data_loader, data_loader_test=get_pascal_voc("pascal_voc_dataset",16,train_size=481,val_size=513)
    eval_steps = len(data_loader_test)
    model=Deeplab3P(name="resnet50d",num_classes=num_classes,pretrained=pretrained_path,sc=True).to(
        device)
    print("evaluating")
    confmat = evaluate(model, data_loader_test, device=device,
                       num_classes=num_classes,eval_steps=eval_steps,print_every=100)
    print(confmat)
def check2():
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 21
    pretrained_path='checkpoints/voc_mobilenetv2'
    #pretrained_path='checkpoints/voc_regnety40'
    data_loader, data_loader_test=get_pascal_voc("pascal_voc_dataset",16,train_size=385,val_size=385)
    eval_steps = 100
    model=Deeplab3P(name='mobilenetv2_100',num_classes=num_classes,pretrained=pretrained_path,sc=False).to(
        device)
    print("evaluating")
    confmat = evaluate(model, data_loader_test, device=device,
                       num_classes=num_classes,eval_steps=eval_steps,print_every=5)
    print(confmat)

def benchmark(config_filename):
    with open(config_filename) as file:
        config=yaml.full_load(file)
        print(config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    mixed_precision=config["mixed_precision"]
    warmup_iter=config["warmup_iter"]
    num_iter=config["num_iter"]
    crop_size=config["train_size"]
    batch_size=config["batch_size"]
    num_classes=config["num_classes"]
    model=get_model(config).to(device)
    data_loader, data_loader_test=get_dataset_loaders(config)
    dic=compute_time_full(model,data_loader,warmup_iter,num_iter,device,crop_size,batch_size,num_classes,mixed_precision)
    for k,v in dic.items():
        print(f"{k}: {v}")

if __name__=='__main__':
    benchmark("PyTorch_DeepLab/configs/voc_regnety40_30epochs_mixed_precision.yaml")
    #main2("PyTorch_DeepLab/configs/voc_regnety40_30epochs_mixed_precision.yaml")
    #check()
    #main()
    #check()
