from torch import nn
import torch
import time
import torch.cuda.amp as amp
import torch.nn.functional

@torch.no_grad()
def compute_eval_time(model,device,warmup_iter,num_iter,crop_size,batch_size,mixed_precision):
    model.eval()
    x=torch.randn(batch_size,3,crop_size,crop_size).to(device)
    times=[]
    for cur_iter in range(warmup_iter+num_iter):
        if cur_iter == warmup_iter:
            times.clear()
        t1=time.time()
        with amp.autocast(enabled=mixed_precision):
            output = model(x)
        torch.cuda.synchronize()
        t2=time.time()
        times.append(t2-t1)
    return average(times)

def average(v):
    return sum(v)/len(v)
def compute_train_time(model,warmup_iter,num_iter,crop_size,batch_size,num_classes,mixed_precision):
    model.train()
    x=torch.randn(batch_size, 3, crop_size, crop_size).cuda(non_blocking=False)
    target=torch.randint(0,num_classes,(batch_size, crop_size, crop_size)).cuda(non_blocking=False)
    fw_times=[]
    bw_times=[]
    scaler = amp.GradScaler(enabled=mixed_precision)
    for cur_iter in range(warmup_iter+num_iter):
        if cur_iter == warmup_iter:
            fw_times.clear()
            bw_times.clear()
        t1=time.time()
        with amp.autocast(enabled=mixed_precision):
            output = model(x)
            loss = nn.functional.cross_entropy(output,target,ignore_index=255)
        torch.cuda.synchronize()
        t2=time.time()
        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        t3=time.time()
        fw_times.append(t2-t1)
        bw_times.append(t3-t2)
    return average(fw_times),average(bw_times)

def compute_loader_time(data_loader,warmup_iter,num_iter):
    times=[]
    data_loader_iter=iter(data_loader)
    for cur_iter in range(warmup_iter+num_iter):
        if cur_iter == warmup_iter:
            times.clear()
        t1=time.time()
        next(data_loader_iter)
        t2=time.time()
        times.append(t2-t1)
    return average(times)


def memory_used(device):
    x=torch.cuda.memory_allocated(device)
    return round(x/1024/1024)
def max_memory_used(device):
    x=torch.cuda.max_memory_allocated(device)
    return round(x/1024/1024)
def memory_test_helper(model,device,crop_size,batch_size,num_classes,mixed_precision):
    model.train()
    scaler = amp.GradScaler(enabled=mixed_precision)
    x=torch.randn(batch_size, 3, crop_size, crop_size).to(device)
    target=torch.randint(0,num_classes,(batch_size, crop_size, crop_size)).to(device)
    t1=memory_used(device)
    with amp.autocast(enabled=mixed_precision):
        output = model(x)
        loss = nn.functional.cross_entropy(output,target,ignore_index=255)
    scaler.scale(loss).backward()
    torch.cuda.synchronize()
    t2=max_memory_used(device)
    torch.cuda.reset_peak_memory_stats(device)
    return t2-t1

def compute_memory_usage(model,device,crop_size,batch_size,num_classes,mixed_precision):
    for p in model.parameters():
        p.grad=None
    try:
        t=memory_test_helper(model,device,crop_size,batch_size,num_classes,mixed_precision)
        print()
    except:
        t=-1
        print("out of memory")
    for p in model.parameters():
        p.grad=None
    return t

def compute_time_no_loader(model,warmup_iter,num_iter,device,crop_size,batch_size,num_classes,mixed_precision):
    model=model.to(device)
    print("benchmarking eval time")
    eval_time=compute_eval_time(model,device,warmup_iter,num_iter,crop_size,batch_size,mixed_precision)
    print("benchmarking train time")
    train_fw_time,train_bw_time=compute_train_time(model,warmup_iter,num_iter,crop_size,batch_size,num_classes,mixed_precision)
    train_time=train_fw_time+train_bw_time
    print("benchmarking memory usage")
    memory_usage=compute_memory_usage(model,device,crop_size,batch_size,num_classes,mixed_precision)
    print("benchmarking loader time")
    dic1={
        "eval_time":eval_time,
        "train_time":train_time,
        "memory_usage":memory_usage
    }
    return dic1

def compute_time_full(model,data_loader,warmup_iter,num_iter,device,crop_size,batch_size,num_classes,mixed_precision):
    model=model.to(device)
    print("benchmarking eval time")
    eval_time=compute_eval_time(model,device,warmup_iter,num_iter,crop_size,batch_size,mixed_precision)
    print("benchmarking train time")
    train_fw_time,train_bw_time=compute_train_time(model,warmup_iter,num_iter,crop_size,batch_size,num_classes,mixed_precision)
    train_time=train_fw_time+train_bw_time
    print("benchmarking memory usage")
    memory_usage=compute_memory_usage(model,device,crop_size,batch_size,num_classes,mixed_precision)
    print("benchmarking loader time")
    loader_time=compute_loader_time(data_loader,warmup_iter,num_iter)
    loader_overhead=max(0,loader_time-train_time)/train_time
    dic1={
        "eval_time":eval_time,
        "train_time":train_time,
        "memory_usage":memory_usage,
        "loader_time":loader_time,
        "loader_overhead":loader_overhead
    }
    dic2={
        "eval_time":eval_time*len(data_loader),
        "train_time":train_time*len(data_loader),
        "memory_usage":memory_usage,
        "loader_time":loader_time,
        "loader_overhead":loader_overhead
    }
    return dic1
