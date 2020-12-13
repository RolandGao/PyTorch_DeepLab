import torch
import torch.utils.data
import numpy as np

def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

def collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets

def get_sampler(dataset,dataset_test,distributed=False):
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        #train_sampler = torch.utils.data.SequentialSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    return train_sampler,test_sampler

def worker_init_fn(worker_id):
    from datetime import datetime
    np.random.seed(datetime.now().microsecond)
def get_dataloader_train(dataset,batch_size):
    num_workers = 4
    #dataset = get_coco(image_folder, ann_file, "train",get_temp_transform())
    train_sampler = torch.utils.data.RandomSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers,
        collate_fn=collate_fn, drop_last=True)
    return data_loader

def get_dataloader_val(dataset_test):
    num_workers = 4
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=num_workers,
        collate_fn=collate_fn)
    return data_loader_test

def find_mean_and_std(data_loader):
    K = 0.8
    n = 0.0
    Ex=torch.zeros(3).float()
    Ex2 = torch.zeros(3).float()
    count=0
    for image,_ in data_loader:
        count+=1
        assert len(image.size())==4
        image = image.transpose(0, 1).flatten(1)
        Ex += (image-K).sum(dim=1)
        Ex2 += ((image - K)**2).sum(dim=1)
        n +=image.size()[1]
        if count==1000:
            break
    mean=Ex/n+K
    variance=(Ex2 - (Ex * Ex)/n)/(n-1)
    std=variance.sqrt()
    return mean, std

def find_class_weights(dataloader,num_classes):
    print_every=4
    class_weights=torch.zeros(num_classes)
    for count,(image,target) in enumerate(dataloader):
        class_weights+=torch.bincount(target[target<num_classes],minlength=num_classes)
        if (count+1) % print_every==0:
            print(f"{count+1} done")
        if count==300:
            break
    n = class_weights.sum().item()
    class_weights = [n / (num_classes * w.item()) if w != 0 else 0 for w in class_weights]
    return class_weights
