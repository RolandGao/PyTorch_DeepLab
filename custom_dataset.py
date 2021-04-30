from PIL import Image
import torch.utils.data as data
import os
import json

class SegmentationDataset(data.Dataset):
    def __init__(self,root,image_set,transforms):
        # images, masks, json splits
        split_f=os.path.join(root,f"{image_set}.json")
        file_names=json.load(open(split_f, 'r'))
        root= os.path.expanduser(root)
        self.transforms=transforms
        image_dir = os.path.join(root, "Images")
        mask_dir=os.path.join(root, "Masks")
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

if __name__=='__main__':
    dataset=SegmentationDataset("yoho",image_set="train",transforms=None)
    print(dataset[1])
