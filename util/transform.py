import torch
from torchvision import transforms

def get_transform(image_size=224):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return transform
def get_diff_transform(x):
   ranHorFlip = transforms.RandomHorizontalFlip()
   ranVerFilp = transforms.RandomVerticalFlip()
   return ranHorFlip(x),ranVerFilp(x)