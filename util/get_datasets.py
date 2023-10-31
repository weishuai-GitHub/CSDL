import os,sys

if  os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import os
from config import *
from torchvision import datasets
from util.transform import get_transform
from torch.utils import data
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from scipy import io as mat_io
import numpy as np

import pandas as pd
from torchvision.datasets.utils import download_url

class CustomCub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = None

    def __init__(self, root, train=True, transform=None, target_transform=None, loader=default_loader, download=True):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.loader = loader
        self.train = train


        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        
        self.targets = self.data.iloc[range(len(self.data))].target-1
        self.targets = list(self.targets)
        self.uq_idxs = np.array(range(len(self)))

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = self.targets[idx]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class CarsDataset(Dataset):
    """
        Cars Dataset
    """
    def __init__(self, train=True, limit=0, data_dir=car_root, transform=None, metas=meta_default_path):

        data_dir = data_dir.format('train') if train else data_dir.format('test')
        metas = metas.format('train_annos') if train else metas.format('test_annos_withlabels')

        self.loader = default_loader
        self.data_dir = data_dir
        self.data = []
        self.targets = []
        self.train = train

        self.transform = transform

        if not isinstance(metas, str):
            raise Exception("Train metas must be string location !")
        labels_meta = mat_io.loadmat(metas)

        for idx, img_ in enumerate(labels_meta['annotations'][0]):
            if limit:
                if idx > limit:
                    break

            # self.data.append(img_resized)
            self.data.append(data_dir + img_[5][0])
            # if self.mode == 'train':
            self.targets.append(img_[4][0][0]-1)

        self.uq_idxs = np.array(range(len(self)))
        self.target_transform = None

    def __getitem__(self, idx):

        image = self.loader(self.data[idx])
        target = self.targets[idx]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        #idx = self.uq_idxs[idx]

        return image, target

    def __len__(self):
        return len(self.targets)

class loadData():
    def __init__(self,datasets,include_classes=range(5),prop_indices_to_subsample = 0.8):
        self.datasets = datasets
        cls_idxs = [x for x, t in enumerate(datasets.targets) if t in include_classes]
        indices_to_subsample = np.random.choice(cls_idxs, 
                                        int(prop_indices_to_subsample * len(cls_idxs)),
                                        replace=False)    
        self.mask = np.zeros((len(datasets.targets),),dtype=np.bool8)
        self.mask[indices_to_subsample] = True
    
    def __getitem__(self, index):
        img,target =  self.datasets.__getitem__(index)
        mask = self.mask[index]
        return img,target,mask
    
    def __len__(self):
        return len(self.datasets.targets)

def get_dataset(data_name='cifar10',is_train=True,transform=None):
    if(data_name=='cifar10'):
        dataset = datasets.CIFAR10(root=cifar_10_root,transform=transform,train=is_train)
    elif(data_name=='cifar100'):
        dataset = datasets.CIFAR100(root=cifar_10_root,transform=transform,train=is_train,download=False)
    elif(data_name=='imagenet100'):
        if is_train:
            root = os.path.join(imagenet_root,'train')
        else:
             root = os.path.join(imagenet_root,'val')
        dataset = datasets.ImageFolder(root,transform=transform)
    elif(data_name=='imagenet10'):
        if is_train:
            root = os.path.join(imagenet10_root,'train')
        else:
             root = os.path.join(imagenet10_root,'test')
        dataset = datasets.ImageFolder(root,transform=transform)
    elif(data_name=='scars'):
        dataset = CarsDataset(data_dir=car_root, transform=transform, metas=meta_default_path, train=is_train)
    elif(data_name=='cub'):
        dataset =  CustomCub2011(root=cub_root, transform=transform, train=is_train,download=False)
    elif(data_name=='cub10'):
        dataset =  datasets.ImageFolder(root=cub10_root,transform=transform)
    elif(data_name=='herbarium'):
        if is_train:
            root = os.path.join(herbarium_dataroot,'small-train')
        else:
            root = os.path.join(herbarium_dataroot,'small-validation')
        dataset = datasets.ImageFolder(root,transform=transform)
    else:
        raise NotImplementedError("don't have dataset:{}".format(data_name))
    return dataset

if __name__=="__main__":
    transform = get_transform(image_size=224)
    datasets = get_dataset(data_name='scars',transform=transform)
    datasets = loadData(datasets,include_classes=range(98))
    loader = data.DataLoader(datasets,128,shuffle=True)
    for i ,(img,labels,mask) in enumerate(loader):
        print(type(labels),type(mask))