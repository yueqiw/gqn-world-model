import collections, os, io
from PIL import Image
import torch
from torchvision.transforms import Resize, Compose, ToTensor
from torch.utils.data import Dataset
from skimage import io as skio
import numpy as np


def transform_viewpoint(v):
    return v / 60.0

def resize_img(img, s=64):
    transform = Compose([Resize(size=(s,s)), ToTensor()])
    return transform(img)

class Face3D(Dataset):
    def __init__(self, root_dir, n_imgs=15, resize=64, 
                transform=None, target_transform=None, sample_type='angle_random'):
        self.root_dir = root_dir 
        self.folder_names = [x for x in os.listdir(os.path.join(self.root_dir)) if x.startswith("face")]
        self.n_imgs = n_imgs 
        self.transform = transform
        self.target_transform = target_transform
        self.resize = 64
        self.sample_type = sample_type

    def __len__(self):
        return len(self.folder_names)

    def __getitem__(self, idx):
        face_path = os.path.join(self.root_dir, self.folder_names[idx], self.sample_type)
        files = [x for x in os.listdir(face_path) if x.endswith(".jpg")]
        if self.n_imgs == 'all':
            use = np.arange(len(files))
        else:
            use = np.random.choice(len(files), self.n_imgs, replace=False) 
        images = []
        viewpoints = []
        for i in use:
            f = files[i]
            vp = np.array([float(x) for x in f.strip('.jpg').split("_")[-3:]])
            img = Image.open(os.path.join(face_path, f)) 
            img = resize_img(img, self.resize)
            #img = ToTensor()(img)
            images.append(img) 
            viewpoints.append(vp)
        images = torch.stack(images)
        viewpoints = np.stack(viewpoints)
        viewpoints = torch.from_numpy(viewpoints).type('torch.FloatTensor')
        # print(images.shape)
        # print(viewpoints.shape)
        # print(images.dtype)
        # print(viewpoints.dtype)

        if self.transform:
            images = self.transform(images)

        if self.target_transform:
            viewpoints = self.target_transform(viewpoints)
        
        return images, viewpoints


        