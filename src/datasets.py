import collections, os, io
from PIL import Image
import torch
from torchvision.transforms import Resize, Compose, ToTensor
from torch.utils.data import Dataset
from skimage import io as skio
import numpy as np
import pickle, gzip 


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


class AgentScenesUnity(Dataset):
    def __init__(self, root_dir, n_actions, n_timesteps=10, resize=None, transform=None, 
                query_transform=None, sample_type='continuous', compress=True):
        self.root_dir = root_dir
        if compress:
            self.filenames = [x for x in os.listdir(os.path.join(self.root_dir)) if x.endswith(".p.gz")]
            self.filenames = sorted(self.filenames, key=lambda x: int(x.strip('.p.gz')))
        else:
            self.filenames = [x for x in os.listdir(os.path.join(self.root_dir)) if x.endswith(".p")]
            self.filenames = sorted(self.filenames, key=lambda x: int(x.strip('.p')))
        self.n_timesteps = n_timesteps 
        self.transform = transform
        self.query_transform = query_transform
        self.resize = 64
        self.sample_type = sample_type
        self.n_actions = n_actions + 1 # plus no action
        self.compress = compress

    def __len__(self):
        return len(self.filenames)
    
    def actions_to_onehot(self, actions, n):
        onehot = torch.FloatTensor(len(actions), n).zero_()
        onehot.scatter_(1, torch.unsqueeze(actions, 1), 1)
        return onehot
    
    def random_timesteps(self, n_total, n_select, method='continuous'):
        if method == 'all':
            return torch.arange(n_total)
        if n_select > n_total: 
            n_select = n_total 
        n = n_total - n_select + 1
        start = np.random.randint(n) 
        timesteps = np.arange(start, start + n_select)
        return timesteps
    
    def transform_time(self, timesteps):
        timesteps = torch.FloatTensor(timesteps)
        normalized = timesteps - timesteps[-2]
        normalized = normalized / 10
        return normalized
    
    def __getitem__(self, idx):
        filepath = os.path.join(self.root_dir, self.filenames[idx])
        
        if self.compress:
            with gzip.open(filepath, "rb") as f:
                data = pickle.load(f)
        else:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
        
        total_timesteps = len(data['previous_action'])

        if total_timesteps < self.n_timesteps:
            return self.__getitem__(np.random.randint(len(self.filenames)))

        timesteps_use = self.random_timesteps(total_timesteps, self.n_timesteps, method='continuous')

        subset_action = [data['previous_action'][i] for i in timesteps_use]
        subset_vector_obs = [data['vector_observation'][i] for i in timesteps_use]
        subset_reward = [data['reward'][i] for i in timesteps_use]
        subset_visual_obs = [data['visual_observation'][i] for i in timesteps_use]


        time_transform = self.transform_time(timesteps_use)
        
        # align previous action and reward with current observations. 
        actions = torch.LongTensor([x[0] for x in subset_action])
        actions_oh = self.actions_to_onehot(actions, self.n_actions)
        
        rewards = torch.FloatTensor(subset_reward)
        vector_obs = torch.from_numpy(np.stack(subset_vector_obs))
        images = np.stack(subset_visual_obs)
        images = torch.FloatTensor(images.transpose((0,3,1,2)))

        if self.transform:
           images = self.transform(images)
        
        queries = torch.cat([time_transform.unsqueeze(1), actions_oh], 1)
            
        return images, queries