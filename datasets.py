import collections, os, io
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

class FaceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir 
    
    def __getitem__(self, idx):
        pass 