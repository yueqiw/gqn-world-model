import collections, os, io, gzip 
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

Context = collections.namedtuple('Context', ['frames', 'cameras'])
Scene = collections.namedtuple('Scene', ['frames', 'cameras'])


def transform_viewpoint(v):
    """
    Transforms the viewpoint vector into a consistent
    representation
    """
    w, z = torch.split(v, 3, dim=-1)
    y, p = torch.split(z, 1, dim=-1)

    # position, [yaw, pitch]
    view_vector = [w, torch.cos(y), torch.sin(y), torch.cos(p), torch.sin(p)]
    v_hat = torch.cat(view_vector, dim=-1)

    return v_hat


class ShepardMetzler(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.filenames = []
        for folder in os.listdir(self.root_dir):
            files = [os.path.join(folder, x) for x in os.listdir(os.path.join(self.root_dir, folder)) if x.endswith(".pt.gz")]
            self.filenames.extend(files)
        #self.filenames = [x for x in os.listdir(self.root_dir) if x.endswith(".pt.gz")]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.filenames)
        # return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        # scene_path = os.path.join(self.root_dir, "{}.pt".format(idx))
        # data = torch.load(scene_path)
        #print(idx)

        gz_scene_path = os.path.join(self.root_dir, self.filenames[idx])
        with gzip.open(gz_scene_path, 'rb') as f:
            # Use an intermediate buffer
            x = io.BytesIO(f.read())
            data = torch.load(x)

        #byte_to_tensor = lambda x: ToTensor()(Image.open(io.BytesIO(x)))
        array_to_tensor = lambda x: ToTensor()(Image.fromarray(x))

        images = torch.stack([array_to_tensor(frame) for frame in data[0]])

        viewpoints = torch.from_numpy(data[1])
        viewpoints = viewpoints.view(-1, 5)

        if self.transform:
            images = self.transform(images)

        if self.target_transform:
            viewpoints = self.target_transform(viewpoints)

        return images, viewpoints