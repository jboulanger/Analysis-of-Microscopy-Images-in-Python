import glob
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,datasets,io
import random
import matplotlib.pyplot as plt

class DnCNN(nn.Module):
    """DnCNN denoising network

    Zhang, Kai, et al. "Beyond a gaussian denoiser: Residual learning of deep
    cnn for image denoising." IEEE transactions on image processing 26.7
    (2017): 3142-3155.
    """
    def __init__(self, depth=17):
        super(DnCNN, self).__init__()
        layers = []
        nf = 64
        layers.append(nn.Conv2d(in_channels=1,out_channels=nf,kernel_size=3,padding=1,bias=False))
        layers.append(nn.GELU())#inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=nf,out_channels=nf,kernel_size=3,padding=1,bias=False))
            layers.append(nn.BatchNorm2d(num_features=nf))
            layers.append(nn.GELU())#inplace=True))
        layers.append(nn.Conv2d(in_channels=nf,out_channels=1,kernel_size=3,padding=1,bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self,x):
        return self.net(x)

class DnCNNDataset(Dataset):
    """DnCNN dataset loading images in a folder"""
    def __init__(self, path, transform=None):
        super().__init__()
        self.imagelist = [
            io.read_image(x, io.ImageReadMode.GRAY).squeeze()
            for x in glob.glob(path)
        ]
        self.transform = transform
    def __len__(self):
        return len(self.imagelist)
    def __getitem__(self, idx):
        image = self.imagelist[idx]
        return self.transform(image)

class DnCNNAugmenter(object):
    """Crop, flip, add noise and return a noisy and residual image"""
    def __init__(self, shape, noise_level):
        """Initialize the transform with shape and noise level
        shape : [H,W] dimension of the crop
        noise : noise level on which to train DnCNN
        """
        self.shape = shape
        self.noise_level = noise_level
    def __call__(self, sample):
        """Apply the transform to the input"""
        img = sample.clone().squeeze()
        # random crops
        c = random.randint(0, img.shape[1] - self.shape[1])
        r = random.randint(0, img.shape[0] - self.shape[0])
        img = img[r:r+self.shape[0],c:c+self.shape[1]]
        # random flips
        for k in [0,1]:
            if random.random() > 0.5:
                img = torch.flip(img, [k])
        # random gamma [0.5,1.5]
        img = 255.0 * torch.pow(img / 255.0, 0.5+random.random())
        # reshape to tensor
        img = img.reshape([1,*self.shape])
        # add noise
        residuals = self.noise_level * torch.randn(img.shape)
        noisy = img + residuals
        return noisy, residuals