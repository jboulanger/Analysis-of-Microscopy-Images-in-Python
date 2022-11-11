import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import io
import random

from . import utils

def benchmark(method, imgpath, blind=False, noise_levels=range(5,30,5)):
    """Denoising benchmark
    Parameters
    ----------
    method  : denoising methof lambda x,sigma: return denoised image
    imgpath : path to the images to denoise
    Returns
    -------
    a dataframe with the metrics
    """
    import pathlib
    import pandas as pd
    from skimage import io as skio
    from skimage import metrics
    import time
    results = []
    for fname in sorted(glob.glob(imgpath)):
        img = skio.imread(fname).astype(float)
        for noise_std in noise_levels:
            noisy = img + np.random.normal(0, noise_std, img.shape)
            t0 = time.perf_counter()
            if blind:
                denoised = method(noisy)
            else:
                denoised = method(noisy, noise_std)
            t1 = time.perf_counter()
            results.append({
                'noise level': noise_std,
                'image name': pathlib.Path(fname).stem,
                'psnr': metrics.peak_signal_noise_ratio(img, denoised, data_range=255),
                'ssim': metrics.structural_similarity(img, denoised),
                'time': t1 - t0
            })
    return pd.DataFrame.from_records(results)


class generalized_anscombe_transform():
    def __init__(self, gain, offset, noise_std):
        self.g0 = gain
        self.edc = sigma**2 - gain * offset

    def __call__(self, x):
        b = 3. / 8.* self.g0 ** 2 + self.edc
        return 2. / self.g0 * np.sqrt(self.gain * x + b)

    def inverse(self, x):
        b = 3. / 8.* self.g0 ** 2 + self.edc
        return 0.25 * self.g0 * x**2  - b / self.g0

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
        noise : noise level on which to train DnCNN, if noise is an interval
                [a,b] then the network will be able to denoise various noise
                levels.
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
        if isinstance(self.noise_level,tuple) or isinstance(self.noise_level,list):
            noise_std = random.uniform(*self.noise_level)
            residuals = noise_std * torch.randn(img.shape)
        else:
            residuals = self.noise_level * torch.randn(img.shape)
        noisy = img + residuals
        return noisy, residuals


class DnCNNDenoiser():
    """DnCNN denoiser
    Load a trained model and apply it to images
    """
    def __init__(self, model, device='cpu'):
        """Initialize the denoiser with a model
        Parameters
        ----------
        model : the pretrained model
        """
        self.model = model
        self.device = device

    def __call__(self, x):
        """Apply the denoiser to a 2D numpy array
        Parameters
        ----------
        x : a noisy image as a numpy array
        Returns
        -------
        the denoised input
        """
        with torch.no_grad():
            input = torch.from_numpy(x).float().reshape([1,1,*x.shape]).to(self.device)
            residuals = self.model(input).cpu().numpy().reshape(x.shape)
            return x - residuals


