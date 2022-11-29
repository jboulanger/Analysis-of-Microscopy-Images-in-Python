from collections import OrderedDict
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
                denoised = method(noisy).astype(float)
            else:
                denoised = method(noisy, noise_std).astype(float)
            t1 = time.perf_counter()
            results.append({
                'noise level': noise_std,
                'image name': pathlib.Path(fname).stem,
                'psnr': metrics.peak_signal_noise_ratio(img, denoised, data_range=255),
                'ssim': metrics.structural_similarity(img, denoised),
                'time': t1 - t0
            })
    return pd.DataFrame.from_records(results)

def pseudoresiduals(data:np.ndarray):
    """Scaled pseudo residuals
    Parameter
    ---------
    data : numpy ndarray
    Returns
    -------
    Returns the pseudo-residuals image as a numpy.ndarray
    """
    from scipy.ndimage import laplace
    import math
    D = sum([1 for s in data.shape if s>1])
    residuals = laplace(data.astype(float))  / math.sqrt(4 * D * D + 2 * D)
    return residuals

def estimate_noise_scale(data:np.ndarray):
    """Estimate the scale of an additive Gaussian white noise
    Parameter
    ---------
    data : numpy ndarray
    Returns
    -------
    Estimate of the additive Gaussian noise standard deviation.
    """
    residuals = pseudoresiduals(data)
    return 1.4826 * np.median(np.abs(residuals - np.median(residuals)))

def compute_local_statistics(data, scale=5):
    """Compute local mean and variance using robust estimators"""
    from scipy.ndimage import median_filter
    residuals = pseudoresiduals(data)
    mean = median_filter(data, scale)
    variance = median_filter(residuals, scale)
    variance = np.square(1.4826 * median_filter(np.abs(residuals - variance), scale))
    return mean, variance

class generalized_anscombe_transform():
    """Generalized Anscombe Transform
    Use the following model
    data = g0 N + offset + W
    where N is Poisson distributed and W normally distributed with scale 'readout'
    Attributes
    ----------
    g0  : gain
    edc : readout^2 - gain * offset
    """
    def __init__(self, gain=1, offset=0, readout=0):
        self.g0 = gain
        self.edc = readout**2 - gain * offset

    def __call__(self, x):
        """Apply the stabilization transform"""
        b = 3. / 8.* self.g0 ** 2 + self.edc
        return 2. / self.g0 * np.sqrt(np.maximum(0, self.g0 * x + b))

    def invert(self, x):
        """Inverse the transform"""
        b = 3. / 8.* self.g0 ** 2 + self.edc
        return 0.25 * self.g0 * x**2  - b / self.g0

    def calibrate(self, data, scale = 10, mode = 'huber'):
        """Single image calibration of the parameters"""
        x, y = compute_local_statistics(data, scale)
        D = sum([1 for s in data.shape if s>1])
        idx = [slice(0,data.shape[k], scale) for k in range(D)]
        x = x[tuple(idx)]
        y = y[tuple(idx)]
        n = np.prod(x.shape)
        x = x.reshape([n,1])
        y = y.reshape([n,1])
        X = np.concatenate([np.ones([n,1]), x], axis=1)
        if mode == 'huber':
            from sklearn.linear_model import HuberRegressor
            reg  = HuberRegressor().fit(X, y.ravel())
            self.g0 = reg.coef_[1]
            self.edc = reg.coef_[0]
        else:
            from sklearn.linear_model import RANSACRegressor
            reg = RANSACRegressor().fit(X, y)
            self.g0 = reg.estimator_.coef_[0,1].item()
            self.edc = reg.estimator_.intercept_.item()

        #import matplotlib.pyplot as plt
        #plt.scatter(x.flatten(), y.flatten())
        #t = np.arange(x.min(), x.max())
        #plt.plot(t, self.g0 * t + self.edc, 'r')

    def __str__(self):
        return f"GAT parameters - gain:{self.g0} edc {self.edc}"


class ImageFolderDataset(Dataset):
    """Pytorch dataset loading images in a folder"""
    def __init__(self, path, transform=None, length=None):
        super().__init__()
        self.imagelist = [
            io.read_image(x, io.ImageReadMode.GRAY).squeeze()
            for x in glob.glob(path)
        ]
        self.transform = transform
        if length is None:
            self.number_of_images = len(self.imagelist)
        else:
            self.number_of_images = length
    def __len__(self):
        return self.number_of_images
    def __getitem__(self, idx):
        image = self.imagelist[idx % len(self.imagelist)]
        return self.transform(image)

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

    def save(self, path):
        torch.save(self.net.state_dict(), path)


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
        if isinstance(self.noise_level,tuple) or isinstance(self.noise_level,list):
            self.blind = True
        else:
            self.blind = False

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
        if self.blind:
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
        self.model.to(device)
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
            z = torch.from_numpy(x).float().reshape([1,1,*x.shape]).to(self.device)
            residuals = self.model(z).cpu().numpy().reshape(x.shape)
            return x - residuals


class DRUNETAugmenter(object):
    """Crop, flip, add noise and return a noisy and noise free image"""
    def __init__(self, shape, noise_level):
        """Initialize the transform with shape and noise level
        shape : [H,W] dimension of the crop
        noise : noise level on which to train DnCNN, if noise is an interval
                [a,b] then the network will be able to denoise various noise
                levels.
        """
        self.shape = shape
        self.noise_level = noise_level
        if isinstance(self.noise_level,tuple) or isinstance(self.noise_level,list):
            self.blind = True
        else:
            self.blind = False

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
        #img = torch.nn.functional.interpolate(img, scale_factor=(0.5,0.5))
        # add noise
        if self.blind:
            noise_std = random.uniform(*self.noise_level)
        else:
            noise_std = self.noise_level
        residuals = noise_std * torch.randn(img.shape)
        noisy = img + noise_std * torch.randn(img.shape)
        noise_map = noise_std * torch.ones(noisy.shape)
        noisy = torch.cat((noisy , noise_map), 0)
        return noisy, img


class Block2d(nn.Module):
    """Basic building block with conv, batchnorm and relu"""
    def __init__(self, in_channels, out_channels, kernel_size=3,padding=1,bias=False):
        super(Block2d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding,bias=bias),
            nn.BatchNorm2d(out_channels,momentum=0.9,eps=1e-4,affine=True),
            nn.LeakyReLU(negative_slope=0.2)
        )
    def forward(self,x):
        return self.net(x)


class ResBlock2d(nn.Module):
    """2D Residual block of resnet with 1x1 conv
    https://d2l.ai/chapter_convolutional-modern/resnet.html
    """
    def __init__(self,in_channels,out_channels,kernel_size=3,padding=1,bias=False):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding,bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels,momentum=0.9,eps=1e-4,affine=True)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding,bias=bias)
        self.bn2  = nn.BatchNorm2d(out_channels,momentum=0.9,eps=1e-4,affine=True)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,bias=bias)

    def forward(self,x):
        return self.relu2(self.conv3(x) + self.bn2(self.conv2(self.relu1(self.bn1(self.conv1(x))))))


class UNet2D(nn.Module):
    """2D U-Net """
    def __init__(self, block, in_channels=1, out_channels=1, nfeatures=[4,8], nblocks=2):
        super(UNet2D, self).__init__()
        self.head = nn.Conv2d(in_channels, nfeatures[0], 3, 1, 1, bias=False)
        self.down = nn.ModuleList([
            nn.Sequential(OrderedDict([
                *[
                    (
                        f'block{k}{l}',
                        block(nfeatures[k if l==0 else k+1], nfeatures[k+1])
                    )
                    for l in range(nblocks)
                ],
                (f'down{k}',nn.Conv2d(nfeatures[k+1], nfeatures[k+1], 3, 1, 1, bias=False))
            ]))
            for k in range(len(nfeatures)-1)
        ])

        self.body =  nn.Sequential(OrderedDict([
            ('block', block(nfeatures[-1], nfeatures[-1]))
            for _ in range(nblocks)
        ]))

        self.up = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('up', nn.ConvTranspose2d(nfeatures[-k-1], nfeatures[-k-1], 3, 1, 1, bias=False)),
                *[
                    (
                        f'block{k}{l}',
                        block(nfeatures[-k-1 if l==0 else -k-2], nfeatures[-k-2])
                    )
                    for l in range(nblocks)
                ]
            ]))
            for k in range(len(nfeatures)-1)
        ])
        self.tail = nn.Conv2d(nfeatures[0], out_channels, 3, 1, 1, bias=False)

    def forward(self,x):
        z = [self.head(x)]
        for k, layer in enumerate(self.down):
            z.append(layer(z[k]))
        y = self.body(z[-1])
        for k, layer in enumerate(self.up):
            y = layer(y+z[-k-1])
        return self.tail(y)


class DRUNET(UNet2D):
    """DRUNET model"""
    def __init__(self):
        super(DRUNET, self).__init__(ResBlock2d, in_channels=2, nfeatures=[16,32,64,128])


class DRUNETDenoiser():
    """DRUNET denoiser
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

    def __call__(self, x, noise_std):
        """Apply the denoiser to a 2D numpy array
        Parameters
        ----------
        x         : a noisy image as a numpy array
        noise_std : standard deviation of the Gaussian noise
        Returns
        -------
        the denoised input
        """
        with torch.no_grad():
            z = torch.from_numpy(x).float().reshape([1,1,*x.shape]).to(self.device)
            noise_map = noise_std * torch.ones(z.shape)
            z = torch.cat((z , noise_map), 1)
            return self.model(z).cpu().numpy().reshape(x.shape)