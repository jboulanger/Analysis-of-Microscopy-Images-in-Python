import math
import numpy as np
import torch

def generate_otf3d_grid(shape, pixel_size):
    """
    Generate a grid on which to evaluate the optical transfer function
    Parameters
    ----------
    shape : list [nx,ny,nz] giving the shape of the final array
    pixel_size : sampling in [x,y,z]    
    Returns
    -------
    kx,ky,kz,z: frequency grid in x and y and spatial space in z
    """
    f = [np.fft.fftfreq(n, pixel_size[k]) for k, n in enumerate(shape)]
    ez = np.concatenate((np.arange(0,shape[2]//2), np.arange((-shape[2])//2,0))) * pixel_size[2]
    [kx,ky,kz] = np.meshgrid(*f)
    z = np.meshgrid(f[0],f[1],ez)[2]
    return kx,ky,kz,z

def generate_otf3d_on_grid(grid,numerical_aperture,wavelength,medium_refractive_index):
    """
    Generate a diffraction limited wide field optical transfer function and point spread function
    Parameters
    ----------
    grid : tuple [kx,ky,z] with grid on which to evaluate the otf
    pixel_size : sampling in [x,y,z]
    numerical aperture : numerical aperture
    wavelength : wavelength of the emitted light
    medium_refractive_index : refractive index of the immersion medium
    Returns
    --------
    otf : the optical transfer function as an array of shape 'shape'
    psf : the point spread function as an array of shape 'shape' centerd in 0,0,0
    """
    d2 = np.square(grid[0]) + np.square(grid[1])
    rho = np.sqrt(d2) * wavelength / numerical_aperture
    P = np.where(rho <= 1.0, 1.0, 0.0)
    defocus = grid[3] * np.sqrt(np.clip((medium_refractive_index / wavelength)**2 - d2,0,None))
    psf = np.square(np.abs(np.fft.fft2(P * np.exp(2j * math.pi * defocus),axes=[0,1])))
    psf = psf / psf.sum()
    otf = np.fft.fftn(psf)
    return otf, psf

def generate_otf3d(shape,pixel_size,wavelength,NA,medium_refractive_index):
    """
    Generate a diffraction limited wide field optical transfer function and point spread function
    Parameters
    ----------
    shape : list [nx,ny,nz] giving the shape of the final array
    pixel_size : sampling in [x,y,z]
    NA : numerical aperture
    wavelength : wavelength of the emitted light
    medium_refractive_index : refractive index of the immersion medium
    Returns
    --------
    otf : the optical transfer function as an array of shape 'shape'
    psf : the point spread function as an array of shape 'shape' centerd in 0,0,0
    """
    grid = generate_otf3d_grid(shape, pixel_size)
    return generate_otf3d_on_grid(grid,NA,wavelength,medium_refractive_index)

def deconvolve_richardson_lucy(data, otf, background=0, iterations=100):
    """ 
    Deconvolve data according to the given otf using a Richardson-Lucy algorithm
    Parameters
    ----------
    data       : numpy array
    otf        : numpy array of the same size than data
    background : background level
    iterations : number of iterations
    Result
    ------
    estimate   : estimated image
    dkl        : Kullback Leibler divergence
    """
    estimate = np.clip(np.real(np.fft.ifftn(otf * np.fft.fftn(data-background))), a_min=1e-6, a_max=None)
    dkl = np.zeros(iterations)
    for k in range(iterations):
        blurred = np.clip(np.real(np.fft.ifftn(otf * np.fft.fftn(estimate+background))), a_min=1e-6, a_max=None)
        ratio = data / blurred
        estimate = estimate * np.real(np.fft.ifftn(otf * np.fft.fftn(ratio)))
        dkl[k] = np.mean(blurred - data + data * np.log(np.clip(ratio,a_min=1e-6,a_max=None)))
    return estimate, dkl


def deconvolve_richardson_lucy_heavy_ball(data, otf, background, iterations):
    """ 
    Deconvolve data according to the given otf using a scaled heavy ball Richardson-Lucy algorithm
    Parameters
    ----------
    data       : numpy array
    otf        : numpy array of the same size than data
    iterations : number of iterations
    Result
    ------
    estimate   : estimated image
    dkl        : the kullback leibler divergence (should tend to 1/2)
    Note
    ----
    https://doi.org/10.1109/tip.2013.2291324
    """
    old_estimate = np.clip(np.real(np.fft.ifftn(otf * np.fft.fftn(data - background))), a_min=0, a_max=None)
    estimate = data
    dkl = np.zeros(iterations)
    for k in range(iterations):
        beta = (k-1.0) / (k+2.0)
        prediction = estimate + beta * (estimate -  old_estimate)
        blurred = np.clip(np.real(np.fft.ifftn(otf * np.fft.fftn(prediction + background))), a_min=1e-6, a_max=None)
        ratio = data / blurred
        gradient = 1.0 - np.real(np.fft.ifftn(otf * np.fft.fftn(ratio)))
        old_estimate = estimate
        estimate = np.clip(prediction - estimate * gradient, a_min=0.1, a_max=None)
        dkl[k] = np.mean(blurred - data + data * np.log(np.clip(ratio,a_min=1e-6, a_max=None)))
    return estimate, dkl

def deconvolve_richardson_lucy_heavy_ball_torch(data, otf, background, iterations, device):
    """ 
    Deconvolve data according to the given otf using a scaled heavy ball Richardson-Lucy algorithm 
    Parameters
    ----------
    data       : torch tensor
    otf        : optical transfer function
    background : bakground level
    iterations : number of iterations
    Result
    ------
    estimate   : estimated image
    dkl        : the kullback leibler divergence (should tend to 1/2)
    Note
    ----
    https://doi.org/10.1109/tip.2013.2291324
    Needs pytorch (>1.9))
    """
    data = torch.from_numpy(data.astype(float),device=device)
    otf = torch.from_numpy(otf.astype(complex),device=device)
    old_estimate = torch.clamp(torch.real(torch.fft.ifftn(otf * torch.fft.fftn(data))), min=1e-6)
    estimate = data
    dkl = torch.zeros(iterations, dtype=float)
    for k in range(iterations):
        beta = (k-1.0) / (k+2.0)
        prediction = estimate + beta * (estimate -  old_estimate)
        blurred = torch.clamp(torch.real(torch.fft.ifftn(otf * torch.fft.fftn(prediction + background))), min=1e-6)
        ratio = data / blurred
        gradient = 1.0 - torch.real(torch.fft.ifftn(otf * torch.fft.fftn(ratio)))
        old_estimate = estimate
        estimate = torch.clamp(prediction - estimate * gradient, min=1e-6)
        dkl[k] = torch.mean(blurred - data + data * torch.log(torch.clamp(ratio,min=1e-6)))
    return estimate.numpy(), dkl.numpy()



