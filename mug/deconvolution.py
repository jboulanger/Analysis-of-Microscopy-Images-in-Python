import math
import numpy as np


class otf_generator():
    """Scalar optical transfer function generator for wide-field microscopes

    This class will cache variable so that calling the generator with different
    aberrations parameters will not require the computation of these quantities.

    To generate a single OTF use the following example code:
    ```
    otf = deconvolution.otf_generator([32,256,256],[300,100,100],500,1,1.3)()[0]
    ```
    """
    def __init__(
        self,
        shape,
        pixel_size,
        wavelength,
        numerical_aperture,
        medium_refractive_index):

        self.shape = shape

        kx = np.fft.fftfreq(shape[2], pixel_size[2]).reshape([1, 1, shape[2]])
        ky = np.fft.fftfreq(shape[1], pixel_size[1]).reshape([1, shape[1], 1])
        z = np.reshape(np.stack(
            (
                np.arange(0, shape[0] // 2),
                np.arange((-shape[0]) // 2, 0)
            )
        ) * pixel_size[0], [shape[0],1,1])

        d2 = np.square(kx) + np.square(ky)

        self.rho = np.sqrt(d2) * (wavelength / numerical_aperture)
        self.phi = np.arctan2(ky,kx)
        corr =  np.power(np.maximum( 1 - d2 / (medium_refractive_index / wavelength)**2, 1e-3), -0.25)
        self.P = np.where(self.rho <= 1.0, 1.0, 0.0) * corr
        self.defocus = z * np.sqrt(
            np.clip((medium_refractive_index / wavelength)**2 - d2, 0, None)
        )

        self.Z = [
            self.rho * np.cos(self.phi),
            self.rho * np.sin(self.phi),
            2.0 * np.power(self.rho, 2.) - 1.,
            np.power(self.rho,2) * np.cos(2.*self.phi),
            np.power(self.rho,2) * np.sin(2.*self.phi),
            (3. * np.power(self.rho, 2.) - 2.) * self.rho * np.cos(self.phi),
            (3. * np.power(self.rho, 2.) - 2.) * self.rho * np.sin(self.phi),
            (6. * np.power(self.rho, 4.) - 6. * np.power(self.rho, 2.) + 1.),
            np.power(self.rho, 3.) * np.cos(3.*self.phi),
            np.power(self.rho, 3.) * np.sin(3.*self.phi),
            (4. * np.power(self.rho, 3.) - 3.) * np.power(self.rho, 2.) * np.cos(2.*self.phi),
            (4. * np.power(self.rho, 3.) - 3.) * np.power(self.rho, 2.) * np.sin(2.*self.phi),
            (10. * np.power(self.rho, 4.) - 12. * np.power(self.rho, 2.) + 3.) * self.rho * np.cos(self.phi),
            (10. * np.power(self.rho, 4.) - 12. * np.power(self.rho, 2.) + 3.) * self.rho * np.sin(self.phi),
            (20. * np.power(self.rho, 4.) - 30. * np.power(self.rho, 2.) + 12.)
        ]

    def __call__(self, aberrations=None):
        """
        Parameters
        ----------
        aberrations : the sequence of aberrations
        Returns
        -------
        The otf and the psf
        """
        W = self.defocus
        if aberrations is not None:
            for k, a in enumerate(aberrations):
                W += a * self.Z[k]

        psf = np.square(
            np.abs(
                np.fft.fft2(self.P * np.exp(2j * math.pi * W))
            )
        )
        psf = psf / psf.sum()
        otf = np.fft.fftn(psf)
        otf[np.abs(otf) < 1e-6] = 0
        return otf, psf


def blur(img: np.ndarray, otf: np.ndarray) -> np.ndarray:
    """Blur the image with the optical transfer function
    Parameters
    ----------
    img : image
    otf : optical transfer function (same size as the image)
    Returns
    -------
    the blurred image
    """
    return np.real(np.fft.ifftn(otf * np.fft.fftn(img)))

def deconvolve_gold_meinel(data, otf, background=0, iterations=100, acceleration=1.3, smooth=0):
    """Deconvolve data according to the given otf using a Gold-Meinel
    algorithm
    Parameters
    ----------
    data         : numpy array
    otf          : numpy array of the same size than data
    background   : background level
    iterations   : number of iterations
    acceleration : acceleration parameter
    Returns
    -------
    estimate   : estimated image
    dkl        : Kullback Leibler divergence

    [1] R. Gold. Rapp. tech. ANL-6984. Argonne National Lab., Ill., 1964.

    """
    from scipy.ndimage import gaussian_filter
    epsilon = 1e-6 # a little number
    estimate = np.maximum(np.real(np.fft.ifftn(otf * np.fft.fftn(data - background))), epsilon)
    dkl = np.zeros(iterations)
    for k in range(iterations):
        blurred = np.real(np.fft.ifftn(otf * np.fft.fftn(estimate + background)))
        ratio = data / blurred
        estimate = estimate * np.power(ratio, acceleration)
        estimate = gaussian_filter(estimate, smooth)
        dkl[k] = np.mean(blurred - data + data * np.log(np.maximum(ratio, epsilon)))
    return estimate, dkl

def deconvolve_richardson_lucy(data, otf, background=0, iterations=100):
    """Deconvolve data according to the given otf using a Richardson-Lucy
    algorithm
    Parameters
    ----------
    data       : numpy array
    otf        : numpy array of the same size than data
    background : background level
    iterations : number of iterations
    Returns
    -------
    estimate   : estimated image
    dkl        : Kullback Leibler divergence

    [1] W. H. Richardson, Bayesian-Based Iterative Method of Image Restoration,
        J. Opt. Soc. Am., vol. 62, no. 1, pp. 55–59, Jan. 1972,
        doi: 10.1364/JOSA.62.000055.
    [2] L. B. Lucy, An iterative technique for the rectification of observed
        distributions, The Astronomical Journal, vol. 79, p. 745, Jun. 1974,
        doi: 10.1086/111605.

    """
    epsilon = 1e-6 # a little number
    estimate = np.maximum(np.real(np.fft.ifftn(otf * np.fft.fftn(data-background))), epsilon)
    dkl = np.zeros(iterations)
    for k in range(iterations):
        blurred = np.maximum(np.real(np.fft.ifftn(otf * np.fft.fftn(estimate+background))), epsilon)
        ratio = data / blurred
        estimate = estimate * np.real(np.fft.ifftn(otf * np.fft.fftn(ratio)))
        estimate = np.maximum(estimate, epsilon)
        dkl[k] = np.mean(blurred - data + data * np.log(np.clip(ratio,a_min=1e-6,a_max=None)))
    return estimate, dkl

def deconvolve_richardson_lucy_heavy_ball(data, otf, background=0, iterations=100):
    """Deconvolve data according to the given otf using a scaled heavy ball
    Richardson-Lucy algorithm

    Parameters
    ----------
    data       : numpy array
    otf        : numpy array of the same size than data
    iterations : number of iterations

    Returns
    -------
    estimate   : estimated image
    dkl        : the kullback leibler divergence (should tend to 1/2)

    Note
    ----
    [1] H. Wang and P. C. Miller, Scaled Heavy-Ball Acceleration of the
        Richardson-Lucy Algorithm for 3D Microscopy Image Restoration, IEEE
        Transactions on Image Processing, vol. 23, no. 2, pp. 848-854, Feb. 2014,
        doi: 10.1109/TIP.2013.2291324.
    """
    old_estimate = np.clip(np.real(np.fft.ifftn(otf * np.fft.fftn(data - background))), a_min=0, a_max=None)
    estimate = data
    dkl = np.zeros(iterations)
    for k in range(iterations):
        beta = (k-1.0) / (k+2.0)
        prediction = estimate + beta * (estimate -  old_estimate)
        blurred = np.maximum(np.real(np.fft.ifftn(otf * np.fft.fftn(prediction + background))),0)
        ratio = data / blurred
        gradient = 1.0 - np.real(np.fft.ifftn(otf * np.fft.fftn(ratio)))
        old_estimate = estimate
        estimate = np.clip(prediction - estimate * gradient, a_min=0.1, a_max=None)
        dkl[k] = np.mean(blurred - data + data * np.log(np.clip(ratio,a_min=1e-6, a_max=None)))
    return estimate, dkl

def deconvolve_wiener(img, otf, snr):
    """Deconvolve the image using a Wiener filter

    Parameters
    ----------
    data       : numpy array
    otf        : numpy array of the same size than data
    snr        : signal to noise ratio
    Returns
    -------
    estimate   : estimated image
    """
    filter = np.conjugate(otf) / (np.square(np.abs(otf)) + (1./np.square(snr)))
    return np.real(np.fft.ifftn(filter * np.fft.fftn(img)))

def deconvolve_dr(img, otf, snr, max_iter=5):
    """Deconvolve the image

    Parameters
    ----------
    data       : numpy array
    otf        : numpy array of the same size than data
    snr        : signal to noise ratio
    Returns
    -------
    estimate   : estimated image
    """
    y = img.copy()
    gamma = 1. / snr
    filter = 1./ (1. + gamma * np.square(np.abs(otf)))
    Atf = gamma * np.conjugate(otf) * np.fft.fftn(img)
    prox1 = lambda x: np.real(np.fft.ifftn(filter * (np.fft.fftn(x) + Atf)))
    prox2 = lambda x: np.maximum(x,0)
    for _ in range(max_iter):
        x = prox2(y)
        y = y + 2. * (prox1(2*x-y)-x)
    return x


def deconvolve_total_variation(
        data:np.ndarray,
        otf:np.ndarray,
        background=0.,
        pixel_size:np.ndarray = np.ones([3,1]),
        regularization:float=0.5,
        max_iter:int=100,
        step_size:float=1,
        beta:float=0.1) -> np.ndarray:
    """Deconvolve the image with a total variation regularization
    Parameters
    ----------
    data       : numpy array
    otf        : numpy array of the same size than data
    background : background as float or nd.array
    Returns
    -------
    estimate   : estimated image
    """
    from scipy import ndimage
    epsilon = 1e-6
    alpha = np.array(pixel_size) / np.array(pixel_size).max()
    estimate = np.maximum(np.real(np.fft.ifftn(otf * np.fft.fftn(data-background))), epsilon)
    D = [np.array([0,-1,1]).reshape(s) for s in [[1,1,3],[1,3,1],[3,1,1]]]
    D = [d*a for d,a in zip(D,alpha)]
    Dstar = [np.array([-1,1,0]).reshape(s) for s in [[1,1,3],[1,3,1],[3,1,1]]]
    Dstar = [d*a for d,a in zip(Dstar,alpha)]
    Hstarf = np.real(np.fft.ifftn(np.conjugate(otf) * np.fft.fftn(data)))
    HtH = np.conjugate(otf) * otf
    for _ in range(max_iter):
        G = [ndimage.convolve(estimate, d, mode='reflect') for d in D]
        N = np.sqrt(sum([np.square(g) for g in G]) + beta)
        curv = sum([ndimage.convolve(g/N,d) for g,d in zip(G,Dstar)])
        veloc = np.real(np.fft.ifftn(HtH * np.fft.fftn(estimate))) - Hstarf - regularization * curv
        veloc = veloc / veloc.max()
        estimate = np.maximum(estimate - step_size * veloc, 0)
    return estimate


def deconvolve_richardson_lucy_total_variation(
        data:np.ndarray,
        otf:np.ndarray,
        background=0.,
        pixel_size:np.ndarray = np.ones([3,1]),
        regularization:float=0.5,
        max_iter:int=100,
        beta:float=0.1) -> np.ndarray:
    """Deconvolve the image with a total variation regularization
    Parameters
    ----------
    data       : numpy array
    otf        : numpy array of the same size than data
    background : background as float or nd.array
    pixel_size : pixel size used to scale the gradients
    regularization : regularization parameter
    max_iter   : number of iterations
    beta       : beta parameter in the TV norm
    Returns
    -------
    estimate   : estimated image
    """
    from scipy import ndimage
    epsilon = 1e-6
    alpha = np.array(pixel_size) / np.array(pixel_size).max()
    estimate = np.maximum(np.real(np.fft.ifftn(otf * np.fft.fftn(data-background))), epsilon)
    D = [np.array([0,-1,1]).reshape(s) for s in [[1,1,3],[1,3,1],[3,1,1]]]
    D = [d*a for d,a in zip(D,alpha)]
    Dstar = [np.array([-1,1,0]).reshape(s) for s in [[1,1,3],[1,3,1],[3,1,1]]]
    Dstar = [d*a for d,a in zip(Dstar,alpha)]
    Hstarf = np.real(np.fft.ifftn(np.conjugate(otf) * np.fft.fftn(data)))
    HtH = np.conjugate(otf) * otf
    for _ in range(max_iter):
        blurred = np.maximum(np.real(np.fft.ifftn(otf * np.fft.fftn(prediction + background))), epsilon)
        ratio = data / blurred
        G = [ndimage.convolve(estimate, d, mode='reflect') for d in D]
        N = np.sqrt(sum([np.square(g) for g in G]) + beta)
        curv = sum([ndimage.convolve(g/N,d) for g,d in zip(G,Dstar)])
        estimate = estimate * np.real(np.fft.ifftn(np.conjugate(otf) * np.fft.fft(ratio))) / (1. - regularization * curv)
    return estimate

# def zernike_polynomial(r,t,coefficients):
#     """  Evaluate Zernike polynomials.

#     Parameters
#     ----------
#     r : radius
#     t : angle
#     coefficients: coefficients (up to 15)

#     Result
#     ------
#     P : zernike polynomial evaluated on rho, theta
#     """
#     if coefficients is None:
#         return np.zeros(r.shape)
#     if coefficients.shape[0] < 15:
#         coefficients = np.pad(coefficients,(0,15-coefficients.shape[0]))
#     r2 = np.square(r)
#     r3 = np.power(r,3.0)
#     r4 = np.power(r,4.0)
#     c = np.cos(t)
#     s = np.sin(t)
#     c2 = np.cos(2*t)
#     s2 = np.sin(2*t)
#     c3 = np.cos(3*t)
#     s3 = np.sin(3*t)
#     P = np.zeros(r.shape)
#     P = P + coefficients[0] * r * c
#     P = P + coefficients[1] * r * s
#     P = P + coefficients[2] * (2. * r2 - 1.)
#     P = P + coefficients[3] * r2 * c2
#     P = P + coefficients[4] * r2 * s2
#     P = P + coefficients[5] * (3.*r2-2.) * r * c
#     P = P + coefficients[6] * (3.*r2-2.) * r * s
#     P = P + coefficients[7] * (6.*r4 - 6.*r2 + 1.)
#     P = P + coefficients[8] * r3 * c3
#     P = P + coefficients[9] * r3 * s3
#     P = P + coefficients[10] * (4.*r3 - 3) * r2 * c2
#     P = P + coefficients[11] * (4.*r3 - 3) * r2 * s2
#     P = P + coefficients[12] * (10 * r4 - 12 * r2 + 3) * r  * c
#     P = P + coefficients[13] * (10 * r4 - 12 * r2 + 3) * r  * s
#     P = P + coefficients[14] * ((20 * r4 - 30 * r2 + 12)  * r2 - 1)
#     P = np.where(r < 1.0, P, 0)
#     return P






# def generate_otf3d_grid(shape, pixel_size):
#     """
#     Generate a grid on which to evaluate the optical transfer function
#     Parameters
#     ----------
#     shape : list [nx,ny,nz] giving the shape of the final array
#     pixel_size : sampling in [x,y,z]
#     Returns
#     -------
#     kx,ky,kz,z: frequency grid in x and y and spatial space in z
#     """
#     f = [np.fft.fftfreq(n, pixel_size[k]) for k, n in enumerate(shape)]
#     ez = np.concatenate((np.arange(0,shape[2]//2), np.arange((-shape[2])//2,0))) * pixel_size[2]
#     [kx,ky,kz] = np.meshgrid(*f)
#     z = np.meshgrid(f[0],f[1],ez)[2]
#     return kx,ky,kz,z

# def generate_otf3d_on_grid(grid,numerical_aperture,wavelength,medium_refractive_index):
#     """
#     Generate a diffraction limited wide field optical transfer function and point spread function
#     Parameters
#     ----------
#     grid : tuple [kx,ky,z] with grid on which to evaluate the otf
#     pixel_size : sampling in [x,y,z]
#     numerical aperture : numerical aperture
#     wavelength : wavelength of the emitted light
#     medium_refractive_index : refractive index of the immersion medium
#     Returns
#     --------
#     otf : the optical transfer function as an array of shape 'shape'
#     psf : the point spread function as an array of shape 'shape' centerd in 0,0,0
#     """
#     d2 = np.square(grid[0]) + np.square(grid[1])
#     rho = np.sqrt(d2) * wavelength / numerical_aperture
#     P = np.where(rho <= 1.0, 1.0, 0.0)
#     defocus = grid[3] * np.sqrt(np.clip((medium_refractive_index / wavelength)**2 - d2,0,None))
#     psf = np.square(np.abs(np.fft.fft2(P * np.exp(2j * math.pi * defocus),axes=[0,1])))
#     psf = psf / psf.sum()
#     otf = np.fft.fftn(psf)
#     return otf, psf

# def generate_otf3d(shape,pixel_size,wavelength,NA,medium_refractive_index):
#     """
#     Generate a diffraction limited wide field optical transfer function and point spread function
#     Parameters
#     ----------
#     shape : list [nx,ny,nz] giving the shape of the final array
#     pixel_size : sampling in [x,y,z]
#     NA : numerical aperture
#     wavelength : wavelength of the emitted light
#     medium_refractive_index : refractive index of the immersion medium
#     Returns
#     --------
#     otf : the optical transfer function as an array of shape 'shape'
#     psf : the point spread function as an array of shape 'shape' centerd in 0,0,0
#     """
#     grid = generate_otf3d_grid(shape, pixel_size)
#     return generate_otf3d_on_grid(grid,NA,wavelength,medium_refractive_index)






# def deconvolve_richardson_lucy_heavy_ball_torch(data, otf, background, iterations, device):
#     """
#     Deconvolve data according to the given otf using a scaled heavy ball Richardson-Lucy algorithm
#     Parameters
#     ----------
#     data       : torch tensor
#     otf        : optical transfer function
#     background : bakground level
#     iterations : number of iterations
#     Result
#     ------
#     estimate   : estimated image
#     dkl        : the kullback leibler divergence (should tend to 1/2)
#     Note
#     ----
#     https://doi.org/10.1109/tip.2013.2291324
#     Needs pytorch (>1.9))
#     """
#     data = torch.from_numpy(data.astype(float),device=device)
#     otf = torch.from_numpy(otf.astype(complex),device=device)
#     old_estimate = torch.clamp(torch.real(torch.fft.ifftn(otf * torch.fft.fftn(data))), min=1e-6)
#     estimate = data
#     dkl = torch.zeros(iterations, dtype=float)
#     for k in range(iterations):
#         beta = (k-1.0) / (k+2.0)
#         prediction = estimate + beta * (estimate -  old_estimate)
#         blurred = torch.clamp(torch.real(torch.fft.ifftn(otf * torch.fft.fftn(prediction + background))), min=1e-6)
#         ratio = data / blurred
#         gradient = 1.0 - torch.real(torch.fft.ifftn(otf * torch.fft.fftn(ratio)))
#         old_estimate = estimate
#         estimate = torch.clamp(prediction - estimate * gradient, min=1e-6)
#         dkl[k] = torch.mean(blurred - data + data * torch.log(torch.clamp(ratio,min=1e-6)))
#     return estimate.numpy(), dkl.numpy()



