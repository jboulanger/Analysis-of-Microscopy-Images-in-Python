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
        dkl[k] = np.mean(blurred - data + data * np.log(np.maximum(ratio,0)))
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
    epsilon = 1e-6
    old_estimate = np.maximum(np.real(np.fft.ifftn(otf * np.fft.fftn(data - background))), epsilon)
    estimate = data
    dkl = np.zeros(iterations)
    for k in range(iterations):
        beta = (k-1.0) / (k+2.0)
        prediction = estimate + beta * (estimate -  old_estimate)
        blurred = np.maximum(np.real(np.fft.ifftn(otf * np.fft.fftn(prediction + background))), epsilon)
        ratio = data / blurred
        gradient = 1.0 - np.real(np.fft.ifftn(otf * np.fft.fftn(ratio)))
        old_estimate = estimate
        estimate = np.maximum(prediction - estimate * gradient, 0)
        dkl[k] = np.mean(blurred - data + data * np.log(np.maximum(ratio, epsilon)))
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
    alpha = np.array(pixel_size) / np.array(pixel_size).max()
    estimate = np.real(np.fft.ifftn(otf * np.fft.fftn(data-background)))
    D = [np.array([0,-1,1]).reshape(s) for s in [[1,1,3],[1,3,1],[3,1,1]]]
    D = [d*a for d,a in zip(D,alpha)]
    Dstar = [np.array([-1,1,0]).reshape(s) for s in [[1,1,3],[1,3,1],[3,1,1]]]
    Dstar = [d*a for d,a in zip(Dstar,alpha)]
    Hstarf = np.real(np.fft.ifftn(np.conjugate(otf) * np.fft.fftn(data)))
    HtH = np.conjugate(otf) * otf
    for _ in range(max_iter):
        G = [ndimage.convolve(estimate, d, mode='reflect') for d in D]
        N = np.sqrt(sum([np.square(g) for g in G]) + beta)
        curv = sum([ndimage.convolve(g / N, d) for g, d in zip(G, Dstar)])
        veloc = np.real(np.fft.ifftn(HtH * np.fft.fftn(estimate)))
        veloc = veloc - Hstarf - regularization * curv
        veloc = veloc / veloc.max()
        estimate = np.maximum(estimate - step_size * veloc, 0)
    return estimate


def deconvolve_richardson_lucy_total_variation(
        data:np.ndarray,
        otf:np.ndarray,
        background = 0.,
        regularization:float = 0.5,
        iterations:int = 100,
        beta:float = 0.1) -> np.ndarray:
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
    D = [np.array([0,-1,1]).reshape(s) for s in [[1,1,3],[1,3,1],[3,1,1]]]
    Dstar = [np.array([-1,1,0]).reshape(s) for s in [[1,1,3],[1,3,1],[3,1,1]]]
    epsilon = 1e-6 # a little number
    estimate = np.real(np.fft.ifftn(otf * np.fft.fftn(data-background)))
    dkl = np.zeros(iterations)
    for k in range(iterations):
        blurred = np.real(np.fft.ifftn(otf * np.fft.fftn(estimate+background)))
        ratio = data /  np.maximum(blurred, epsilon)
        estimate = estimate * np.real(np.fft.ifftn(otf * np.fft.fftn(ratio)))
        G = [ndimage.convolve(estimate, d, mode='reflect') for d in D]
        N = np.sqrt(sum([np.square(g) for g in G]) + beta)
        curv = sum([ndimage.convolve(g / N , d) for g,d in zip(G, Dstar)])
        estimate = estimate / np.maximum(1.- regularization * curv, epsilon)
        dkl[k] = np.mean(blurred - data + data * np.log(np.maximum(ratio, epsilon)))
    return estimate
