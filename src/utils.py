import scipy.ndimage
import numpy as np

def generate_test_image(shape,N=10,L=100,smooth=10):
    """
    Generate a test image with fibers

    Parameters
    ----------
    shape : list of int giving the shape of the generated image
    N : number of fibers
    L : number of points in each fiber
    smooth : smoothness of the fibers
    
    Result
    ------
    img : numpy array of given shape normalized between 0 and 1
    """
    D = len(shape)
    # generate points on smooth curves as a N x L x D array
    P = np.tile(np.repeat(np.reshape(np.array(shape), [1,1,D]), N, axis=0) * (.1 + .8 * np.random.rand(N,1,D)), [1,L,1])
    P = P + np.cumsum(scipy.ndimage.gaussian_filter1d(2*np.random.randn(N,L,D), smooth, axis=1),axis=1)
    space = tuple([np.arange(k) for k in shape])
    X = np.meshgrid(*space)
    img = np.zeros(shape)
    for p in np.reshape(P,(N*L,D)):
        img = img + np.exp(-0.5*np.sum(np.stack([np.square(p[k]-X[k]) for k in range(D)]), axis=0))
    img = img / img.max()
    return img

def estimate_awgn_std(data:np.array):
    """
     Estimate additive Gaussian white noise standard deviation 

     Parameters
     ----------
     data : numpy array
     
     Result
     ------
     sigma : estimate of the noise standard deviation
    """
    # compute pseudo residuals for each axis 
    # normalize the filter so that sum h^2=1 ( 1/sqrt(6) )
    h = 0.4082482904638631 * np.array([1.,-2.,1.]) 
    flt = data
    for axis in range(len(data.shape)) :
        flt = np.apply_along_axis(lambda x: np.convolve(x,h), axis, flt)
    # comute MAD / scipy.stats.norm.ppf(3/4)
    sigma = 1.482602218 * np.median(np.abs(flt-np.median(flt))) 
    return sigma
