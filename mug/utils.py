import scipy.ndimage
import numpy as np

def generate_wavy_circle_contour(x0,y0,radius,amplitude,smoothness,length):
    """Generate a awvy circle contour

    Example :
    im = np.zeros((400,300))
    x, y = generate_wavy_circle_contour(200,150,100,10,0.5,128)
    rr,cc = polygon(x.ravel(),y.ravel(),im.shape)
    im[rr,cc] = 1
    plt.imshow(im)
    """
    t = np.linspace(0,2*math.pi,length).reshape((length,1))
    f = np.exp(-smoothness*length*np.abs(np.fft.fftfreq(length))).reshape((length,1))
    circle = radius * np.cos(t) + 1j * radius * np.sin(t)
    s = circle + x0 + 1j * y0
    s = s + amplitude * rng.normal(0,0.1,size=(length,1)) * circle
    s = np.fft.ifftn(f*np.fft.fftn(s))
    return np.real(s),np.imag(s)

def generate_nuclei2D_image(shape):
    from skimage.filters import gaussian
    im = np.zeros(shape)
    for k in range(100):
        x0 = im.shape[1] * rng.uniform()
        y0 = im.shape[0] * rng.uniform()
        x, y = generate_wavy_circle_contour(x0,y0,50,5,0.5,64)
        rr,cc = polygon(y.ravel(),x.ravel(),im.shape)
        if np.all(im[rr,cc]==0):
            im[rr,cc] = k+1

    fx,fy = np.meshgrid(np.fft.fftfreq(im.shape[1]),np.fft.fftfreq(im.shape[0]))

    g = 1/(1+10*np.sqrt((np.power(fx,2)+np.power(fy,2))))
    texture = np.real(np.fft.ifftn(g*np.fft.fftn(rng.normal(size=im.shape))))
    texture = (texture - texture.min()) / (texture.max()-texture.min())
    texture = np.fmax(texture - 0.3, 0)
    im = gaussian((im>0) * texture, 1)

    #plt.imshow(im*texture,cmap='gray')
    #plt.imsave("/home/jeromeb/Desktop/nuk.png",im,cmap="gray")

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
