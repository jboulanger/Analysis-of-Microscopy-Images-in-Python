import numpy as np
import math

def fibers(
        shape,
        pixel_size = [1., 1., 1.],
        N = 10,
        L = 100,
        smooth = 10.0):
    """ Generate a test image with fibers
    Parameters
    ----------
    shape : list of int giving the shape of the generated image
    N : number of fibers
    L : number of points in each fiber
    smooth : smoothness of the fibers along their length
    pixel_size : image pixel size in nm
    Result
    ------
    img : numpy array of given shape normalized between 0 and 1
    """
    from scipy.ndimage import gaussian_filter1d
    D = len(shape)

    alpha = np.reshape(np.array(pixel_size, dtype=np.float), [1, D])
    alpha = np.min(alpha) / alpha  # this will be broadcasted to N,L,D

    sigma = np.array(pixel_size, dtype=np.float)
    sigma = 0.75 * np.max(sigma) / sigma

    # generate points on smooth curves as a N x L x D array
    P = 0.1 + 0.8 * np.random.rand(N, 1, D)
    P *= np.repeat(np.reshape(np.array(shape), [1, 1, D]), N, axis=0)
    P = np.tile(P, [1, L, 1])
    step = np.random.randn(N, L, D)
    step = gaussian_filter1d(step, smooth, axis=1)
    vel = np.sqrt(np.sum(np.square(step), axis=2)).reshape(N,L,1)
    step = step / (2.0 * vel) * alpha
    P = P + np.cumsum(step, axis=1)
    P = P.reshape([N * L, D])

    # bounding boxes
    block_shape = [3, 3, 3]

    # keep points whose bounding box will not overalap
    idx = [
        np.all([
            (p[k] > -block_shape[k]//2) and (p[k] < (shape[k] + block_shape[k]//2))
            for k in range(3)
            ])
        for p in P
        ]
    P = P[idx, :]

    # compute bounding boxes
    bbox = [
        [
            slice(
                int(max(0, p[k] - block_shape[k] // 2)),
                int(min(p[k] + block_shape[k] // 2 + 1, shape[k])),
                1)
            for k in range(3)
        ]
        for p in P
    ]

    # compute the coordinate grid
    X = np.meshgrid(*[np.arange(n) for n in shape], indexing='ij')
    img = np.zeros(shape, dtype=np.float)
    for p, b in zip(P, bbox):
        cX = [x[tuple(b)] for x in X]
        deltas = [
            np.square((p[k] - block_shape[k] // 2 - cX[k]) / sigma[k])
            for k in range(D)
        ]
        tile = np.exp(-0.5 * np.sum(np.stack(deltas), axis=0))
        img[tuple(b)] += tile

    img = (img - img.min()) / (img.max() - img.min())
    return img



def wavy_circle_contour(x0,y0,radius,amplitude,smoothness,length):
    """Generate a wavy circle contour

    Example
    -------
    im = np.zeros((400,300))
    x, y = wavy_circle_contour(200,150,100,10,0.5,128)
    rr,cc = polygon(x.ravel(),y.ravel(),im.shape)
    im[rr,cc] = 1
    plt.imshow(im)
    """
    from numpy.random import default_rng
    rng = default_rng()
    t = np.linspace(0,2*math.pi,length).reshape((length,1))
    f = np.exp(-smoothness*length*np.abs(np.fft.fftfreq(length))).reshape((length,1))
    circle = radius * np.cos(t) + 1j * radius * np.sin(t)
    s = circle + x0 + 1j * y0
    s = s + amplitude * rng.normal(0,0.1,size=(length,1)) * circle
    s = np.fft.ifftn(f*np.fft.fftn(s))
    x = np.real(s)
    y = np.imag(s)
    return x, y

def nuclei2D(shape, N):
    from skimage.filters import gaussian
    from skimage.draw import polygon
    from numpy.random import default_rng
    rng = default_rng()
    labels = np.zeros(shape)
    for k in range(N):
        x0 = labels.shape[1] * rng.uniform()
        y0 = labels.shape[0] * rng.uniform()
        x, y = wavy_circle_contour(x0,y0,50,5,0.5,64)
        rr,cc = polygon(y.ravel(),x.ravel(),labels.shape)
        if np.all(labels[rr,cc]==0):
            labels[rr,cc] = k+1

    fx,fy = np.meshgrid(np.fft.fftfreq(labels.shape[1]),np.fft.fftfreq(labels.shape[0]))
    g = 1/(1+10*np.sqrt((np.power(fx,2)+np.power(fy,2))))
    texture = np.real(np.fft.ifftn(g*np.fft.fftn(rng.normal(size=labels.shape))))
    texture = (texture - texture.min()) / (texture.max()-texture.min())
    texture = np.fmax(texture - 0.3, 0)
    im = gaussian((labels>0) * texture, 1)
    return im, labels
