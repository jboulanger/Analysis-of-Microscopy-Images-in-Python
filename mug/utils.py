import math
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image


def download_and_unzip(url, extract_to='.'):
    """Download and unzip an zip file
    Parameters
    ----------
    url : the location of the zip file
    extact_to : path where to extract the zip file
    """
    from urllib.request import urlopen
    from io import BytesIO
    from zipfile import ZipFile
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)


def show_image_list(images, titles):
    """Display an image list with titles and no axes"""
    _, ax = plt.subplots(1,len(images))
    for k in range(len(images)):
        ax[k].imshow(images[k], cmap='gray')
        ax[k].axis('off')
        ax[k].set_title(titles[k])


def mip3d(image):
    """Compute 3 maximum intensity projection and make a montage
    Parameters
    ----------
    image : image (3d) [Z,Y.X]
    Results
    -------
    a all projections arranged in as a single 2d image
    """
    xy = np.amax(image,0).squeeze()
    yz = np.amax(image,1).squeeze().transpose()
    xz = np.amax(image,2).squeeze()
    zz = np.zeros([image.shape[0],image.shape[0]])
    return np.concatenate(
        (np.concatenate((xy,xz),0), np.concatenate((yz,zz),0)),
        1
    )


def slice3d(image):
    """Compute a slice of 3D image
    Parameters
    ----------
    image : image (3d) [Z,Y.X]
    Results
    -------
    all slices arranged in a single 2d image
    """
    xy = image[image.shape[0]//2,:,:].squeeze()
    yz = image[:,image.shape[1]//2,:].squeeze().transpose()
    xz = image[:,:,image.shape[2]//2].squeeze()
    zz = np.zeros([image.shape[0],image.shape[0]])
    return np.concatenate(
        (np.concatenate((xy,xz),0), np.concatenate((yz,zz),0)),
        1
    )


# Register extra colormaps
for c in ['red', 'green', 'blue']:
    if c not in mpl.colormaps:
        mpl.colormaps.register(cmap=LinearSegmentedColormap.from_list(c, ['black',c]))
if 'gray' not in mpl.colormaps:
        mpl.colormaps.register(cmap=LinearSegmentedColormap.from_list('gray', ['black','white']))


def normalize_contrast(array:np.ndarray, saturation=0):
    array = array.astype(float)
    try:
        _ = iter(saturation)
    except TypeError:
        amin, amax = np.percentile(array, [saturation, 100.0-saturation])
    else:
         amin, amax = np.percentile(array, saturation)

    if amax != amin:
        return np.clip((array - amin) / (amax - amin),0,1)
    else:
        return np.clip(array,0,1)

def cmap_to_colormap(name):
    """Convert a colormap name to a """
    try:
        _ = iter(name)
    except TypeError:
        return mpl.colormaps[name]
    else:
        return [mpl.colormaps[x] for x in name]


def to_pil(array:np.ndarray, cmap, saturation=0) -> Image:
    """Convert an array [C,X,Y] to a PIL RGB image"""
    colors = cmap_to_colormap(cmap)
    if len(array.shape) == 2:
        img = cmap(normalize_contrast(array, saturation))
    else:
        img = sum([
            f(normalize_contrast(x, saturation)) for x,f in zip(array, colors)
        ])
    return Image.fromarray((255*img).astype(np.uint8))


def imshow(img,cmap, saturation=0):
    plt.imshow(to_pil(img, cmap, saturation))
    plt.axis('off')


def power_spectrum_density(x, applylog=True):
    """Compute the power spectrum density of the input
    Parameters
    ----------
    x : signal/image
    Returns
    -------
    the power spectrum
    """
    psd = np.fft.fftshift(np.abs(np.fft.fftn(x)))
    if applylog:
        return np.log(1e-6 + psd)
    else:
        return psd


def train_network(model, optimizer, loss_fn, train_dl, valid_dl, epochs=100, device='cpu', scheduler=None):
    """Training loops"""
    history = {
        'train_loss':[], 'train_epoch':[],
        'valid_loss':[], 'valid_epoch':[],
        'learning_rate': []
    }
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_dl:
            loss = loss_fn(model(batch[0].to(device)), batch[1].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item()
            loss.detach()
            if scheduler is not None:
                scheduler.step()
        history['train_loss'].append(train_loss)
        history['train_epoch'].append(epoch)
        history['learning_rate'].append(scheduler.get_last_lr()[0])
        if (epoch+1)%10 == 0:
            model.eval()
            valid_loss = 0
            for batch in valid_dl:
                valid_loss += loss_fn(model(batch[0].to(device)), batch[1].to(device)).detach().item()
            history['valid_loss'].append(valid_loss)
            history['valid_epoch'].append(epoch)


    return history


def lr_finder(model,optimizer,loss_fn,dl,device='cpu',lrmin=1e-6,lrmax=1):
    """Learning rate finder

    Increases the learning rate using a geometric progression following the
    paper Cyclical Learning Rates for Training Neural Networks from L. Smith
    https://arxiv.org/abs/1506.01186

    The increase of the learning rate is implemented using a lambda scheduler.

    Parameter
    ---------
    model : network architecture
    optimizer : optimizer initiallized with lrmin
    loss_fn : loss function
    device  : device on which to run the calculations
    lrmin   : minimum of the range of the learning rate
    lrmax   : maximum of the range of the learning rate

    Results
    -------
    lropt : optimal learning rate found
    history : dictionnary with loss and learning rate

    """
    import numpy as np
    from torch import optim
    from scipy.ndimage import gaussian_filter1d

    f = lambda  k: (lrmax/lrmin) ** ((k)/(len(dl)-1))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, f)

    history = {
        'loss':[],
        'learning rate': []
    }

    for batch in dl:
        loss = loss_fn(model(batch[0].to(device)), batch[1].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss.detach()
        history['loss'].append(loss.item())
        history['learning rate'].append(scheduler.get_last_lr()[0])
        scheduler.step()

    y = np.log(np.array(history['loss']))
    opt = np.argmin(gaussian_filter1d(y, 2))
    lropt = history['learning rate'][opt]
    return lropt, history


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

