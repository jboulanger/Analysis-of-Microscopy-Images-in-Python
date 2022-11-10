import tifffile
from pathlib import Path

def load_tiff(filename):
    return tifffile.imread(filename)

def load_nd2(filename):
    with ND2Reader('../data/example.nd2') as images:
        data = np.array()


# Define a mapping between extensions and loading function
img_formats = {
    'tif' : load_tiff,
    'nd2' : load_nd2
}

# define the generic loading function
def load_image(filename):
    """
    Load image from a file from tiff, nd2, lif and czi files into a numpy array

    Parameter
    ---------
    filename : the filename

    Result
    ------
    data : a 5D numpy array T P C Z Y X
    """
    # get the file extension in lower case
    ext = Path(filename).suffix.lower()
    # use the command from img_format dictionnary
    return img_formats[ext]
    


