

# Processing and analysis of microscopy images in python

These material aims at gathering tips to analysis of microscopy images. We'll see how to define simple analysis workflow from the raw image to quantitative number.

The objective is to be able to create workflows to analyse microscopy data.

You can launch the notebooks in binder  and start interacting with the code without installing anything on your computer.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jboulanger/Analysis-of-Microscopy-Images-in-Python/HEAD)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jboulanger/Analysis-of-Microscopy-Images-in-Python/)



## Installation & setup
To be able to run the code on your own compute, we can follow these steps:

1. Install [visual code](https://code.visualstudio.com/download)
2. Activate [extensions](https://code.visualstudio.com/docs/languages/python) in visual code for
[python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and
[notebooks](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
3. Install [miniconda3](https://docs.conda.io/en/latest/miniconda.html)
4. Download the code using git or directly [here](https://github.com/jboulanger/Analysis-of-Microscopy-Images-in-Python/archive/refs/heads/main.zip) and unzip it
5. Open visual code and open the folder Analysis-of-Microscopy-Images-in-Python in visual code
6. In visual code open a new terminal (on windows make sure the terminal is cmd and not PS). Create the environement installing necessary packages:
```
conda env create -f envs/linux/environment.yml
```
7. Register the jupyter kernel
```
conda activate imaging
python -m ipykernel install --user --name imaging
```
8. Open a notebook in the nbs folder, on the top right of the notebook you should be able to change the kernel from `Python` to `imaging`.



## Content
The notebooks in the example folder try to cover various aspects of image analysis encountered in optical/fluorescence microsopy.

- Basic python concepts
- Packages and environments
- Opening images with various microscopy formats
- Visualization of multi-dimensional datasets
- Image enhancement
- Deconvolution
- Segmentation (cells, nucleis, membranes)
- Motion estimation
- Tracking
- Statistical analysis



