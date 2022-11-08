

# Processing and analysis of microscopy images in python

These material aims at gathering tips to analysis of microscopy images. We'll
see how to define simple analysis workflow from the raw image to quantitative number.

The objective is not to learn computer science, programing or even the python
language but to be able to create workflows to analyse microscopy data.

You can lunch the notebooks in binder [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jboulanger/Analysis-of-Microscopy-Images-in-Python/HEAD) and start interacting with the code without installing anything on your computer.


## Installation & setup
These steps are for illustrative purpose only and aimed at people not familiar with coding in python. For python, we can use a minimal version of miniconda which is a package manager for precompiled modules. For editing text, we use visual studio code.

- Install visual studio code https://code.visualstudio.com/download
- Activate python extensions in visual code ([])
- Install miniconda3 https://docs.conda.io/en/latest/miniconda.html
- Create an environement
	1. Install modules manually one by one (see notebooks) and install needed packages one by one. In the miniconda3 prompt or in visual code, opening a command prompt:
        ```
        conda create --name imageanalysis
        conda activate imageanalsyis
        ```
        and install the jupyter kernel
        ```
        conda install ipykernel
        python -m ipykernel install --user --name imageanalysis
        ```
	2. Or create the environment with all the necessary module typing in a terminal
        ```
        conda env create -f environment.yml
        ```
	Note that to export the current environment to an yml file use: ```conda env export > environment.yml```

- Download the code using git or directly [here](https://github.com/jboulanger/Analysis-of-Microscopy-Images-in-Python/archive/refs/heads/main.zip)

- We can run the examples in a jupyter notebook in the browser as well. Open the anaconda prompt, go to the directory where the notebooks are saved by typing  ```cd path_to_the_folder```  and then launch the notebook with ```jupyter notebook``` .

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
