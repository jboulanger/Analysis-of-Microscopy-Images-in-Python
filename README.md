# Processing of microscopy images in python

These material aims at gathering tips to analysis of microscopy images. We'll see how to define simple analysis workflow from the raw image to quantitative number.

## Installation & setup
For python, we'll use a minimal version of miniconda which is a package manager for precompiled modules. For editing text, we use visual studio code.
- Install visual studio code https://code.visualstudio.com/download
- Activate python extensions
- Install miniconda3 https://docs.conda.io/en/latest/miniconda.html
- Create an environement either 
    - Install modules manually one by one. In the anaconda prompt (win key+type miniconda+enter), type 
        ```
        conda create --name imageanalysis
        conda activate imageanalsyis
        ```
        Let's add support for Jupyter kernel
        ```
        conda install ipykernel
        ```
        We'll also need scikit-image and matplotlib for basic image manipulation
        ```
        conda install scikit-image
        ```
  - We can also create the environement with all the necessary module using ```conda env create -f environment.yml```.  Note that to export the current environement to an yml file use: ```conda env export > environment.yml```

- We run the examples in a jupyter notebook in the browser. Open the anaconda prompt, go to the directory where the notebooks are saved by typing  ```cd path_to_the_folder```  and then launch the notebook with ```jupyter notebook``` .

## Loading microscopy images
Image data acquired in microscopy are stored in various formats: TIF, LSM, CZI, LIF, ND2. The notebook example1-loading-data.ipynb ges through a few examples using various existing modules.

## Visualization
Visualization of 2D images can be performed using the matplotlib library. matplotlib will also be able to render isosurfaces in 2D.
We'll use napari to visualize 3D data set using volume rendering.

## Enhancement

## Registration and displacement

## Segmentation


## 5. Creating automated workflow



