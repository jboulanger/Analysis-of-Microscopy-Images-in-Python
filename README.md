# Processing and analysis of microscopy images in python

These material aims at gathering tips to analysis of microscopy images. We'll see how to define simple analysis workflow from the raw image to quantitative number.

## Installation & setup
For python, we'll use a minimal version of miniconda which is a package manager for precompiled modules. For editing text, we use for example visual studio code.
- Install visual studio code https://code.visualstudio.com/download
- Activate python extensions
- Install miniconda3 https://docs.conda.io/en/latest/miniconda.html
- Create an environement either
    - Install modules manually one by one (see notebooks). In the anaconda prompt (win key+type miniconda+enter), type
        ```
        conda create --name imageanalysis
        conda activate imageanalsyis
        ```
        Let's add support for Jupyter kernel
        ```
        conda install ipykernel
        ```
		Later add one by one the needed packages
  - We can also create the environement with all the necessary module using ```conda env create -f environment.yml```.  Note that to export the current environement to an yml file use: ```conda env export > environment.yml```
- Download the code using git or [here](https://github.com/jboulanger/Analysis-of-Microscopy-Images-in-Python/archive/refs/heads/main.zip)
- We can run the examples in a jupyter notebook in the browser. Open the anaconda prompt, go to the directory where the notebooks are saved by typing  ```cd path_to_the_folder```  and then launch the notebook with ```jupyter notebook``` .

## Content
The notebooks in the example folder try to cover various aspects of image analysis encountered in optical/fluorescence microsopy.
