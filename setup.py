from setuptools import setup, find_packages

setup(
    name='mug',  # Name of the package. This will be used, when the project is imported as a package.
    version='0.0.1',
    packages=find_packages(include=['mug'])  # Pip will automatically install the dependences provided here.
)