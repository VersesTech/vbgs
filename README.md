# Variational Bayes Gaussian Splatting

This repository contains code accompanying the paper "Variational Bayes Gaussian Splatting" 

## Installation 

The package for optimizing VBGS can be installed using the following command int he root folder: 
```
pip install -e . 
```

For the GPU version with cuda run: 
```
pip install -e .[gpu]
```

Note: There is a dependency conflict between the torch cuda version and the jax cuda version. To use the renderer from the [Gaussian Splatting repository](https://github.com/graphdeco-inria/gaussian-splatting), please create a separate virtual environment and install this as instructed. Then install the `cpu` version of this repository and run the scripts for rendering within this environment. 

## Model Training  

The scripts for training can be found in the `scripts` folder. [Hydra](hydra.cc) is used for configuring the parameters, and each script has an accompanying config file in `scripts/configs`. 

### Image Experiments

Code for constructing the model for image data can be found in `scripts/model_image.py`.

Running the ImageNet benchmark: 
```
python scripts/train_image.py
```
Running the ImageNet continual learning benchmark: 
```
python scripts/train_image_continual.py
```

### Object Experiments

Code for constructing the model for 3D data can be found in `scripts/model_volume.py`.
