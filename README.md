# Variational Bayes Gaussian Splatting

This repository contains code accompanying the paper [Variational Bayes Gaussian Splatting](https://arxiv.org/abs/2410.03592
) by Toon Van de Maele, Ozan Catal, Alexander Tschantz, Christopher L. Buckley, and Tim Verbelen. 

If you find this repository helpful, please refer to our work using: 
```
@misc{vandemaele2024variationalbayesgaussiansplatting,
      title={Variational Bayes Gaussian Splatting}, 
      author={Toon Van de Maele and Ozan Catal and Alexander Tschantz and Christopher L. Buckley and Tim Verbelen},
      year={2024},
      eprint={2410.03592},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.03592}, 
}
```

## Installation 

The package for optimizing VBGS can be installed using the following command in the root folder: 
```
pip install -e . 
```

For the GPU version with cuda run: 
```
pip install -e .[gpu]
```

### Installation of the render environment

There is a dependency conflict between the torch cuda version and the jax cuda version. For this reason, we use a different virtual environment to run the cuda-based Gaussian splatting renderer which uses the GPU enabled rendering, and a cpu version of `vbgs`.

To use the renderer from the [Gaussian Splatting repository](https://github.com/graphdeco-inria/gaussian-splatting). Please create a new virtual environment and clone this repository at the same parent location as `vbgs`. Install the gaussian-splatting submodules (`simple-knn`, `diff-gaussian-rasterization`) in this new virtula environment by running `python setup.py install`. Now you can install the **`cpu`** version of `vbgs` within this environment. 

## Downloading the Data 

The image experiments pull the imagenet dataset using [Huggingface datasets](https://huggingface.co/docs/datasets/en/index) directly in the train script. 

For the 3D objects, the Blender dataset can be downloaded using [nerfstudio](https://docs.nerf.studio/quickstart/existing_dataset.html) and set the path accordingly in `scripts/config/blender.yaml`. 

For the 3D rooms, the Habitat test scenes can be downloaded and rendered using the [Dust3r data preprocessing pipeline](https://github.com/naver/dust3r/tree/main/datasets_preprocess/habitat). Set the path accordingly in `scripts/config/habitat.yaml`. `scripts/habitat_to_blender.py` contains code to transform this dataset into the Blender format.

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

Code for constructing the model for 3D data can be found in `scripts/model_volume.py`. Rendering of an object can be done using the `scripts/render_volume.ipynb` notebook in the render virtual environment.

Running the Blender objects benchmark and continual benchmark. This script stores the model after integrating each frame (continual) and at the end (full). 
```
python scripts/train_objects.py
```

### Room Experiments


Running the Habitat benchmark and continual benchmark is similar to the object case. This script stores the model after integrating each frame (continual) and at the end (full). Rendering of a room can be done using the `scripts/render_volume.ipynb` notebook in the render virtual environment.
.
```
python scripts/train_rooms.py
```

## License

Copyright 2024 VERSES AI, Inc.

Licensed under the VERSES Academic Research License (the “License”);
you may not use this file except in compliance with the license.

You may obtain a copy of the License at

    https://github.com/VersesTech/vbgs/blob/main/LICENSE.txt

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
