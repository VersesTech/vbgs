#!/bin/bash
cd .. && git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
echo $(which python) && \
pip install opencv-python plyfile && \
cd gaussian-splatting && \
git checkout 2eee0e26d2d5fd00ec462df47752223952f6bf4e  && \
git submodule update && \
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia && \
cd submodules/simple-knn && python setup.py install && \
cd ../diff-gaussian-rasterization && python setup.py install && \
cd ../../../vbgs && pip install -e .[gpu] 
