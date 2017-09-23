#!/bin/bash
set -e

TORCH_URL="http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl"

nvidia-smi

pip install numpy
pip install Pillow
pip install tqdm
pip install scipy
pip install $TORCH_URL 
pip install torchvision
