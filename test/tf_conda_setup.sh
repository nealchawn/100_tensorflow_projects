#!/usr/bin/env bash
# shebang

source ~/.bashrc

conda create -n tf_gpu_wsl python=3.11 pip -y
conda activate tf_gpu_wsl

python -m pip install --upgrade pip
python -m pip install "tensorflow[and-cuda]"

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"