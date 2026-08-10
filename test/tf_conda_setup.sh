#!/usr/bin/env bash
# shebang

source ~/.bashrc

conda create -n tf_gpu_wsl python=3.11 pip -y
conda activate tf_gpu_wsl

python -m pip install --upgrade pip
python -m pip install "tensorflow[and-cuda]"

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"


## setup jupyter nodebook + vscode
## https://ipython.readthedocs.io/en/stable/install/kernel_install.html?utm_source=chatgpt.com

conda activate tf_gpu_wsl

conda install -c conda-forge ipykernel -y


conda install -c conda-forge jupyterlab ipykernel -y
conda install -c conda-forge jupyterlab -y
jupyter lab

## Registering a Conda environment this way creates a kernel entry that can be selected by its display name in Jupyter and VS Code.
## The --name value is used by Jupyter internally. These commands will overwrite any existing kernel with the same name.
## --display-name is what you see in the notebook menus.

python -m ipykernel install \
  --user \
  --name tf_gpu_wsl \
  --display-name "Python (tf_gpu_wsl)"