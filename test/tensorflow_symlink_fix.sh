#!/usr/bin/env bash
TF_DIR="$(dirname "$(python -c 'import tensorflow; print(tensorflow.__file__)')")"

echo "$TF_DIR"

cd "$TF_DIR"

ln -svf ../nvidia/*/lib/*.so* .


## get gpu working
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"

NVIDIA_LIBS="$(
  find "$SITE/nvidia" -type d -name lib -print |
  paste -sd: -
)"

export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$NVIDIA_LIBS"

python test_gpu.py