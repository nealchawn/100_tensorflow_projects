#!/usr/bin/env bash

set -euo pipefail

CONDA_HOME="$HOME/miniconda3"
INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"

echo "Updating Ubuntu packages..."
sudo apt update
sudo apt upgrade -y

echo "Installing development tools..."
sudo apt install -y \
  build-essential \
  ca-certificates \
  curl \
  git

echo "Checking WSL GPU access..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
  /usr/lib/wsl/lib/nvidia-smi
else
  echo "ERROR: NVIDIA GPU is not visible inside WSL."
  echo "Update WSL and the Windows NVIDIA driver before continuing."
  exit 1
fi

if [[ ! -x "$CONDA_HOME/bin/conda" ]]; then
  echo "Installing Miniconda..."

  curl -fsSL \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -o "$INSTALLER"

  bash "$INSTALLER" -b -p "$CONDA_HOME"
  rm -f "$INSTALLER"
else
  echo "Miniconda is already installed at $CONDA_HOME."
fi

echo "Initializing Conda for Bash..."
"$CONDA_HOME/bin/conda" init bash
"$CONDA_HOME/bin/conda" config --set auto_activate_base false

echo
echo "WSL development setup complete."
echo "Run: source ~/.bashrc"
echo "Then create your project environment."