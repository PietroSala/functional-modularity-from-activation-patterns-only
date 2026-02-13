#! /bin/bash 

ENV_NAME="StrangeNet"
PYTHON_VERSION=3.11
PYTHON_BIN=python$PYTHON_VERSION
VENV_DIR=./venv

sudo dnf install -y $PYTHON_BIN
sudo dnf install -y tk ffmpeg
sudo dnf install -y $PYTHON_BIN-tkinter

mkdir ./checkpoint
mkdir ./data
mkdir ./video

# create env
if [[ ! -d $VENV_DIR ]]; then
    `$PYTHON_BIN -m venv $VENV_DIR --prompt $ENV_NAME`
fi

source ./venv/bin/activate
which pip
which python
pip install -r requirements.txt


# PIP packages
#pip install --force-reinstall torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
#pip install --force-reinstall -r requirements.txt


# unnecessary
#conda deactivate 