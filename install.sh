#! /bin/bash 

ENV_NAME="ModularNet"
PYTHON_VERSION=3.12
PYTHON_BIN=python3
PIP_BIN=pip3
VENV_DIR=./.venv

sudo apt install -y python3 python3-pip python3-venv
sudo apt install -y tk ffmpeg
sudo apt install -y python3-tk

mkdir ./checkpoint
mkdir ./data
mkdir ./video

# create env
if [[ ! -d $VENV_DIR ]]; then
    `$PYTHON_BIN -m venv $VENV_DIR --prompt $ENV_NAME`
fi

source $VENV_DIR/bin/activate
which $PIP_BIN
which $PYTHON_BIN
`$PIP_BIN install -r requirements.txt`
$PIP_BIN install -e modularnet/

pip install --upgrade kaleido
plotly_get_chrome -y
# PIP packages
#pip install --force-reinstall torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
#pip install --force-reinstall -r requirements.txt


# unnecessary
#conda deactivate 