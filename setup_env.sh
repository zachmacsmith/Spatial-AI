#!/bin/bash

echo "======================================================================"
echo "SETTING UP NEW ENVIRONMENT: egoenv_clean (Python 3.10)"
echo "======================================================================"

# 1. Create Conda Environment
echo "Creating conda environment..."
conda create -n egoenv_clean python=3.10 -y

# 2. Install FFmpeg (Crucial for video processing)
echo "Installing ffmpeg..."
conda install -n egoenv_clean -c conda-forge ffmpeg -y

# 3. Install Python Dependencies
echo "Installing python dependencies..."
# We use the full path to pip to ensure we install into the new environment
# regardless of which environment is currently active in the shell.
# Assuming standard anaconda/miniconda path structure.
# We try to find the env path first.

ENV_PATH=$(conda info --base)/envs/egoenv_clean
PIP_PATH="$ENV_PATH/bin/pip"

if [ ! -f "$PIP_PATH" ]; then
    # Fallback for some systems where envs are elsewhere
    PIP_PATH=$(conda env list | grep egoenv_clean | awk '{print $NF}')/bin/pip
fi

echo "Using pip at: $PIP_PATH"
"$PIP_PATH" install -r requirements_v2.txt

echo "======================================================================"
echo "SETUP COMPLETE"
echo "======================================================================"
echo ""
echo "To use the new environment, run:"
echo "conda activate egoenv_clean"
echo ""
