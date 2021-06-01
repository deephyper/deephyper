#!/bin/bash -x

source /lus/theta-fs0/software/thetagpu/conda/tf_master/2020-11-11/mconda3/setup.sh
conda config --set pip_interop_enabled False
conda create -p dhgpu --clone base -y
conda activate dhgpu/
pip install pip --upgrade
pip install deephyper