#!/bin/bash

module load conda/2021-09-22
conda create -p dhgpu --clone base -y
conda activate dhgpu/
pip install pip --upgrade
pip install deephyper