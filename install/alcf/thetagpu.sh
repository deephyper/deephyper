#!/bin/bash

module load conda/2022-07-01
conda create -p dhgpu --clone base -y
conda activate dhgpu/
pip install pip --upgrade
pip install deephyper