#!/bin/bash

module load miniconda-3
conda create -p dh-theta python=3.8 -y
conda activate dh-theta/
conda install gxx_linux-64 gcc_linux-64 -y
# DeepHyper + Analytics Tools (Parsing logs, Plots, Notebooks)
pip install deephyper[analytics]
conda install tensorflow -c intel -y
