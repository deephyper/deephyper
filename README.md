Directory structure 
===================
```
benchmarks
    directory for problems
experiments
    directory for saving the running the experiments and storing the results
search
    directory for source files
```
Install instructions
====================

With anaconda do the following:

```
conda create -n dl-hps python=3
source activate dl-hps
conda install h5py
conda install scikit-learn
conda install pandas
conda install mpi4py
conda install -c conda-forge keras
conda install -c conda-forge scikit-optimize
git clone https://github.com/scikit-optimize/scikit-optimize.git
cd scikit-optimize
pip install -e.
```
Usage
=====
```
cd search

usage: async-search.py [-h] [-v] [--prob_dir [PROB_DIR]] [--exp_dir [EXP_DIR]]
                       [--exp_id [EXP_ID]] [--max_evals [MAX_EVALS]]
                       [--max_time [MAX_TIME]]

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  --prob_dir [PROB_DIR]
                        problem directory
  --exp_dir [EXP_DIR]   experiments directory
  --exp_id [EXP_ID]     experiments id
  --max_evals [MAX_EVALS]
                        maximum number of evaluations
  --max_time [MAX_TIME]
                        maximum time in secs
```
Example
=======
```
mpiexec -np 2 python async-search.py --prob_dir=../benchmarks/b1 --exp_dir=../experiments/ --exp_id=exp-01 --max_evals=10 --max_time=60 
```