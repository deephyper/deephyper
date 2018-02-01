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
conda install -c conda-forge xgboost 
```

Usage (with Balsam)
=====================


Run once 
----------
```    
    source activate dl-hps   # balsam is installed here too (commands like “balsam ls” must work)

    cd directory_containing_dl-hps
    mv dl-hps dl_hps         # important: change to underscore (import system relies on this)
    export PYTHONPATH=$(pwd) # now dl_hps package is importable from anywhere 

    cd dl_hps/search
    balsam app --name eval_point --description "run dl_hps.search.worker.py to evaluate a point from disk" --executable worker.py
    balsam app --name hps-driver --description "run async_driver" --executable async-search.py
```

From a qsub bash script (or in this case, an interactive session)
----------------------------------------------------------------------
```
    qsub -A datascience -n 8 -t 60 -q debug-cache-quad -I 

    source ~/.bash_profile    # this should set LD_library_path correctly for mpi4py and make conda available (see balsam quickstart guide)
    source activate dl-hps   # balsam is installed here too (commands like “balsam ls” should work)

    cd directory_containing_dl_hps
    export PYTHONPATH=$(pwd) # now dl_hps package is importable from anywhere

    balsam job --name dl_driver --workflow "dl-hps" --app "hps-driver" --wall-minutes 5 --num-nodes 1 --ranks-per-node 1 --args '--exp_dir=~/workflows/dl_hps/experiments --exp_id=exp-01 --max_evals=10 --max_time=6000’

    # notice that above, we are referring to the driver itself only taking 1 rank/1 node (not the whole workflow)
    balsam launcher --consume --max-ranks-per-node 4   
    # will auto-recognize 8 nodes and allow only 4 addition_rnn.py tasks to run simultaneously on a node
```
