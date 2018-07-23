## Cooley

```
soft add +anaconda
```


```
conda create --name deephyper-cooley intelpython3_core  python=3.6
source activate deephyper-cooley
conda install h5py scikit-learn pandas mpi4py

conda config --add channels conda-forge
conda install tensorflow-gpu absl-py
conda install keras scikit-optimize
conda install xgboost deap

cd hpc-edge-service
pip install -e .

pip install filelock
pip install git+https://github.com/tkipf/keras-gcn.git
```

## Jupyter