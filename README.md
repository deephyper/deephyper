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

How to define your own autotuning problem
=========================================
This will be illustrated with the example in /benchmarks/b1 directory. 

In this example, we want to tune the network in addition_rnn.py that gets the following command line parameters and returns the output value
```
usage: addition_rnn.py [-h] [-v] [--backend [{tensorflow,theano,cntk}]]
                       [--activation [{softmax,elu,selu,softplus,softsign,relu,tanh,sigmoid,hard_sigmoid,linear,LeakyReLU,PReLU,ELU,ThresholdedReLU}]]
                       [--loss [{mse,mae,mape,msle,squared_hinge,categorical_hinge,hinge,logcosh,categorical_crossentropy,sparse_categorical_crossentropy,binary_crossentropy,kullback_leibler_divergence,poisson,cosine_proximity}]]
                       [--epochs [EPOCHS]] [--batch_size [BATCH_SIZE]]
                       [--init [{Zeros,Ones,Constant,RandomNormal,RandomUniform,TruncatedNormal,VarianceScaling,Orthogonal,Identity,lecun_uniform,glorot_normal,glorot_uniform,he_normal,lecun_normal,he_uniform}]]
                       [--dropout [DROPOUT]]
                       [--optimizer [{sgd,rmsprop,adagrad,adadelta,adam,adamax,nadam}]]
                       [--clipnorm [CLIPNORM]] [--clipvalue [CLIPVALUE]]
                       [--learning_rate [LR]] [--momentum [MOMENTUM]]
                       [--decay [DECAY]] [--nesterov [NESTEROV]] [--rho [RHO]]
                       [--epsilon [EPSILON]] [--beta1 [BETA1]]
                       [--beta2 [BETA2]] [--rnn_type [{LSTM,GRU,SimpleRNN}]]
                       [--hidden_size [HIDDEN_SIZE]] [--layers [LAYERS]]

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  --backend [{tensorflow,theano,cntk}]
                        Keras backend
  --activation [{softmax,elu,selu,softplus,softsign,relu,tanh,sigmoid,hard_sigmoid,linear,LeakyReLU,PReLU,ELU,ThresholdedReLU}]
                        type of activation function hidden layer
  --loss [{mse,mae,mape,msle,squared_hinge,categorical_hinge,hinge,logcosh,categorical_crossentropy,sparse_categorical_crossentropy,binary_crossentropy,kullback_leibler_divergence,poisson,cosine_proximity}]
                        type of loss
  --epochs [EPOCHS]     number of epochs
  --batch_size [BATCH_SIZE]
                        batch size
  --init [{Zeros,Ones,Constant,RandomNormal,RandomUniform,TruncatedNormal,VarianceScaling,Orthogonal,Identity,lecun_uniform,glorot_normal,glorot_uniform,he_normal,lecun_normal,he_uniform}]
                        type of initialization
  --dropout [DROPOUT]   float [0, 1). Fraction of the input units to drop
  --optimizer [{sgd,rmsprop,adagrad,adadelta,adam,adamax,nadam}]
                        type of optimizer
  --clipnorm [CLIPNORM]
                        float >= 0. Gradients will be clipped when their L2
                        norm exceeds this value.
  --clipvalue [CLIPVALUE]
                        float >= 0. Gradients will be clipped when their
                        absolute value exceeds this value.
  --learning_rate [LR]  float >= 0. Learning rate
  --momentum [MOMENTUM]
                        float >= 0. Parameter updates momentum
  --decay [DECAY]       float >= 0. Learning rate decay over each update
  --nesterov [NESTEROV]
                        boolean. Whether to apply Nesterov momentum?
  --rho [RHO]           float >= 0
  --epsilon [EPSILON]   float >= 0
  --beta1 [BETA1]       float >= 0
  --beta2 [BETA2]       float >= 0
  --rnn_type [{LSTM,GRU,SimpleRNN}]
                        type of RNN
  --hidden_size [HIDDEN_SIZE]
                        number of epochs
  --layers [LAYERS]     number of epochs
```

The search space and a default starting point is defined in problem.py

```
class Problem():
    def __init__(self):
        space = OrderedDict()
        
        #bechmark specific parameters
        space['rnn_type'] = ['LSTM', 'GRU', 'SimpleRNN']
        space['hidden_size'] = (10, 100)
        space['layers'] = (1, 30)
        #network parameters
        space['activation'] = ['softmax', 'elu', 'selu', 'softplus', 'relu', 'tanh', 'sigmoid']
        #space['loss'] = ['mse', 'mae', 'mape', 'msle', 'squared_hinge', 'categorical_hinge', 'hinge', 'logcosh', 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'binary_crossentropy', 'kullback_leibler_divergence', 'poisson', 'cosine_proximity']
        space['epochs'] = (2, 10)
        space['batch_size'] = (8, 1024)
        #space['init'] = ['Zeros', 'Ones', 'Constant', 'RandomNormal', 'RandomUniform', 'TruncatedNormal', 'VarianceScaling', 'Orthogonal', 'Identity', 'lecun_uniform', 'glorot_normal', 'glorot_uniform', 'he_normal', 'lecun_normal', 'he_uniform']
        #space['dropout'] = (0.0, 1.0)
        space['optimizer'] = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']
        # common optimizer parameters
        space['clipnorm'] = (1e-04, 1e01)
        space['clipvalue'] = (1e-04, 1e01)
        # optimizer parameters
        space['learning_rate'] = (1e-04, 1e01)
        space['momentum'] =  (0, 1e01)
        space['decay'] =  (0, 1e01)
        space['nesterov'] = [False, True]
        space['rho'] = (1e-04, 1e01)
        space['epsilon'] = (1e-08, 1e01)
        space['beta1'] = (1e-04, 1e01)
        space['beta2'] = (1e-04, 1e01)

        self.space = space
        self.params = self.space.keys()
        self.starting_point = ['LSTM', 10, 1, 'softmax', 5, 32, 'sgd', 1.0, 0.5, 0.01, 0, 0, False, 0.9, 1e-08, 0.9, 0.999]
```
In evalaute.py, you have to define three functions.

First, define how to construct the command line in 
```
def commandLine(x, params) 
```

Second, define how to evalaute a point in
```
def evaluate(x, evalCounter, params, prob_dir, job_dir, result_dir): 
```

Third, define how to read the results in 
```
def readResults(fname, evalnum):
```

Finally, in job.tmpl, call the executable (see the example)
