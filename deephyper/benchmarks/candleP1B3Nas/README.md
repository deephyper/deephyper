## P1B3: MLP Regression Drug Response Prediction

**Overview**: Given drug screening results on NCI60 cell lines, build a deep learning network that can predict the growth percentage from cell line gene expression data, drug concentration and drug descriptors.

**Relationship to core problem**: This benchmark is a simplified form of the core drug response prediction problem in which we need to combine multiple molecular assays and a diverse array of drug feature sets to make a prediction.

**Expected outcome**: Build a DNN that can predict growth percentage of a cell line treated with a new drug.

### Benchmark Specs Requirements

#### Description of the Data
* Data source: Dose response screening results from NCI; 5-platform normalized expression data from NCI; Dragon7 generated drug descriptors based on 2D chemical structures from NCI
* Input dimensions: ~30K with default options: 26K normalized expression levels by gene + 4K drug descriptors [+ drug concentration]
Output dimensions: 1 (growth percentage)
* Sample size: millions of screening results (combinations of cell line and drug); filtered and balanced down to ~1M
* Notes on data balance: original data imbalanced with many drugs that have little inhibition effect

#### Expected Outcomes
* Regression. Predict percent growth for any NCI-60 cell line and drug combination 
* Dimension: 1 scalar value corresponding to the percent growth for a given drug concentration. Output range: [-100, 100]

#### Evaluation Metrics
* Accuracy or loss function: mean squared error or rank order.
* Expected performance of a naïve method: mean response, linear regression or random forest regression.

#### Description of the Network
* Proposed network architecture: MLP, MLP with convolution-like layers
* Number of layers: 5-7 layers

### Running the baseline implementation

```
$ cd Pilot1/P1B3
$ python p1b3_baseline_keras2.py
```

#### Example output
With the default parameters, running the benchmark takes about two days (training takes ~2 hours/epoch on K80). 
```
Using TensorFlow backend.
Args: Namespace(activation='relu', batch_normalization=False, batch_size=100, category_cutoffs=[0.0], cell_features=['expression'], conv=[0, 0, 0], dense=[1000, 500, 100, 50], drop=0.1, drug_features=['descriptors'], epochs=20, feature_subsample=0, locally_connected=False, logfile=None, loss='mse', max_logconc=-4.0, min_logconc=-5.0, optimizer='sgd', pool=10, save='save', scaling='std', scramble=False, subsample='naive_balancing', test_cell_split=0.15, test_steps=0, train_steps=0, val_split=0.2, val_steps=0, verbose=False, workers=1)
Loaded 2328562 unique (D, CL) response sets.
Distribution of dose response:
             GROWTH
count  1.014891e+06
mean  -1.296296e+00
std    6.216944e+01
min   -1.000000e+02
25%   -5.600000e+01
50%    0.000000e+00
75%    4.600000e+01
max    2.700000e+02
Category cutoffs: [0.0]
Dose response bin counts:
  Class 0:  502018 (0.4947) - between -1.00 and +0.00
  Class 1:  512873 (0.5053) - between +0.00 and +2.70
  Total:   1014891
Rows in train: 800121, val: 200030, test: 14740
Input features shapes:
  drug_concentration: (1,)
  cell_expression: (25722,)
  drug_descriptors: (3809,)
Total input dimensions: 29532
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 1000)              29533000
_________________________________________________________________
activation_1 (Activation)    (None, 1000)              0
_________________________________________________________________
dropout_1 (Dropout)          (None, 1000)              0
_________________________________________________________________
dense_2 (Dense)              (None, 500)               500500
_________________________________________________________________
activation_2 (Activation)    (None, 500)               0
_________________________________________________________________
dropout_2 (Dropout)          (None, 500)               0
_________________________________________________________________
dense_3 (Dense)              (None, 100)               50100
_________________________________________________________________
activation_3 (Activation)    (None, 100)               0
_________________________________________________________________
dropout_3 (Dropout)          (None, 100)               0
_________________________________________________________________
dense_4 (Dense)              (None, 50)                5050
_________________________________________________________________
activation_4 (Activation)    (None, 50)                0
_________________________________________________________________
dropout_4 (Dropout)          (None, 50)                0
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 51
=================================================================
Total params: 30,088,701.0
Trainable params: 30,088,701.0
Non-trainable params: 0.0

Epoch 1/20
800100/800100 [==============================] - 5491s - loss: 0.2913 - val_loss: 0.2385 - val_acc: 0.7196 - test_loss: 0.2641 - test_acc: 0.6903
Epoch 2/20
800100/800100 [==============================] - 5462s - loss: 0.2307 - val_loss: 0.2052 - val_acc: 0.7498 - test_loss: 0.2507 - test_acc: 0.7010
Epoch 3/20
800100/800100 [==============================] - 5415s - loss: 0.2035 - val_loss: 0.1859 - val_acc: 0.7668 - test_loss: 0.2492 - test_acc: 0.7059
Epoch 4/20
800100/800100 [==============================] - 5505s - loss: 0.1855 - val_loss: 0.1743 - val_acc: 0.7787 - test_loss: 0.2504 - test_acc: 0.7063
Epoch 5/20
800100/800100 [==============================] - 5426s - loss: 0.1724 - val_loss: 0.1621 - val_acc: 0.7897 - test_loss: 0.2502 - test_acc: 0.7040
Epoch 6/20
800100/800100 [==============================] - 5519s - loss: 0.1618 - val_loss: 0.1554 - val_acc: 0.7964 - test_loss: 0.2558 - test_acc: 0.6993
...

```

### Variations of the problem and command line examples
This benchmark can be run with additional or alternative molecular and drug feature sets. Various network architectural and training related hyperparameters can also be set at the command line. Here are some examples.

#### Use multiple cell line and drug feature sets
```
python p1b3_baseline_keras2.py --cell_features all --drug_features all --conv 10 10 1 5 5 1 -epochs 200
```
This will train a convolution network for 200 epochs, using three sets of cell line features (gene expression, microRNA, proteome) and two sets of drug features (Dragon7 descriptors, encoded latent representation from Aspuru-Guzik's SMILES autoencoder), and will bring the total input feature dimension to 40K.
```
Input features shapes:
  drug_concentration: (1,)
  cell_expression: (25722,)
  cell_microRNA: (453,)
  cell_proteome: (9760,)
  drug_descriptors: (3809,)
  drug_SMILES_latent: (292,)
Total input dimensions: 40037
```
The `--conv 10 10 1 5 5 1` parameter adds 2 convolution layers to the default 4-layer (1000-500-100-50) dense network. The first 3-tuple (10, 10, 1) denotes a convolution layer with 10 filters of kernel length 10 and stride 1; the second convolution layer has 5 filters with length 5 and stride 1. 

#### Run a toy version of the benchmark
```
python p1b3_baseline_keras2.py --feature_subsample 500 -e 5 --train_steps 100 --val_steps 10 --test_steps 10
```
This will take only minutes to run and can be used to test the environment setup. The `--feature_subsample 500` parameter instructs the benchmark to sample 500 random columns from each feature set. The steps parameters reduce the number of batches to use for each epoch.

#### Use locally-connected layers with batch normalization
```
python p1b3_baseline_keras2.py --conv 10 10 1 --pool 100 --locally_connected --optimizer adam --batch_normalization --batch_size 64
```
This example adds a locally-connected layer to the MLP and changes the optimizer and batch size. The locally connected layer is a convolution layer with unshared weights, so it tends to increase the number of parameters dramatically. Here we use a pooling size of 100 to reduce the parameters. This example also adds a batch normalization layer between any core layer and its activation. Batch normalization is known to speed up training in some settings. 


### Preliminary performance
Some of the best validation loss values we have seen are in the 0.04-0.06 range, which roughly corresponds to about 20-25% percent growth error per data point. We are running hyperparameter searches. 

During model training, a log file records the history of various metrics and the model with the best validation loss is saved in HDF5. Here are some examples of error distribution plots that are created whenever the model is improved. 

![Histogram of errors: Random vs Epoch 1](https://raw.githubusercontent.com/ECP-CANDLE/Benchmarks/master/Pilot1/P1B3/images/histo_It0.png)

![Histogram of errors after 141 epochs](https://raw.githubusercontent.com/ECP-CANDLE/Benchmarks/master/Pilot1/P1B3/images/histo_It140.png)

![Measure vs Predicted percent growth after 141 epochs](https://raw.githubusercontent.com/ECP-CANDLE/Benchmarks/master/Pilot1/P1B3/images/meas_vs_pred_It140.png)
