## Combo: Predicting Tumor Cell Line Response to Drug Pairs

**Overview**: Given combination drug screening results on NCI60 cell lines available at the NCI-ALMANAC database, build a deep learning network that can predict the growth percentage from the cell line molecular features and the descriptors of both drugs.

**Relationship to core problem**: This benchmark is an example one of the core capabilities needed for the Pilot 1 Drug Response problem: combining multiple molecular assays and drug descriptors in a single deep learning framework for response prediction.

**Expected outcome**: Build a DNN that can predict growth percentage of a cell line treated with a pair of drugs.

### Benchmark Specs Requirements

#### Description of the Data
* Data source: Combo drug response screening results from NCI-ALMANAC; 5-platform normalized expression, microRNA expression, and proteome abundance data from the NCI; Dragon7 generated drug descriptors based on 2D chemical structures from NCI
* Input dimensions: ~30K with default options: 26K normalized expression levels by gene + 4K drug descriptors; 59 cell lines; a subset of 54 FDA-approved drugs
Output dimensions: 1 (growth percentage)
* Sample size: 85,303 (cell line, drug 1, drug 2) tuples from the original 304,549 in the NCI-ALMANAC database
* Notes on data balance: there are more ineffective drug pairs than effective pairs; data imbalance is somewhat reduced by using only the best dose combination for each (cell line, drug 1, drug 2) tuple as training and validation data

#### Expected Outcomes
* Regression. Predict percent growth for any NCI-60 cell line and drugs combination
* Dimension: 1 scalar value corresponding to the percent growth for a given drug concentration. Output range: [-100, 100]

#### Evaluation Metrics
* Accuracy or loss function: mean squared error, mean absolute error, and R^2
* Expected performance of a naïve method: mean response, linear regression or random forest regression.

#### Description of the Network
* Proposed network architecture: two-stage neural network that is jointly trained for feature encoding and response prediction; shared submodel for each drug in the pair
* Number of layers: 3-4 layers for feature encoding submodels and response prediction submodels, respectively

### Running the baseline implementation

```
$ cd Pilot1/Combo
$ python combo_baseline_keras2.py
```

#### Example output
```
python combo_baseline_keras2.py --use_landmark_genes --warmup_lr --reduce_lr -z 256

Using TensorFlow backend.
Params: {'activation': 'relu', 'batch_size': 256, 'dense': [1000, 1000, 1000], 'dense_feature_layers': [1000, 1000, 1000], 'drop': 0, 'epochs': 10, 'learning_rate': None, 'loss': 'mse', 'optimizer': 'adam', 'residual': False, 'rng_seed': 2017, 'save': 'save/combo', 'scaling': 'std', 'feature_subsample': 0, 'validation_split': 0.2, 'solr_root': '', 'timeout': -1, 'cell_features': ['expression'], 'drug_features': ['descriptors'], 'cv': 1, 'max_val_loss': 1.0, 'base_lr': None, 'reduce_lr': True, 'warmup_lr': True, 'batch_normalization': False, 'gen': False, 'use_combo_score': False, 'config_file': '/home/fangfang/work/Benchmarks.combo/Pilot1/Combo/combo_default_model.txt', 'verbose': False, 'logfile': None, 'train_bool': True, 'shuffle': True, 'alpha_dropout': False, 'gpus': [], 'experiment_id': 'EXP.000', 'run_id': 'RUN.000', 'use_landmark_genes': True, 'cp': False, 'tb': False, 'datatype': <class 'numpy.float32'>}
Loaded 311737 unique (CL, D1, D2) response sets.
Filtered down to 85303 rows with matching information.
Unique cell lines: 59
Unique drugs: 54
Distribution of dose response:
             GROWTH
count  85303.000000
mean       0.281134
std        0.513399
min       -1.000000
25%        0.036594
50%        0.351372
75%        0.685700
max        1.693300
Rows in train: 68243, val: 17060
Input features shapes:
  cell.expression: (923,)
  drug1.descriptors: (3809,)
  drug2.descriptors: (3809,)
Total input dimensions: 8541
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 923)               0
_________________________________________________________________
dense_1 (Dense)              (None, 1000)              924000
_________________________________________________________________
dense_2 (Dense)              (None, 1000)              1001000
_________________________________________________________________
dense_3 (Dense)              (None, 1000)              1001000
=================================================================
Total params: 2,926,000

Trainable params: 2,926,000
Non-trainable params: 0
_________________________________________________________________
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_2 (InputLayer)         (None, 3809)              0
_________________________________________________________________
dense_4 (Dense)              (None, 1000)              3810000
_________________________________________________________________
dense_5 (Dense)              (None, 1000)              1001000
_________________________________________________________________
dense_6 (Dense)              (None, 1000)              1001000
=================================================================
Total params: 5,812,000
Trainable params: 5,812,000
Non-trainable params: 0
_________________________________________________________________
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input.cell.expression (InputLaye (None, 923)           0
____________________________________________________________________________________________________
input.drug1.descriptors (InputLa (None, 3809)          0
____________________________________________________________________________________________________
input.drug2.descriptors (InputLa (None, 3809)          0
____________________________________________________________________________________________________
cell.expression (Model)          (None, 1000)          2926000     input.cell.expression[0][0]
____________________________________________________________________________________________________
drug.descriptors (Model)         (None, 1000)          5812000     input.drug1.descriptors[0][0]
                                                                   input.drug2.descriptors[0][0]
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 3000)          0           cell.expression[1][0]
                                                                   drug.descriptors[1][0]
                                                                   drug.descriptors[2][0]
____________________________________________________________________________________________________
dense_7 (Dense)                  (None, 1000)          3001000     concatenate_1[0][0]
____________________________________________________________________________________________________
dense_8 (Dense)                  (None, 1000)          1001000     dense_7[0][0]
____________________________________________________________________________________________________
dense_9 (Dense)                  (None, 1000)          1001000     dense_8[0][0]
____________________________________________________________________________________________________
dense_10 (Dense)                 (None, 1)             1001        dense_9[0][0]
====================================================================================================
Total params: 13,742,001
Trainable params: 13,742,001
Non-trainable params: 0
____________________________________________________________________________________________________
Between random pairs in y_val:
  mse: 0.5290
  mae: 0.5737
  r2: -1.0006
  corr: -0.0003
Train on 68243 samples, validate on 17060 samples
Epoch 1/10
68243/68243 [==============================] - 5s - loss: 0.8597 - mae: 0.3433 - r2: -2.2767 - val_loss: 0.1018 - val_mae: 0.2360 - val_r2: 0.6123
Epoch 2/10
68243/68243 [==============================] - 4s - loss: 0.0905 - mae: 0.2227 - r2: 0.6535 - val_loss: 0.0929 - val_mae: 0.2274 - val_r2: 0.6456
Epoch 3/10
68243/68243 [==============================] - 4s - loss: 0.0749 - mae: 0.2004 - r2: 0.7129 - val_loss: 0.0669 - val_mae: 0.1895 - val_r2: 0.7451
Epoch 4/10
68243/68243 [==============================] - 4s - loss: 0.0656 - mae: 0.1869 - r2: 0.7490 - val_loss: 0.0712 - val_mae: 0.1928 - val_r2: 0.7283
Epoch 5/10
68243/68243 [==============================] - 4s - loss: 0.0580 - mae: 0.1749 - r2: 0.7776 - val_loss: 0.0535 - val_mae: 0.1658 - val_r2: 0.7959
Epoch 6/10
68243/68243 [==============================] - 4s - loss: 0.0523 - mae: 0.1651 - r2: 0.7996 - val_loss: 0.0492 - val_mae: 0.1596 - val_r2: 0.8122
Epoch 7/10
68243/68243 [==============================] - 5s - loss: 0.0457 - mae: 0.1535 - r2: 0.8247 - val_loss: 0.0476 - val_mae: 0.1571 - val_r2: 0.8184
Epoch 8/10
68243/68243 [==============================] - 4s - loss: 0.0403 - mae: 0.1433 - r2: 0.8454 - val_loss: 0.0453 - val_mae: 0.1512 - val_r2: 0.8271
Epoch 9/10
68243/68243 [==============================] - 4s - loss: 0.0364 - mae: 0.1365 - r2: 0.8602 - val_loss: 0.0387 - val_mae: 0.1382 - val_r2: 0.8519
Epoch 10/10
68243/68243 [==============================] - 4s - loss: 0.0332 - mae: 0.1303 - r2: 0.8727 - val_loss: 0.0382 - val_mae: 0.1339 - val_r2: 0.8541
Comparing y_true and y_pred:
  mse: 0.0382
  mae: 0.1339
  r2: 0.8557
  corr: 0.9254
```

#### Inference

There is a separate inference script that can be used to predict drug pair response on combinations of sample sets and drug sets with a trained model.
```
$ python infer.py --sample_set NCIPDM --drug_set ALMANAC

Using TensorFlow backend.
Predicting drug response for 6381440 combinations: 590 samples x 104 drugs x 104 drugs
100%|██████████████████████████████████████████████████████████████████████| 639/639 [14:56<00:00,  1.40s/it]
```
Example trained model files can be downloaded here: [saved.model.h5](http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/saved.model.h5) and [saved.weights.h5](http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/saved.weights.h5).

The inference script also accepts models trained with [dropout as a Bayesian Approximation](https://arxiv.org/pdf/1506.02142.pdf) for uncertainty quantification. Here is an example command line to make 100 point predictions for each sample-drugs combination in a subsample of the GDSC data.

```
$ python infer.py -s GDSC -d NCI_IOA_AOA --ns 10 --nd 5 -m saved.uq.model.h5 -w saved.uq.weights.h5 -n 100

$ cat comb_pred_GDSC_NCI_IOA_AOA.tsv
Sample  Drug1   Drug2   N       PredGrowthMean  PredGrowthStd   PredGrowthMin   PredGrowthMax
GDSC.22RV1      NSC.102816      NSC.102816      100     0.1688  0.0899  -0.0762 0.3912
GDSC.22RV1      NSC.102816      NSC.105014      100     0.3189  0.0920  0.0914  0.5550
GDSC.22RV1      NSC.102816      NSC.109724      100     0.6514  0.0894  0.4739  0.9055
GDSC.22RV1      NSC.102816      NSC.118218      100     0.5682  0.1164  0.2273  0.8891
GDSC.22RV1      NSC.102816      NSC.122758      100     0.3787  0.0833  0.1779  0.5768
GDSC.22RV1      NSC.105014      NSC.102816      100     0.1627  0.1060  -0.0531 0.5077
...
```

A version of trained model files with dropout are available here: [saved.uq.model.h5](http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/saved.uq.model.h5) and [saved.uq.weights.h5](http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/saved.uq.weights.h5).

