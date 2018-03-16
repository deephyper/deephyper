## P1B2: Sparse Classifier Disease Type Prediction from Somatic SNPs

**Overview**: Given patient somatic SNPs, build a deep learning network that can classify the cancer type.

**Relationship to core problem**: Exercise two core capabilities we need to build: (1) classification based on very sparse input data; (2) evaluation of the information content and predictive value in a molecular assay with auxiliary learning tasks.

**Expected outcome**: Build a DNN that can classify sparse data.

### Benchmark Specs Requirements 

#### Description of the Data
* Data source: SNP data from GDC MAF files
* Input dimensions: 28,205 (aggregated variation impact by gene from 2.7 million unique SNPs)
* Output dimensions: 10 class probabilities (9 most abundant cancer types in GDC + 1 “others”)
* Sample size: 4,000 (3000 training + 1000 test)
* Notes on data balance and other issues: data balance achieved via undersampling; “others” category drawn from all remaining lower-abundance cancer types in GDC

#### Expected Outcomes
* Classification
* Output range or number of classes: 10

#### Evaluation Metrics
* Accuracy or loss function: Standard approaches such as F1-score, accuracy, ROC-AUC, cross entropy, etc. 
* Expected performance of a naïve method: linear regression or ensemble methods without feature selection

#### Description of the Network
* Proposed network architecture: MLP with regularization
* Number of layers: ~5 layers

### Running the baseline implementation

```
cd Pilot1/P1B2
python p1b2_baseline_keras2.py
```
The training and test data files will be downloaded the first time this is run and will be cached for future runs.

#### Example output

```
Using Theano backend.
Using gpu device 0: Tesla K80 (CNMeM is enabled with initial size: 95.0% of memory, cuDNN 5004)
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
dense_1 (Dense)                  (None, 1024)          28881920    dense_input_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 512)           524800      dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 256)           131328      dense_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            2570        dense_3[0][0]
====================================================================================================
Total params: 29540618
____________________________________________________________________________________________________
None
Train on 2400 samples, validate on 600 samples
Epoch 1/20
2400/2400 [==============================] - 4s - loss: 2.2763 - acc: 0.1867 - val_loss: 2.0218 - val_acc: 0.1583
Epoch 2/20
2400/2400 [==============================] - 4s - loss: 1.7935 - acc: 0.4292 - val_loss: 1.6934 - val_acc: 0.3567
Epoch 3/20
2400/2400 [==============================] - 4s - loss: 1.2334 - acc: 0.6737 - val_loss: 1.6227 - val_acc: 0.4350
Epoch 4/20
2400/2400 [==============================] - 4s - loss: 0.8207 - acc: 0.8046 - val_loss: 1.5323 - val_acc: 0.4833
Epoch 5/20
2400/2400 [==============================] - 4s - loss: 0.5790 - acc: 0.9083 - val_loss: 1.4425 - val_acc: 0.5217
Epoch 6/20
2400/2400 [==============================] - 4s - loss: 0.4103 - acc: 0.9537 - val_loss: 1.3111 - val_acc: 0.5733
Epoch 7/20
2400/2400 [==============================] - 4s - loss: 0.3260 - acc: 0.9750 - val_loss: 1.4730 - val_acc: 0.5517
Epoch 8/20
2400/2400 [==============================] - 4s - loss: 0.2798 - acc: 0.9842 - val_loss: 1.5848 - val_acc: 0.5433
Epoch 9/20
2400/2400 [==============================] - 4s - loss: 0.2603 - acc: 0.9883 - val_loss: 1.7059 - val_acc: 0.5150
Epoch 10/20
2400/2400 [==============================] - 4s - loss: 0.2233 - acc: 0.9921 - val_loss: 1.8539 - val_acc: 0.4800
Epoch 11/20
2400/2400 [==============================] - 4s - loss: 0.2182 - acc: 0.9904 - val_loss: 2.0269 - val_acc: 0.5050
Epoch 12/20
2400/2400 [==============================] - 4s - loss: 0.2096 - acc: 0.9896 - val_loss: 1.5704 - val_acc: 0.5617
Epoch 13/20
2400/2400 [==============================] - 4s - loss: 0.1965 - acc: 0.9900 - val_loss: 1.6173 - val_acc: 0.5617
Epoch 14/20
2400/2400 [==============================] - 4s - loss: 0.1928 - acc: 0.9896 - val_loss: 1.5245 - val_acc: 0.5950
Epoch 15/20
2400/2400 [==============================] - 4s - loss: 0.1836 - acc: 0.9900 - val_loss: 1.6587 - val_acc: 0.5567
Epoch 16/20
2400/2400 [==============================] - 4s - loss: 0.1757 - acc: 0.9950 - val_loss: 1.5838 - val_acc: 0.5683
Epoch 17/20
2400/2400 [==============================] - 4s - loss: 0.1752 - acc: 0.9917 - val_loss: 1.6328 - val_acc: 0.5700
Epoch 18/20
2400/2400 [==============================] - 4s - loss: 0.1695 - acc: 0.9929 - val_loss: 1.6954 - val_acc: 0.5650
Epoch 19/20
2400/2400 [==============================] - 4s - loss: 0.1635 - acc: 0.9933 - val_loss: 1.6397 - val_acc: 0.5717
Epoch 20/20
2400/2400 [==============================] - 4s - loss: 0.1643 - acc: 0.9917 - val_loss: 1.7129 - val_acc: 0.5617

best_val_loss=1.31111 best_val_acc=0.59500

Best model saved to: model.A=sigmoid.B=64.D=None.E=20.L1=1024.L2=512.L3=256.P=1e-05.h5

Evaluation on test data: {'accuracy': 0.5500}
```

### Preliminary performance

The XGBoost classifier below achieves ~55% average accuracy on
validation data in the five-fold cross validation experiment. This
suggests there may be a low ceiling for the MLP results; there may not
be enough information in this set of SNP data to classify cancer types
accurately.

```
cd Pilot1/P1B2
python p1b2_xgboost.py
```

