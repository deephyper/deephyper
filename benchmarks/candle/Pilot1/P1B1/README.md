## P1B1: Autoencoder Compressed Representation for Gene Expression

**Overview**: Given a sample of gene expression data, build a sparse autoencoder that can compress the expression profile into a low-dimensional vector.

**Relationship to core problem**: Many molecular assays generate large numbers of features that can lead to time-consuming processing and over-fitting in learning tasks; hence, a core capability we intend to build is feature reduction.

**Expected outcome**: Build an autoencoder that collapse high dimensional expression profiles into low dimensional vectors without much loss of information.

### Benchmark Specs Requirements 

#### Description of the Data
* Data source: RNA-seq data from GDC 
* Input dimensions: 60,484 floats; log(1+x) transformed FPKM-UQ values
* Output dimensions: Same as input
* Latent representation dimension: 1000
* Sample size: 4,000 (3000 training + 1000 test)
* Notes on data balance and other issues: unlabeled data draw from a diverse set of cancer types

#### Expected Outcomes
* Reconstructed expression profiles
* Output range: float; same as log transformed input

#### Evaluation Metrics
* Accuracy or loss function: mean squared error
* Expected performance of a naïve method: landmark genes picked by linear regression 

#### Description of the Network
* Proposed network architecture: MLP with encoding layers, dropout layers, bottleneck layer, and decoding layers
* Number of layers: At least three hidden layers including one encoding layer, one bottleneck layer, and one decoding layer

### Running the baseline implementation

```
cd Pilot1/P1B1
python p1b1_baseline_keras2.py
```
The training and test data files will be downloaded the first time this is run and will be cached for future runs. The baseline implementation supports three types of autoencoders controlled by the `--model` parameter: regular autoencoder (`ae`), variational autoencoder (`vae`), and conditional variational autoencoder (`cvae`).

#### Example output
```
Using Theano backend.
Using gpu device 0: Tesla K80 (CNMeM is enabled with initial size: 95.0% of memory, cuDNN 5004)
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_1 (InputLayer)             (None, 60483)         0
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 2000)          120968000   input_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 600)           1200600     dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 2000)          1202000     dense_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 60483)         121026483   dense_3[0][0]
====================================================================================================
Total params: 244397083
____________________________________________________________________________________________________
None
Train on 2400 samples, validate on 600 samples
Epoch 1/2
2400/2400 [==============================] - 8s - loss: 0.0420 - val_loss: 0.0383
Epoch 2/2
2400/2400 [==============================] - 8s - loss: 0.0376 - val_loss: 0.0377
```

### Preliminary performance

The current best performance in terms of validation correlation for the three types of autoencoders are as follows:

* AE: 0.96
* VAE: 0.86
* CVAE: 0.89

Here is an visual example of the 2D latent representation from VAE color coded by cancer types.

![VAE latent representation](https://raw.githubusercontent.com/ECP-CANDLE/Benchmarks/frameworks/Pilot1/P1B1/images/VAE-latent-2D.png)

