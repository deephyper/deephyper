from __future__ import print_function

import argparse
import h5py
import logging
import os

import numpy as np

import keras
from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.layers import BatchNormalization, Dense, Dropout, Input, Lambda
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TensorBoard
from keras.metrics import binary_crossentropy, mean_squared_error
from scipy.stats.stats import pearsonr
from sklearn.manifold import TSNE

from keras.models import load_model
import hashlib
import pickle
import sys

BNAME = 'p1b1'

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.metrics import r2_score
    from sklearn.metrics import accuracy_score

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import p1b1
import p1_common
import p1_common_keras
from solr_keras import CandleRemoteMonitor, compute_trainable_params, TerminateOnTimeOut


np.set_printoptions(precision=4)


BNAME = 'p1b1'

def extension_from_parameters(param_dict):
    extension = ''
    for key in sorted(param_dict):
        if key != 'epochs':
            print ('%s: %s' % (key, param_dict[key]))
            extension += '.{}={}'.format(key,param_dict[key])
    print(extension)
    return extension

def save_meta_data(data, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_meta_data(filename):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data



def get_p1b1_parser():
    parser = argparse.ArgumentParser(prog='p1b1_baseline', formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Train Autoencoder - Pilot 1 Benchmark 1')
    return p1b1.common_parser(parser)


def covariance(x, y):
    return K.mean(x * y) - K.mean(x) * K.mean(y)


def corr(y_true, y_pred):
    cov = covariance(y_true, y_pred)
    var1 = covariance(y_true, y_true)
    var2 = covariance(y_pred, y_pred)
    return cov / (K.sqrt(var1 * var2) + K.epsilon())


def xent(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)


def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


class MetricHistory(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print("\n")

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.validation_data[0])
        r2 = r2_score(self.validation_data[1], y_pred)
        corr, _ = pearsonr(self.validation_data[1].flatten(), y_pred.flatten())
        print("\nval_r2:", r2)
        print(y_pred.shape)
        print("\nval_corr:", corr, "val_r2:", r2)
        print("\n")


class LoggingCallback(Callback):
    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
        msg = "[Epoch: %i] %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in sorted(logs.items())))
        self.print_fcn(msg)


def plot_history(out, history, metric='loss', title=None):
    title = title or 'model {}'.format(metric)
    val_metric = 'val_{}'.format(metric)
    plt.figure(figsize=(16, 9))
    plt.plot(history.history[metric])
    plt.plot(history.history[val_metric])
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    png = '{}.plot.{}.png'.format(out, metric)
    plt.savefig(png, bbox_inches='tight')


def plot_scatter(data, classes, out):
    cmap = plt.cm.get_cmap('gist_rainbow')
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1], c=classes, cmap=cmap, lw=0.5, edgecolor='black', alpha=0.7)
    plt.colorbar()
    png = '{}.png'.format(out)
    plt.savefig(png, bbox_inches='tight')


def build_type_classifier(x_train, y_train, x_test, y_test):
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)
    from xgboost import XGBClassifier
    clf = XGBClassifier(max_depth=6, n_estimators=100)
    clf.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], verbose=False)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(acc)
    return clf


def initialize_parameters():
    # Get command-line parameters
    parser = get_p1b1_parser()
    args = parser.parse_args()
    # Get parameters from configuration file
    file_params = p1b1.read_config_file(args.config_file)
    # Consolidate parameter set. Command-line parameters overwrite file configuration
    params = p1_common.args_overwrite_config(args, file_params)
    # print(params)
    return params


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)

    import random
    random.seed(seed)

    if K.backend() == 'tensorflow':
        import tensorflow as tf
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        tf.set_random_seed(seed)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)


def verify_path(path):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)


def set_up_logger(logfile, verbose):
    verify_path(logfile)
    fh = logging.FileHandler(logfile)
    fh.setFormatter(logging.Formatter("[%(asctime)s %(process)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    fh.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(''))
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)

    p1b1.logger.setLevel(logging.DEBUG)
    p1b1.logger.addHandler(fh)
    p1b1.logger.addHandler(sh)
    return p1b1.logger


def save_cache(cache_file, x_train, y_train, x_val, y_val, x_test, y_test, x_labels, y_labels):
    with h5py.File(cache_file, 'w') as hf:
        hf.create_dataset("x_train",  data=x_train)
        hf.create_dataset("y_train",  data=y_train)
        hf.create_dataset("x_val", data=x_val)
        hf.create_dataset("y_val", data=y_val)
        hf.create_dataset("x_test", data=x_test)
        hf.create_dataset("y_test", data=y_test)
        hf.create_dataset("x_labels", (len(x_labels), 1), 'S100', data=[x.encode("ascii", "ignore") for x in x_labels])
        hf.create_dataset("y_labels", (len(y_labels), 1), 'S100', data=[x.encode("ascii", "ignore") for x in y_labels])


def load_cache(cache_file):
    with h5py.File(cache_file, 'r') as hf:
        x_train = hf['x_train'][:]
        y_train = hf['y_train'][:]
        x_val = hf['x_val'][:]
        y_val = hf['y_val'][:]
        x_test = hf['x_test'][:]
        y_test = hf['y_test'][:]
        x_labels = [x[0].decode('unicode_escape') for x in hf['x_labels'][:]]
        y_labels = [x[0].decode('unicode_escape') for x in hf['y_labels'][:]]
    return x_train, y_train, x_val, y_val, x_test, y_test, x_labels, y_labels


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def run(params):
    args = Struct(**params)
    # Construct extension to save model
    ext = p1b1.extension_from_parameters(params, '.keras')
    prefix = '{}{}'.format(params['save'], ext)
    
    logfile = params['logfile'] if params['logfile'] else prefix+'.log'
    verify_path(logfile)
    logger = set_up_logger(logfile, params['verbose'])

    logger.info('Params: {}'.format(params))

    # Get default parameters for initialization and optimizer functions
    keras_defaults = p1_common.keras_default_config()
    seed = params['rng_seed']
    set_seed(seed)

    # Load dataset
    x_train, y_train, x_val, y_val, x_test, y_test, x_labels, y_labels = p1b1.load_data(params, seed)

    # cache_file = 'data_l1000_cache.h5'
    # save_cache(cache_file, x_train, y_train, x_val, y_val, x_test, y_test, x_labels, y_labels)
    # x_train, y_train, x_val, y_val, x_test, y_test, x_labels, y_labels = load_cache(cache_file)

    logger.info("Shape x_train: {}".format(x_train.shape))
    logger.info("Shape x_val:   {}".format(x_val.shape))
    logger.info("Shape x_test:  {}".format(x_test.shape))

    logger.info("Range x_train: [{:.3g}, {:.3g}]".format(np.min(x_train), np.max(x_train)))
    logger.info("Range x_val:   [{:.3g}, {:.3g}]".format(np.min(x_val), np.max(x_val)))
    logger.info("Range x_test:  [{:.3g}, {:.3g}]".format(np.min(x_test), np.max(x_test)))

    logger.debug('Class labels')
    for i, label in enumerate(y_labels):
        logger.debug('  {}: {}'.format(i, label))

    extension = extension_from_parameters(args.__dict__)
    hex_name = hashlib.sha224(extension.encode('utf-8')).hexdigest()
    model_name = '{}-{}.h5'.format(BNAME, hex_name)
    model_mda_name = '{}-{}.pkl'.format(BNAME, hex_name)
    initial_epoch = 0
    resume = False
    if os.path.exists(model_name) and os.path.exists(model_mda_name):
        print('model and meta data exists; loading model from h5 file')
        model = load_model(model_name)
        saved_param_dict = load_meta_data(model_mda_name)
        initial_epoch = saved_param_dict['epochs']
        if initial_epoch < args.__dict__['epochs']:
            resume = True
    # clf = build_type_classifier(x_train, y_train, x_val, y_val)
    if resume:
        n_classes = len(y_labels)
        cond_train = y_train
        cond_val = y_val
        cond_test = y_test

        input_dim = x_train.shape[1]
        cond_dim = cond_train.shape[1]
        latent_dim = params['latent_dim']

        activation = params['activation']
        dropout = params['drop']
        dense_layers = params['dense']
        dropout_layer = keras.layers.noise.AlphaDropout if params['alpha_dropout'] else Dropout

        # Initialize weights and learning rule
        initializer_weights = p1_common_keras.build_initializer(params['initialization'], keras_defaults, seed)
        initializer_bias = p1_common_keras.build_initializer('constant', keras_defaults, 0.)

        if dense_layers is not None:
            if type(dense_layers) != list:
                dense_layers = list(dense_layers)
        else:
            dense_layers = []

        # Encoder Part
        x_input = Input(shape=(input_dim,))
        cond_input = Input(shape=(cond_dim,))
        h = x_input
        if params['model'] == 'cvae':
            h = keras.layers.concatenate([x_input, cond_input])

        for i, layer in enumerate(dense_layers):
            if layer > 0:
                x = h
                h = Dense(layer, activation=activation,
                        kernel_initializer=initializer_weights,
                        bias_initializer=initializer_bias)(h)
                if params['residual']:
                    try:
                        h = keras.layers.add([h, x])
                    except ValueError:
                        pass
                if params['batch_normalization']:
                    h = BatchNormalization()(h)
                if dropout > 0:
                    h = dropout_layer(dropout)(h)

        if params['model'] == 'ae':
            encoded = Dense(latent_dim, activation=activation,
                            kernel_initializer=initializer_weights,
                            bias_initializer=initializer_bias)(h)
        else:
            epsilon_std = params['epsilon_std']
            z_mean = Dense(latent_dim, name='z_mean')(h)
            z_log_var = Dense(latent_dim, name='z_log_var')(h)
            encoded = z_mean

            def vae_loss(x, x_decoded_mean):
                xent_loss = binary_crossentropy(x, x_decoded_mean)
                kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
                return K.mean(xent_loss + kl_loss/input_dim)

            def sampling(params):
                z_mean_, z_log_var_ = params
                batch_size = K.shape(z_mean_)[0]
                epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                        mean=0., stddev=epsilon_std)
                return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

            z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
            if params['model'] == 'cvae':
                z_cond = keras.layers.concatenate([z, cond_input])

        # Decoder Part
        decoder_input = Input(shape=(latent_dim,))
        h = decoder_input
        if params['model'] == 'cvae':
            h = keras.layers.concatenate([decoder_input, cond_input])

        for i, layer in reversed(list(enumerate(dense_layers))):
            if layer > 0:
                x = h
                h = Dense(layer, activation=activation,
                        kernel_initializer=initializer_weights,
                        bias_initializer=initializer_bias)(h)
                if params['residual']:
                    try:
                        h = keras.layers.add([h, x])
                    except ValueError:
                        pass
                if params['batch_normalization']:
                    h = BatchNormalization()(h)
                if dropout > 0:
                    h = dropout_layer(dropout)(h)

        decoded = Dense(input_dim, activation='sigmoid',
                        kernel_initializer=initializer_weights,
                        bias_initializer=initializer_bias)(h)

        # Build autoencoder model
        if params['model'] == 'cvae':
            encoder = Model([x_input, cond_input], encoded)
            decoder = Model([decoder_input, cond_input], decoded)
            model = Model([x_input, cond_input], decoder([z, cond_input]))
            loss = vae_loss
            metrics = [xent, corr, mse]
        elif params['model'] == 'vae':
            encoder = Model(x_input, encoded)
            decoder = Model(decoder_input, decoded)
            model = Model(x_input, decoder(z))
            loss = vae_loss
            metrics = [xent, corr, mse]
        else:
            encoder = Model(x_input, encoded)
            decoder = Model(decoder_input, decoded)
            model = Model(x_input, decoder(encoded))
            loss = params['loss']
            metrics = [xent, corr]

        model.summary()
        decoder.summary()

        if params['cp']:
            model_json = model.to_json()
            with open(prefix+'.model.json', 'w') as f:
                print(model_json, file=f)

        # Define optimizer
        # optimizer = p1_common_keras.build_optimizer(params['optimizer'],
        #                                             params['learning_rate'],
        #                                             keras_defaults)
        optimizer = optimizers.deserialize({'class_name': params['optimizer'], 'config': {}})
        base_lr = params['base_lr'] or K.get_value(optimizer.lr)
        if params['learning_rate']:
            K.set_value(optimizer.lr, params['learning_rate'])

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        # calculate trainable and non-trainable params
        params.update(compute_trainable_params(model))

        def warmup_scheduler(epoch):
            lr = params['learning_rate'] or base_lr * params['batch_size']/100
            if epoch <= 5:
                K.set_value(model.optimizer.lr, (base_lr * (5-epoch) + lr * epoch) / 5)
            logger.debug('Epoch {}: lr={}'.format(epoch, K.get_value(model.optimizer.lr)))
            return K.get_value(model.optimizer.lr)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        warmup_lr = LearningRateScheduler(warmup_scheduler)
        #checkpointer = ModelCheckpoint(params['save']+ext+'.weights.h5', save_best_only=True, save_weights_only=True)
        #tensorboard = TensorBoard(log_dir="tb/tb{}".format(ext))
        #candle_monitor = CandleRemoteMonitor(params=params)
        timeout_monitor = TerminateOnTimeOut(params['timeout'])
        #history_logger = LoggingCallback(logger.debug)

        callbacks = [timeout_monitor]
        if params['reduce_lr']:
            callbacks.append(reduce_lr)
        if params['warmup_lr']:
            callbacks.append(warmup_lr)
        if params['cp']:
            callbacks.append(checkpointer)
        if params['tb']:
            callbacks.append(tensorboard)

        x_val2 = np.copy(x_val)
        np.random.shuffle(x_val2)
        start_scores = p1b1.evaluate_autoencoder(x_val, x_val2)
        logger.info('\nBetween random pairs of validation samples: {}'.format(start_scores))

        if params['model'] == 'cvae':
            inputs = [x_train, cond_train]
            val_inputs = [x_val, cond_val]
            test_inputs = [x_test, cond_test]
        else:
            inputs = x_train
            val_inputs = x_val
            test_inputs = x_test

        outputs = x_train
        val_outputs = x_val
        test_outputs = x_test

    history = model.fit(inputs, outputs,
                        verbose=2,
                        batch_size=params['batch_size'],
                        initial_epoch=initial_epoch, epochs=params['epochs'],
                        callbacks=callbacks,
                        validation_data=(val_inputs, val_outputs))

    model.save(model_name)                                                                                                                                         
    save_meta_data(args.__dict__, model_mda_name)
    
    


    if params['cp']:
        encoder.save(prefix+'.encoder.h5')
        decoder.save(prefix+'.decoder.h5')

    #plot_history(prefix, history, 'loss')
    #plot_history(prefix, history, 'corr', 'streaming pearson correlation')

    # Evalute model on test set
    x_pred = model.predict(test_inputs)
    scores = p1b1.evaluate_autoencoder(x_pred, x_test)
    logger.info('\nEvaluation on test data: {}'.format(scores))
    print(scores)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    print('OUTPUT:', -score[1])
 
    if params['tsne']:
        tsne = TSNE(n_components=2, random_state=seed)
        x_test_encoded_tsne = tsne.fit_transform(x_test_encoded)
        plot_scatter(x_test_encoded_tsne, y_test_classes, prefix+'.latent.tsne')

    logger.handlers = []

    return history


def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()
    if K.backend() == 'tensorflow':
        K.clear_session()
