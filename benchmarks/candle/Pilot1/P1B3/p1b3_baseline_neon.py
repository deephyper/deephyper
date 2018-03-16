from __future__ import division, print_function

import os
import sys
import argparse
import logging

import numpy as np

import neon
from neon.util.argparser import NeonArgparser
from neon.callbacks.callbacks import Callbacks
from neon.layers import GeneralizedCost, Affine, Conv, Dropout, Pooling, Reshape
from neon.models import Model
from neon.backends import gen_backend
from neon.transforms import Identity
# from neon.transforms import MeanSquared
#from neon import transforms


import p1b3
import p1_common
import p1_common_neon


def get_p1b3_parser():
    
    # Construct neon arg parser. It generates a large set of options by default
    parser = NeonArgparser(__doc__)
    # Specify the default config_file
    parser.add_argument("--config_file", dest='config_file', type=str,
                        default=os.path.join(p1b3.file_path, 'p1b3_default_model.txt'),
                        help="specify model configuration file")
    
    # Parse other options that are not included on neon arg parser
    parser = p1_common.get_p1_common_parser(parser)
    
    # Arguments that are applicable just to p1b3
    parser = p1b3.p1b3_parser(parser)
    
    return parser


#np.set_printoptions(threshold=np.nan)
#np.random.seed(SEED)


class ConcatDataIter(neon.NervanaObject):
    """
    Data iterator for concatenated features
    Modeled after ArrayIterator: https://github.com/NervanaSystems/neon/blob/master/neon/data/dataiterator.py
    """

    def __init__(self, data_loader,
                 partition='train',
                 ndata=None,
                 lshape=None,
                 datatype=np.float32):
        """
        During initialization, the input data will be converted to backend tensor objects
        (e.g. CPUTensor or GPUTensor). If the backend uses the GPU, the data is copied over to the
        device.
        """
        super(ConcatDataIter, self).__init__()
        self.data = data_loader
        self.gen = p1b3.DataGenerator(data_loader, partition=partition, batch_size=self.be.bsz, concat=True)
        self.ndata = ndata or self.gen.num_data
        assert self.ndata >= self.be.bsz
        self.datatype = datatype
        self.gen = self.gen.flow()
        self.start = 0
        self.ybuf = None
        self.shape = lshape or data_loader.input_dim
        self.lshape = lshape

    @property
    def nbatches(self):
        """
        Return the number of minibatches in this dataset.
        """
        return (self.ndata - self.start) // self.be.bsz

    def reset(self):
        self.start = 0

    def __iter__(self):
        """
        Returns a new minibatch of data with each call.

        Yields:
            tuple: The next minibatch which includes both features and labels.
        """

        def transpose_gen(z):
            return (self.be.array(z), self.be.iobuf(z.shape[1]),
                    lambda _in, _out: self.be.copy_transpose(_in, _out))

        for i1 in range(self.start, self.ndata, self.be.bsz):
            bsz = min(self.be.bsz, self.ndata - i1)
            # islice1, oslice1 = slice(0, bsz), slice(i1, i1 + bsz)
            islice1, oslice1 = slice(0, bsz), slice(0, bsz)
            islice2, oslice2 = None, None
            if self.be.bsz > bsz:
                islice2, oslice2 = slice(bsz, None), slice(0, self.be.bsz - bsz)
                self.start = self.be.bsz - bsz

            x, y = next(self.gen)
            x = np.ascontiguousarray(x).astype(self.datatype)
            y = np.ascontiguousarray(y).astype(self.datatype)

            X = [x]
            y = y.reshape(y.shape + (1,))

            self.Xdev, self.Xbuf, self.unpack_func = list(zip(*[transpose_gen(x) for x in X]))
            self.dbuf, self.hbuf = list(self.Xdev), list(self.Xbuf)
            self.unpack_func = list(self.unpack_func)

            self.ydev, self.ybuf, yfunc = transpose_gen(y)
            self.dbuf.append(self.ydev)
            self.hbuf.append(self.ybuf)
            self.unpack_func.append(yfunc)

            for buf, dev, unpack_func in zip(self.hbuf, self.dbuf, self.unpack_func):
                unpack_func(dev[oslice1], buf[:, islice1])
                if oslice2:
                    unpack_func(dev[oslice2], buf[:, islice2])

            inputs = self.Xbuf[0] if len(self.Xbuf) == 1 else self.Xbuf
            targets = self.ybuf if self.ybuf else inputs

            yield (inputs, targets)


def main():
    # Get command-line parameters
    parser = get_p1b3_parser()
    args = parser.parse_args()
    #print('Args:', args)
    # Get parameters from configuration file
    fileParameters = p1b3.read_config_file(args.config_file)
    #print ('Params:', fileParameters)
    
    # Correct for arguments set by default by neon parser
    # (i.e. instead of taking the neon parser default value fall back to the config file,
    # if effectively the command-line was used, then use the command-line value)
    # This applies to conflictive parameters: batch_size, epochs and rng_seed
    if not any("--batch_size" in ag or "-z" in ag for ag in sys.argv):
        args.batch_size = fileParameters['batch_size']
    if not any("--epochs" in ag or "-e" in ag for ag in sys.argv):
        args.epochs = fileParameters['epochs']
    if not any("--rng_seed" in ag or "-r" in ag for ag in sys.argv):
        args.rng_seed = fileParameters['rng_seed']

    # Consolidate parameter set. Command-line parameters overwrite file configuration
    gParameters = p1_common.args_overwrite_config(args, fileParameters)
    print ('Params:', gParameters)

    # Determine verbosity level
    loggingLevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loggingLevel, format='')
    # Construct extension to save model
    ext = p1b3.extension_from_parameters(gParameters, '.neon')

    # Get default parameters for initialization and optimizer functions
    kerasDefaults = p1_common.keras_default_config()
    seed = gParameters['rng_seed']

    # Build dataset loader object
    loader = p1b3.DataLoader(seed=seed, dtype=gParameters['datatype'],
                             val_split=gParameters['validation_split'],
                             test_cell_split=gParameters['test_cell_split'],
                             cell_features=gParameters['cell_features'],
                             drug_features=gParameters['drug_features'],
                             feature_subsample=gParameters['feature_subsample'],
                             scaling=gParameters['scaling'],
                             scramble=gParameters['scramble'],
                             min_logconc=gParameters['min_logconc'],
                             max_logconc=gParameters['max_logconc'],
                             subsample=gParameters['subsample'],
                             category_cutoffs=gParameters['category_cutoffs'])


    # Re-generate the backend after consolidating parsing and file config
    gen_backend(backend=args.backend,
                rng_seed=seed,
                device_id=args.device_id,
                batch_size=gParameters['batch_size'],
                datatype=gParameters['datatype'],
                max_devices=args.max_devices,
                compat_mode=args.compat_mode)

    # Initialize weights and learning rule
    initializer_weights = p1_common_neon.build_initializer(gParameters['initialization'], kerasDefaults, seed)
    initializer_bias = p1_common_neon.build_initializer('constant', kerasDefaults, 0.)

    activation = p1_common_neon.get_function(gParameters['activation'])()

    # Define model architecture
    layers = []
    reshape = None

    if 'dense' in gParameters: # Build dense layers
        for layer in gParameters['dense']:
            if layer:
                layers.append(Affine(nout=layer, init=initializer_weights,
                        bias=initializer_bias, activation=activation))
            if gParameters['drop']:
                layers.append(Dropout(keep=(1-gParameters['drop'])))
    else: # Build convolutional layers
        reshape = (1, loader.input_dim, 1)
        layer_list = list(range(0, len(gParameters['conv']), 3))
        for l, i in enumerate(layer_list):
            nb_filter = gParameters['conv'][i]
            filter_len = gParameters['conv'][i+1]
            stride = gParameters['conv'][i+2]
            # print(nb_filter, filter_len, stride)
            # fshape: (height, width, num_filters).
            layers.append(Conv((1, filter_len, nb_filter), strides={'str_h':1, 'str_w':stride},
                init=initializer_weights, activation=activation))
            if gParameters['pool']:
                layers.append(Pooling((1, gParameters['pool'])))

    layers.append(Affine(nout=1, init=initializer_weights,
                        bias=initializer_bias, activation=neon.transforms.Identity()))

    # Build model
    model = Model(layers=layers)

    # Define neon data iterators
    train_samples = int(loader.n_train)
    val_samples = int(loader.n_val)

    if 'train_samples' in gParameters:
        train_samples = gParameters['train_samples']
    if 'val_samples' in gParameters:
        val_samples = gParameters['val_samples']

    train_iter = ConcatDataIter(loader, ndata=train_samples,
            lshape=reshape, datatype=gParameters['datatype'])
    val_iter = ConcatDataIter(loader, partition='val', ndata=val_samples,
            lshape=reshape, datatype=gParameters['datatype'])

    # Define cost and optimizer
    cost = GeneralizedCost(p1_common_neon.get_function(gParameters['loss'])())
    optimizer = p1_common_neon.build_optimizer(gParameters['optimizer'],
                                            gParameters['learning_rate'],
                                            kerasDefaults)

    callbacks = Callbacks(model, eval_set=val_iter, eval_freq = 1)#**args.callback_args)

    model.fit(train_iter, optimizer=optimizer, num_epochs=gParameters['epochs'], cost=cost, callbacks=callbacks)


if __name__ == '__main__':
    main()
