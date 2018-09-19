from __future__ import print_function
import sys
from pprint import pprint
import os

here = os.path.dirname(os.path.abspath(__file__))
top = os.path.dirname(os.path.dirname(os.path.dirname(here)))
sys.path.append(top)
BNAME = os.path.splitext(os.path.basename(__file__))[0]

from deephyper.benchmarks import util 

timer = util.Timer()
timer.start('module loading')

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os


from keras.callbacks import EarlyStopping
from deephyper.benchmarks.util import TerminateOnTimeOut


from keras import layers
from deephyper.benchmarks import keras_cmdline
from keras.models import load_model
import hashlib
import pickle
from deephyper.benchmarks.mnistmlp.load_data import load_data

from numpy.random import seed
from tensorflow import set_random_seed
timer.end()

seed(1)
set_random_seed(2)

def run(param_dict):
    param_dict = keras_cmdline.fill_missing_defaults(augment_parser, param_dict)
    optimizer = keras_cmdline.return_optimizer(param_dict)
    pprint(param_dict)
    
    timer.start('stage in')
    if param_dict['data_source']:
        data_source = param_dict['data_source']
    else:
        data_source = os.path.dirname(os.path.abspath(__file__))
        data_source = os.path.join(data_source, 'data')

    (x_train, y_train), (x_test, y_test) = load_data(
        origin=os.path.join(data_source, 'mnist.npz'),
        dest=param_dict['stage_in_destination']
    )
    
    timer.end()

    num_classes = 10

    BATCH_SIZE = param_dict['batch_size']
    EPOCHS = param_dict['epochs']
    DROPOUT = param_dict['dropout']
    ACTIVATION = param_dict['activation']
    NHIDDEN = param_dict['nhidden']
    NUNITS = param_dict['nunits']
    TIMEOUT = param_dict['timeout']

    timer.start('preprocessing')

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model_path = param_dict['model_path']
    model_mda_path = None
    model = None
    initial_epoch = 0

    if model_path:
        savedModel = util.resume_from_disk(BNAME, param_dict, data_dir=model_path)
        model_mda_path = savedModel.model_mda_path
        model_path = savedModel.model_path
        model = savedModel.model
        initial_epoch = savedModel.initial_epoch

    if model is None:
        model = Sequential()
        model.add(Dense(NUNITS, activation=ACTIVATION, input_shape=(784,)))
        model.add(Dropout(DROPOUT))
        for i in range(NHIDDEN):
            model.add(Dense(NUNITS, activation=ACTIVATION))
            model.add(Dropout(DROPOUT))
        model.add(Dense(num_classes, activation='softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

    timer.end()

    #earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=50, verbose=1, mode='auto')
    timeout_monitor = TerminateOnTimeOut(TIMEOUT)
    callbacks_list = [timeout_monitor]

    timer.start('model training')
    history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    initial_epoch=initial_epoch,
                    epochs=EPOCHS,
                    verbose=1,
                    callbacks=callbacks_list,
                    validation_split = 0.3)
                    #validation_data=(x_test, y_test))
    timer.end()
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    if model_path:
        timer.start('model save')
        model.save(model_path)  
        util.save_meta_data(param_dict, model_mda_path)
        timer.end()

    print('OUTPUT:', -score[1])
    return -score[1]


def augment_parser(parser):

    parser.add_argument('--nunits', action='store', dest='nunits',
                        nargs='?', const=2, type=int, default='512',
                        help='number of units/layer in MLP')

    parser.add_argument('--nhidden', action='store', dest='nhidden',
                        nargs='?', const=2, type=int, default='2',
                        help='number of hidden layers in MLP')

    return parser

if __name__ == "__main__":
    parser = keras_cmdline.create_parser()
    parser = augment_parser(parser)
    cmdline_args = parser.parse_args()
    param_dict = vars(cmdline_args)
    run(param_dict)
