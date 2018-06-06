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
from deephyper.benchmarks.cifar10cnn.load_data import load_data
from keras.preprocessing.image import ImageDataGenerator
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
        origin=os.path.join(data_source, 'cifar-10-python.tar.gz'),
        dest=param_dict['stage_in_destination'],
    )

    timer.end()

    num_classes = 10

    BATCH_SIZE = param_dict['batch_size']
    EPOCHS = param_dict['epochs']
    DROPOUT = param_dict['dropout']
    ACTIVATION = param_dict['activation']
    NUNITS = param_dict['nunits']
    F1_SIZE = param_dict['f1_size']
    F2_SIZE = param_dict['f2_size']
    F1_UNITS = param_dict['f1_units']
    F2_UNITS = param_dict['f2_units']
    P_SIZE = param_dict['p_size']
    DATA_AUGMENTATION = param_dict['data_aug']
    TIMEOUT = param_dict['timeout']


    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    timer.start('preprocessing')

    # Convert class vectors to binary class matrices.
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
        
        model.add(Conv2D(F1_UNITS, (F1_SIZE, F1_SIZE), padding='same',
                        input_shape=x_train.shape[1:]))
        model.add(Activation(ACTIVATION))
        model.add(Conv2D(F1_UNITS, (F1_SIZE, F1_SIZE)))
        model.add(Activation(ACTIVATION))
        model.add(MaxPooling2D(pool_size=(P_SIZE, P_SIZE), padding='same'))
        model.add(Dropout(DROPOUT))

        model.add(Conv2D(F2_UNITS, (F2_SIZE, F2_SIZE), padding='same'))
        model.add(Activation(ACTIVATION))
        model.add(Conv2D(F2_UNITS, (F2_SIZE, F2_SIZE)))
        model.add(Activation(ACTIVATION))
        model.add(MaxPooling2D(pool_size=(P_SIZE, P_SIZE), padding='same'))
        model.add(Dropout(DROPOUT))

        model.add(Flatten())
        model.add(Dense(NUNITS))
        model.add(Activation(ACTIVATION))
        model.add(Dropout(DROPOUT))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    timer.end()
    
    #earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=50, verbose=1, mode='auto')
    timeout_monitor = TerminateOnTimeOut(TIMEOUT)
    callbacks_list = [timeout_monitor]


    timer.start('model training')
    if not DATA_AUGMENTATION:
        history = model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        initial_epoch=initial_epoch,
                        verbose=1, shuffle=True,
                        callbacks=callbacks_list,
                        validation_split=0.30)
                        #validation_data=(x_test, y_test))
    else:
        datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
        steps_per_epoch = len(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE))
        datagen.fit(x_train)
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE), epochs=EPOCHS,
                            initial_epoch=initial_epoch,
                            callbacks=callbacks_list,
                            steps_per_epoch=steps_per_epoch, verbose=1,
                            validation_data=datagen.flow(x_test, y_test, batch_size=BATCH_SIZE), 
                            validation_steps=10,
                            workers=1)
                            #validation_split=0.30,
                            #validation_data=(x_test, y_test), 
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
    parser.add_argument('--data_aug', action='store', type=util.str2bool, default=False,
                        help='boolean. data_augmentation?')

    parser.add_argument('--f1_size', action='store', dest='f1_size',
                        nargs='?', const=2, type=int, default='3',
                        help='Filter 1 dim')

    parser.add_argument('--f2_size', action='store', dest='f2_size',
                        nargs='?', const=2, type=int, default='3',
                        help='Filter 2 dim')

    parser.add_argument('--f1_units', action='store', dest='f1_units',
                        nargs='?', const=2, type=int, default='32',
                        help='Filter 1 units')

    parser.add_argument('--f2_units', action='store', dest='f2_units',
                        nargs='?', const=2, type=int, default='64',
                        help='Filter 2 units')

    parser.add_argument('--p_size', action='store', dest='p_size',
                        nargs='?', const=2, type=int, default='2',
                        help='pool size')

    parser.add_argument('--nunits', action='store', dest='nunits',
                        nargs='?', const=2, type=int, default='512',
                        help='number of units in FC layer')
    parser.add_argument('--dropout2', type=float, default=0.5, 
                        help='dropout after FC layer')

    return parser

if __name__ == "__main__":
    parser = keras_cmdline.create_parser()
    parser = augment_parser(parser)
    cmdline_args = parser.parse_args()
    param_dict = vars(cmdline_args)
    run(param_dict)
