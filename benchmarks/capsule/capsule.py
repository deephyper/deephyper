"""Train a simple CNN-Capsule Network on the CIFAR10 small images dataset.

Without Data Augmentation:
It gets to 75% validation accuracy in 10 epochs,
and 79% after 15 epochs, and overfitting after 20 epochs

With Data Augmentation:
It gets to 75% validation accuracy in 10 epochs,
and 79% after 15 epochs, and 83% after 30 epcohs.
In my test, highest validation accuracy is 83.79% after 50 epcohs.

This is a fast Implement, just 20s/epcoh with a gtx 1070 gpu.
"""

from __future__ import print_function
from keras import backend as K
from keras.engine.topology import Layer
from keras import activations
from keras import utils
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
import sys
import os
import time
from pprint import pprint
import hashlib

here = os.path.dirname(os.path.abspath(__file__))
top = os.path.dirname(os.path.dirname(os.path.dirname(here)))
sys.path.append(top)
from deephyper.benchmarks import keras_cmdline 


BNAME = 'capsule'

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


# the squashing function.
# we use 0.5 in stead of 1 in hinton's paper.
# if 1, the norm of vector will be zoomed out.
# if 0.5, the norm will be zoomed in while original norm is less than 0.5
# and be zoomed out while original norm is greater than 0.5.
def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x


# define our own softmax function instead of K.softmax
# because K.softmax can not specify axis.
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


# define the margin loss like hinge loss
def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)


class Capsule(Layer):
    """A Capsule Implement with Pure Keras
    There are two vesions of Capsule.
    One is like dense layer (for the fixed-shape input),
    and the other is like timedistributed dense (for various length input).

    The input shape of Capsule must be (batch_size,
                                        input_num_capsule,
                                        input_dim_capsule
                                       )
    and the output shape is (batch_size,
                             num_capsule,
                             dim_capsule
                            )

    Capsule Implement is from https://github.com/bojone/Capsule/
    Capsule Paper: https://arxiv.org/abs/1710.09829
    """

    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs):
        """Following the routing algorithm from Hinton's paper,
        but replace b = b + <u,v> with b = <u,v>.

        This change can improve the feature representation of Capsule.

        However, you can replace
            b = K.batch_dot(outputs, hat_inputs, [2, 3])
        with
            b += K.batch_dot(outputs, hat_inputs, [2, 3])
        to realize a standard routing.
        """

        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)

        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


def defaults():
    def_parser = keras_cmdline.create_parser()
    def_parser = augment_parser(def_parser)
    return vars(def_parser.parse_args(''))


def run(param_dict):
    default_params = defaults()
    for key in default_params:
        if key not in param_dict:
            param_dict[key] = default_params[key]
    pprint(param_dict)
    optimizer = keras_cmdline.return_optimizer(param_dict)
    print(param_dict)

    BATCH_SIZE = param_dict['batch_size']
    EPOCHS = param_dict['epochs']
    DATA_AUG = param_dict['data_aug']
    NUM_CONV = param_dict['num_conv']
    DIM_CAPS = param_dict['dim_capsule']
    ROUTINGS = param_dict['routings']
    SHARE_WEIGHTS = param_dict['share_weights']

    extension = extension_from_parameters(param_dict)
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
        if initial_epoch < param_dict['epochs']:
            resume = True
        else:
            initial_epoch = 0

    num_classes = 10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

    # A common Conv2D model
    input_image = Input(shape=(None, None, 3))
    x = input_image #Conv2D(64, (3, 3), activation='relu')(input_image)
    for i in range(NUM_CONV):
        x = Conv2D(64, (3, 3), activation='relu')(x)
    x = AveragePooling2D((2, 2))(x)
    for i in range(NUM_CONV):
        x = Conv2D(128, (3, 3), activation='relu')(x)

    """now we reshape it as (batch_size, input_num_capsule, input_dim_capsule)
    then connect a Capsule layer.

    the output of final model is the lengths of 10 Capsule, whose dim=16.

    the length of Capsule is the proba,
    so the problem becomes a 10 two-classification problem.
    """

    x = Reshape((-1, 128))(x)
    capsule = Capsule(10, DIM_CAPS, ROUTINGS, SHARE_WEIGHTS)(x)
    output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)
    model = Model(inputs=input_image, outputs=output)

    # we use a margin loss
    model.compile(loss=margin_loss, optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    # we can compare the performance with or without data augmentation
    data_augmentation = True

    if not DATA_AUG:
        print('Not using data augmentation.')
        history = model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(x_test, y_test),
            shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by dataset std
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in 0 to 180 degrees
            width_shift_range=0.1,  # randomly shift images horizontally
            height_shift_range=0.1,  # randomly shift images vertically
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        history = model.fit_generator(
            datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
            steps_per_epoch=EPOCHS,
            validation_data=(x_test, y_test),
            workers=4)
    
    model.save(model_name)  
    save_meta_data(param_dict, model_mda_name)

    val_acc = history.history['val_acc']
    print('===Validation accuracy:', val_acc[-1])
    print('OUTPUT:', -val_acc[-1])

def augment_parser(parser):

    parser.add_argument('--data_aug', action='store', dest='data_aug',
                        nargs='?', const=1, type=bool, default=False,
                        help='boolean. Whether to apply data augumentation?')


    parser.add_argument('--num_conv', action='store', dest='num_conv',
                        nargs='?', const=2, type=int, default='2',
                        help='number of convolution layers')

    parser.add_argument('--dim_capsule', action='store', dest='dim_capsule',
                        nargs='?', const=2, type=int, default='16',
                        help='dimension of capsule')

    parser.add_argument('--routings', action='store', dest='routings',
                        nargs='?', const=2, type=int, default='3',
                        help='dimension of capsule')

    parser.add_argument('--share_weights', action='store', dest='share_weights',
                        nargs='?', const=1, type=bool, default=True,
                        help='boolean. share weights?')


    return parser

if __name__ == "__main__":
    parser = keras_cmdline.create_parser()
    parser = augment_parser(parser)
    cmdline_args = parser.parse_args()
    param_dict = vars(cmdline_args)
    print(param_dict)
    run(param_dict)
