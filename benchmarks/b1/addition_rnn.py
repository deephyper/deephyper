# -*- coding: utf-8 -*-
'''An implementation of sequence to sequence learning for performing addition
Input: "535+61"
Output: "596"
Padding is handled by using a repeated sentinel character (space)

Input may optionally be inverted, shown to increase performance in many tasks in:
"Learning to Execute"
http://arxiv.org/abs/1410.4615
and
"Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target.

Two digits inverted:
+ One layer LSTM (128 HN), 5k training examples = 99% train/test accuracy in 55 epochs

Three digits inverted:
+ One layer LSTM (128 HN), 50k training examples = 99% train/test accuracy in 100 epochs

Four digits inverted:
+ One layer LSTM (128 HN), 400k training examples = 99% train/test accuracy in 20 epochs

Five digits inverted:
+ One layer LSTM (128 HN), 550k training examples = 99% train/test accuracy in 30 epochs
'''

import sys
import os
import time
from pprint import pprint

here = os.path.dirname(os.path.abspath(__file__))
top = os.path.dirname(os.path.dirname(os.path.dirname(here)))
sys.path.append(top)

start = time.time()
from keras.models import Sequential
from keras import layers
import numpy as np
from six.moves import range
from deephyper.benchmarks import keras_cmdline 

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
from keras.models import load_model
import hashlib
import pickle


set_random_seed(2)
load_time = time.time() - start
print(f"module import time: {load_time:.3f} seconds")
    
TRAINING_SIZE = 500
DIGITS = 3
INVERT = True
MAXLEN = DIGITS + 1 + DIGITS
BNAME = 'addition_rnn'

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

class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        """Initialize character table.

        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One hot encode given string C.

        # Arguments
            num_rows: Number of rows in the returned one hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)

def generate_data():
    # Parameters for the model and dataset.

    # Maximum length of input is 'int + int' (e.g., '345+678'). Maximum length of
    # int is DIGITS.

    # All the numbers, plus sign and space for padding.
    chars = '0123456789+ '
    ctable = CharacterTable(chars)

    questions = []
    expected = []
    seen = set()
    print('Generating data...')
    while len(questions) < TRAINING_SIZE:
        f = lambda: int(''.join(np.random.choice(list('0123456789'))
                        for i in range(np.random.randint(1, DIGITS + 1))))
        a, b = f(), f()
        # Skip any addition questions we've already seen
        # Also skip any such that x+Y == Y+x (hence the sorting).
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        # Pad the data with spaces such that it is always MAXLEN.
        q = '{}+{}'.format(a, b)
        query = q + ' ' * (MAXLEN - len(q))
        ans = str(a + b)
        # Answers can be of maximum size DIGITS + 1.
        ans += ' ' * (DIGITS + 1 - len(ans))
        if INVERT:
            # Reverse the query, e.g., '12+345  ' becomes '  543+21'. (Note the
            # space used for padding.)
            query = query[::-1]
        questions.append(query)
        expected.append(ans)
    print('Total addition questions:', len(questions))

    print('Vectorization...')
    x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
    y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(questions):
        x[i] = ctable.encode(sentence, MAXLEN)
    for i, sentence in enumerate(expected):
        y[i] = ctable.encode(sentence, DIGITS + 1)

    # Shuffle (x, y) in unison as the later parts of x will almost all be larger
    # digits.
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    # Explicitly set apart 10% for validation data that we never train over.
    split_at = len(x) - len(x) // 10
    (x_train, x_val) = x[:split_at], x[split_at:]
    (y_train, y_val) = y[:split_at], y[split_at:]

    print('Training Data:')
    print(x_train.shape)
    print(y_train.shape)

    print('Validation Data:')
    print(x_val.shape)
    print(y_val.shape)
    return x_train, y_train, x_val, y_val, chars

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

    x_train, y_train, x_val, y_val, chars = generate_data()

    if param_dict['rnn_type'] == 'GRU':
        RNN = layers.GRU
    elif param_dict['rnn_type'] == 'SimpleRNN':
        RNN = layers.SimpleRNN
    else:
        RNN = layers.LSTM

    HIDDEN_SIZE = param_dict['hidden_size']
    BATCH_SIZE = param_dict['batch_size']
    LAYERS = param_dict['layers']
    activation = param_dict['activation']

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
    
    if not resume:
        print('Build model...')
        model = Sequential()
        # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
        # Note: In a situation where your input sequences have a variable length,
        # use input_shape=(None, num_feature).
        model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
        # As the decoder RNN's input, repeatedly provide with the last hidden state of
        # RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
        # length of output, e.g., when DIGITS=3, max output is 999+999=1998.
        model.add(layers.RepeatVector(DIGITS + 1))
        # The decoder RNN could be multiple layers stacked or a single layer.
        for _ in range(LAYERS):
            # By setting return_sequences to True, return not only the last output but
            # all the outputs so far in the form of (num_samples, timesteps,
            # output_dim). This is necessary as TimeDistributed in the below expects
            # the first dimension to be the timesteps.
            model.add(RNN(HIDDEN_SIZE, return_sequences=True))
        # Apply a dense layer to the every temporal slice of an input. For each of step
        # of the output sequence, decide which character should be chosen.
        model.add(layers.TimeDistributed(layers.Dense(len(chars))))
        model.add(layers.Activation(activation))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.summary()
    train_history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, initial_epoch=initial_epoch, epochs=param_dict['epochs'], validation_data=(x_val, y_val))
    train_loss = train_history.history['loss']
    val_acc = train_history.history['val_acc']
    print('===Train loss:', train_loss[-1])
    print('===Validation accuracy:', val_acc[-1])
    print('OUTPUT:', -val_acc[-1])
    
    model.save(model_name)  
    save_meta_data(param_dict, model_mda_name)
    return -val_acc[-1]

def augment_parser(parser):
    parser.add_argument('--rnn_type', action='store',
                        dest='rnn_type',
                        nargs='?', const=1, type=str, default='LSTM',
                        choices=['LSTM', 'GRU', 'SimpleRNN'],
                        help='type of RNN')

    parser.add_argument('--hidden_size', action='store', dest='hidden_size',
                        nargs='?', const=2, type=int, default='128',
                        help='number of epochs')

    parser.add_argument('--layers', action='store', dest='layers',
                        nargs='?', const=2, type=int, default='1',
                        help='number of epochs')
    return parser


if __name__ == "__main__":
    parser = keras_cmdline.create_parser()
    parser = augment_parser(parser)
    cmdline_args = parser.parse_args()
    param_dict = vars(cmdline_args)
    print(param_dict)
    run(param_dict)
