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
import os
from pprint import pprint
import sys

here = os.path.dirname(os.path.abspath(__file__))
top = os.path.dirname(os.path.dirname(os.path.dirname(here)))
sys.path.append(top)
BNAME = os.path.splitext(os.path.basename(__file__))[0]

from deephyper.benchmarks import util 

timer = util.Timer()
timer.start('module loading')

from deephyper.benchmarks.util import TerminateOnTimeOut
import numpy as np
from six.moves import range
from keras.models import Sequential
from keras import layers
from keras.models import load_model
from deephyper.benchmarks import keras_cmdline 
from keras.callbacks import EarlyStopping
from numpy.random import seed
from tensorflow import set_random_seed
timer.end()

seed(1)
set_random_seed(2)
    
TRAINING_SIZE = 50000
DIGITS = 3
INVERT = True
MAXLEN = DIGITS + 1 + DIGITS


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

def run(param_dict):
    timer.start('preprocessing')
    param_dict = keras_cmdline.fill_missing_defaults(augment_parser, param_dict)
    pprint(param_dict)
    optimizer = keras_cmdline.return_optimizer(param_dict)

    x_train, y_train, x_val, y_val, chars = generate_data()

    if param_dict['rnn_type'] == 'GRU':
        RNN = layers.GRU
    elif param_dict['rnn_type'] == 'SimpleRNN':
        RNN = layers.SimpleRNN
    else:
        RNN = layers.LSTM

    HIDDEN_SIZE = param_dict['nhidden']
    BATCH_SIZE = param_dict['batch_size']
    NLAYERS = param_dict['nlayers']
    DROPOUT = param_dict['dropout']
    ACTIVATION = param_dict['activation']
    EPOCHS = param_dict['epochs']
    TIMEOUT = param_dict['timeout']
    patience = param_dict['patience']
    delta = param_dict['delta']

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
        print('Building model...')
        model = Sequential()
        # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
        # Note: In a situation where your input sequences have a variable length,
        # use input_shape=(None, num_feature).
        model.add(RNN(HIDDEN_SIZE, activation=ACTIVATION, dropout=DROPOUT, input_shape=(MAXLEN, len(chars))))
        # As the decoder RNN's input, repeatedly provide with the last hidden state of
        # RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
        # length of output, e.g., when DIGITS=3, max output is 999+999=1998.
        model.add(layers.RepeatVector(DIGITS + 1))
        # The decoder RNN could be multiple layers stacked or a single layer.
        for _ in range(NLAYERS):
            # By setting return_sequences to True, return not only the last output but
            # all the outputs so far in the form of (num_samples, timesteps,
            # output_dim). This is necessary as TimeDistributed in the below expects
            # the first dimension to be the timesteps.
            model.add(RNN(HIDDEN_SIZE, activation=ACTIVATION, dropout=DROPOUT, return_sequences=True))
        # Apply a dense layer to the every temporal slice of an input. For each of step
        # of the output sequence, decide which character should be chosen.
        model.add(layers.TimeDistributed(layers.Dense(len(chars))))
        model.add(layers.Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.summary()
    timer.end()

    earlystop = EarlyStopping(monitor='val_acc', min_delta=delta, patience=patience, verbose=1, mode='auto')
    timeout_monitor = TerminateOnTimeOut(TIMEOUT)
    callbacks_list = [timeout_monitor]

    timer.start('model training')
    train_history = model.fit(x_train, y_train, callbacks=callbacks_list, batch_size=BATCH_SIZE, 
                                initial_epoch=initial_epoch, epochs=EPOCHS, validation_split=0.30)#, validation_data=(x_val, y_val))
    timer.end()
    
    score = model.evaluate(x_val, y_val, batch_size=BATCH_SIZE)
    print('===Validation loss:', score[0])
    print('===Validation accuracy:', score[1])
    print('OUTPUT:', -score[1])
    
    if model_path:
        timer.start('model save')
        model.save(model_path)
        util.save_meta_data(param_dict, model_mda_path)
        timer.end()
    return -score[1]

def augment_parser(parser):
    parser.add_argument('--rnn_type', action='store',
                        dest='rnn_type',
                        nargs='?', const=1, type=str, default='LSTM',
                        choices=['LSTM', 'GRU', 'SimpleRNN'],
                        help='type of RNN')

    parser.add_argument('--nhidden', action='store', dest='nhidden',
                        nargs='?', const=2, type=int, default='128',)

    parser.add_argument('--nlayers', action='store', dest='nlayers',
                        nargs='?', const=2, type=int, default='1',)
    return parser


if __name__ == "__main__":
    parser = keras_cmdline.create_parser()
    parser = augment_parser(parser)
    cmdline_args = parser.parse_args()
    param_dict = vars(cmdline_args)
    print(param_dict)
    run(param_dict)
