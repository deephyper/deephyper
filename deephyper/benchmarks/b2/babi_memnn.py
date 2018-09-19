'''Trains a memory network on the bAbI dataset.

References:

- Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush,
  "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks",
  http://arxiv.org/abs/1502.05698

- Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
  "End-To-End Memory Networks",
  http://arxiv.org/abs/1503.08895

Reaches 98.6% accuracy on task 'single_supporting_fact_10k' after 120 epochs.
Time per epoch: 3s on CPU (core i7).
'''
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

from deephyper.benchmarks.util import TerminateOnTimeOut
print("using python:", sys.executable)
print("using deephyper lib:", os.path.abspath(util.__file__))
print("importing keras...")

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from keras.layers import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from functools import reduce
import tarfile
import numpy as np
import re

from keras import layers
from deephyper.benchmarks import keras_cmdline
import hashlib
import pickle
print("using keras:", layers.__path__)

from numpy.random import seed
from tensorflow import set_random_seed
timer.end()

seed(1)
set_random_seed(2)


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format

    If only_supporting is true, only the sentences
    that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file,
    retrieve the stories,
    and then convert the sentences into a single story.

    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    inputs, queries, answers = [], [], []
    for story, query, answer in data:
        inputs.append([word_idx[w] for w in story])
        queries.append([word_idx[w] for w in query])
        answers.append(word_idx[answer])
    return (pad_sequences(inputs, maxlen=story_maxlen),
            pad_sequences(queries, maxlen=query_maxlen),
            np.array(answers))
    

def run(param_dict):
    param_dict = keras_cmdline.fill_missing_defaults(augment_parser, param_dict)
    optimizer = keras_cmdline.return_optimizer(param_dict)
    pprint(param_dict)
    
    challenges = {
        # QA1 with 10,000 samples
        'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
        # QA2 with 10,000 samples
        'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
    }
    challenge_type = 'single_supporting_fact_10k'
    challenge = challenges[challenge_type]
    
    timer.start('stage in')
    if param_dict['data_source']:
        data_source = param_dict['data_source']
    else:
        data_source = os.path.dirname(os.path.abspath(__file__))
        data_source = os.path.join(data_source, 'data')

    try:
        paths = util.stage_in(['babi-tasks-v1-2.tar.gz'],
                              source=data_source,
                              dest=param_dict['stage_in_destination'])
        path = paths['babi-tasks-v1-2.tar.gz']
    except:
        print('Error downloading dataset, please download it manually:\n'
              '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
              '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
        raise

    print('Extracting stories for the challenge:', challenge_type)
    with tarfile.open(path) as tar:
        train_stories = get_stories(tar.extractfile(challenge.format('train')))
        test_stories = get_stories(tar.extractfile(challenge.format('test')))
    timer.end()

    timer.start('preprocessing')

    vocab = set()
    for story, q, answer in train_stories + test_stories:
        vocab |= set(story + q + [answer])
    vocab = sorted(vocab)

    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
    story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
    query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

    print('-')
    print('Vocab size:', vocab_size, 'unique words')
    print('Story max length:', story_maxlen, 'words')
    print('Query max length:', query_maxlen, 'words')
    print('Number of training stories:', len(train_stories))
    print('Number of test stories:', len(test_stories))
    print('-')
    print('Here\'s what a "story" tuple looks like (input, query, answer):')
    print(train_stories[0])
    print('-')
    print('Vectorizing the word sequences...')

    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    inputs_train, queries_train, answers_train = vectorize_stories(train_stories, word_idx, story_maxlen, query_maxlen)
    inputs_test, queries_test, answers_test = vectorize_stories(test_stories, word_idx, story_maxlen, query_maxlen)

    print('-')
    print('inputs: integer tensor of shape (samples, max_length)')
    print('inputs_train shape:', inputs_train.shape)
    print('inputs_test shape:', inputs_test.shape)
    print('-')
    print('queries: integer tensor of shape (samples, max_length)')
    print('queries_train shape:', queries_train.shape)
    print('queries_test shape:', queries_test.shape)
    print('-')
    print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
    print('answers_train shape:', answers_train.shape)
    print('answers_test shape:', answers_test.shape)
    print('-')
    print('Compiling...')


    BATCH_SIZE = param_dict['batch_size']
    EPOCHS = param_dict['epochs']
    DROPOUT = param_dict['dropout']
    NHIDDEN = param_dict['nhidden']
    ACTIVATION = param_dict['activation']
    TIMEOUT = param_dict['timeout']

    if param_dict['rnn_type'] == 'GRU':
        RNN = layers.GRU
    elif param_dict['rnn_type'] == 'SimpleRNN':
        RNN = layers.SimpleRNN
    else:
        RNN = layers.LSTM

    
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
        # placeholders
        input_sequence = Input((story_maxlen,))
        question = Input((query_maxlen,))

        # encoders
        # embed the input sequence into a sequence of vectors
        input_encoder_m = Sequential()
        input_encoder_m.add(Embedding(input_dim=vocab_size,
                                    output_dim=64))
        input_encoder_m.add(Dropout(DROPOUT))
        # output: (samples, story_maxlen, embedding_dim)

        # embed the input into a sequence of vectors of size query_maxlen
        input_encoder_c = Sequential()
        input_encoder_c.add(Embedding(input_dim=vocab_size,
                                    output_dim=query_maxlen))
        input_encoder_c.add(Dropout(DROPOUT))
        # output: (samples, story_maxlen, query_maxlen)

        # embed the question into a sequence of vectors
        question_encoder = Sequential()
        question_encoder.add(Embedding(input_dim=vocab_size,
                                    output_dim=64,
                                    input_length=query_maxlen))
        question_encoder.add(Dropout(DROPOUT))
        # output: (samples, query_maxlen, embedding_dim)

        # encode input sequence and questions (which are indices)
        # to sequences of dense vectors
        input_encoded_m = input_encoder_m(input_sequence)
        input_encoded_c = input_encoder_c(input_sequence)
        question_encoded = question_encoder(question)

        # compute a 'match' between the first input vector sequence
        # and the question vector sequence
        # shape: `(samples, story_maxlen, query_maxlen)`
        match = dot([input_encoded_m, question_encoded], axes=(2, 2))
        match = Activation('softmax')(match)

        # add the match matrix with the second input vector sequence
        response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
        response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

        # concatenate the match matrix with the question vector sequence
        answer = concatenate([response, question_encoded])

        # the original paper uses a matrix multiplication for this reduction step.
        # we choose to use a RNN instead.
        answer = RNN(NHIDDEN, activation=ACTIVATION)(answer)  # (samples, 32)

        # one regularization layer -- more would probably be needed.
        answer = Dropout(DROPOUT)(answer)
        answer = Dense(vocab_size)(answer)  # (samples, vocab_size)
        # we output a probability distribution over the vocabulary
        answer = Activation('softmax')(answer)

        # build the final model
        model = Model([input_sequence, question], answer)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        model.summary()

    timer.end()


    #earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=50, verbose=1, mode='auto')
    timeout_monitor = TerminateOnTimeOut(TIMEOUT)
    callbacks_list = [timeout_monitor]

    timer.start('model training')
    train_history = model.fit([inputs_train, queries_train], answers_train, callbacks=callbacks_list, 
                                batch_size=BATCH_SIZE, initial_epoch=initial_epoch, epochs=EPOCHS, validation_split=0.30) #, validation_data=([inputs_test, queries_test], answers_test))
    timer.end()
    
    score = model.evaluate([inputs_test, queries_test], answers_test, batch_size=BATCH_SIZE)
    print('===Validation loss:', score[0])
    print('===Validation accuracy:', score[1])
    print('OUTPUT:', -score[1])

    
    #train_loss = train_history.history['loss']
    #val_acc = train_history.history['val_acc']
    #print('===Train loss:', train_loss[-1])
    #print('===Validation accuracy:', val_acc[-1])
    #print('OUTPUT:', -val_acc[-1])
    
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
                        nargs='?', const=2, type=int, default='128',
                        help='number of epochs')
    return parser


if __name__ == "__main__":
    parser = keras_cmdline.create_parser()
    parser = augment_parser(parser)
    cmdline_args = parser.parse_args()
    param_dict = vars(cmdline_args)
    run(param_dict)
