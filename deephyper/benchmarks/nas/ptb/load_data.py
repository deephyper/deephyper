"""
Created by Dipendra Jha (dipendra@u.northwestern.edu) on 7/24/18

Utilities for parsing PTB text files.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import numpy as np

import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if sys.version_info[0] >= 3:
      return f.read().replace("\n", "<eos>").split()
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def _create_input_output(raw_data, batch_size, num_steps):
    data_len = len(raw_data)
    batch_len = data_len//batch_size
    data = np.reshape(raw_data[:batch_size*batch_len],(batch_size, batch_len))
    epoch_size = (batch_len-1)//num_steps
    print(data_len, batch_len, epoch_size)
    assert epoch_size > 0, "epoch_size == 0, decrease batch_size or num_steps"
    start_inds = range(epoch_size)
    X = np.reshape(data[:,:epoch_size*num_steps], (batch_size, epoch_size, num_steps))
    Y = np.reshape(data[:,1:epoch_size*num_steps+1], (batch_size, epoch_size, num_steps))
    print (X.shape, Y.shape)
    X = np.reshape(X, (batch_size, epoch_size, num_steps))
    Y = np.reshape(Y, (batch_size, epoch_size, num_steps))
    X = np.swapaxes(X,0,1)
    Y = np.swapaxes(Y,0,1)
    X = np.reshape(X, (-1, num_steps))
    Y = np.reshape(Y, (-1, num_steps))
    return (X, Y)


def load_data(config=None):
    """Load PTB raw data from data directory "data_path".
    Reads PTB text files, converts strings to integer ids and create train, valid and test arrays.
    The PTB dataset comes from Tomas Mikolov's webpage:

    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

    Args:
        data_path: string path to the directory where simple-examples.tgz has been extracted.
        num_steps: the number of unrolls (time steps).
        batch_size: the batch size

    Returns:
        tuple(train_X, train_y, valid_X, valid_y, test_X, test_y, vocabulary)

    """
    num_steps = 10
    data_path = '/projects/datascience/regele/deephyper/benchmark/ptbNas/DATA'
    batch_size = 32
    if config and 'batch_size' in config: batch_size = config['batch_size']
    if config and 'num_steps' in config: num_steps = config['num_steps']
    #if config and 'dest' in config: data_path = config['dest']
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    train_X, train_y = _create_input_output(train_data,batch_size, num_steps)
    valid_X, valid_y = _create_input_output(valid_data, batch_size, num_steps)
    test_X, test_y = _create_input_output(test_data, batch_size, num_steps)
    return (train_X, train_y), (valid_X, valid_y), (test_X, test_y), word_to_id
