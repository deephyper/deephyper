Benchmarks
**********

Benchmarks are here for you to test the performance of different search algorithm and reproduce our results. They can also help you to test your installation of deephyper or
discover the many parameters of a search. In deephyper we have two different kind of benchmarks. The first type is `hyper parameters search` benchmarks and the second type is  `neural architecture search` benchmarks. To see a full explanation about the different kind of search please refer to the following page `Search <search.html>`_ . To access the benchmarks from python just use ``deephyper.benchmarks.name``.

Hyper Parameters Search (HPS)
=============================

============== ================ ===============
      Hyper Parameters Search Benchmarks
-----------------------------------------------
     Name            Type          Description
============== ================ ===============
 b1
 b2
 b3
 capsule
 cifar10cnn     Classification   https://www.cs.toronto.edu/~kriz/cifar.html
 dummy1
 dummy2
 gcn
 mnistcnn       Classification   http://yann.lecun.com/exdb/mnist/
 mnistmlp       Classification   http://yann.lecun.com/exdb/mnist/
 rosen2
 rosen10
 rosen30
============== ================ ===============

How to create a benchmark HPS
-----------------------------

For HPS a benchmark is defined by a problem definition and a function that runs the model.

::

      problem_folder/
            __init__.py
            problem.py
            model_run.py

The problem contains the parameters you want to search over. They are defined
by their name, their space and a default value for the starting point. Deephyper recognizes three types of parameters :
- continuous
- discrete ordinal (for instance integers)
- discrete non-ordinal (for instance a list of tokens)
For example if we want to create an hyper parameter search problem for Mnist dataset :


::

    from deephyper.benchmarks import HpProblem

    Problem = HpProblem()
    Problem.add_dim(p_name='num_n_l1', p_space=[i for i in range(1, 30)], p_default=15)


and that's it, we just defined a problem with one dimension 'num_n_l1' where we are going to search the best number of neurons for the first dense layer.

Now we need to define how to run hour mnist model while taking in account this 'num_n_l1' parameter chosen by the search. Let's take an basic example from Keras documentation with a small modification to use the 'num_n_l1' parameter :


::

    '''Trains a simple deep NN on the MNIST dataset.
    Gets to 98.40% test accuracy after 20 epochs
    (there is *a lot* of margin for parameter tuning).
    2 seconds per epoch on a K520 GPU.
    '''

    from __future__ import print_function

    import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.optimizers import RMSprop

    def run_model(param_dict):
        batch_size = 128
        num_classes = 10
        epochs = 20

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

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

        model = Sequential()

        #--------- HERE : we use our 'num_n_l1' parameter ---------------
        num_n_l1 = param_dict['num_n_l1']
        model.add(Dense(num_n_l1, activation='relu', input_shape=(784,)))
        #----------------------------------------------------------------

        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy',
                    optimizer=RMSprop(),
                    metrics=['accuracy'])

        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return score[1]


.. WARNING::
    When designing a new optimization experiment, keep in mind `model_run.py`
    must be runnable from an arbitrary working directory. This means that Python
    modules simply located in the same directory as the `model_run.py` will not be
    part of the default Python import path, and importing them will cause an `ImportError`!

To ensure that modules located alongside the `model_run.py` script are always importable, a
quick workaround is to explicitly add the problem folder to `sys.path` at the top of the script::
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    # import user modules below here
