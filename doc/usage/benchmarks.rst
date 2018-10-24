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
            problem.py
            model_run.py

The problem contains the parameters you want to search over. They are defined by their name, their type and space. It also contains the starting point of the search. Three types of parameters exists :
      - continuous
      - discrete ordinal (for instance integers)
      - discrete non ordinal (for instance a list of tokens)




Neural Architecture Search
==========================

============== ================ ===============
      Neural Architecture Search Benchmarks
-----------------------------------------------
     Name            Type          Description
============== ================ ===============
 cifar10Nas     Classification   https://www.cs.toronto.edu/~kriz/cifar.html
 linearRegNas   Regression       Generation of points in 1 dimension corresponding to y=x
 mnistNas       Classification   http://yann.lecun.com/exdb/mnist/
============== ================ ===============
