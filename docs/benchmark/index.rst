********************
Available benchmarks
********************

.. automodule:: deephyper.benchmark

.. _available-hps-benchmarks:

Available HPS benchmarks
========================

.. automodule:: deephyper.benchmark.hps

============== ================ =====================================
      Hyper Parameters Search Benchmarks ``deephyper.benchmark.hps``
---------------------------------------------------------------------
     Name            Type                    Description
============== ================ =====================================
 mnistmlp       Classification   http://yann.lecun.com/exdb/mnist/
 polynome2      Dummy
============== ================ =====================================

.. _available-nas-benchmarks:

Available NAS benchmarks
========================

.. automodule:: deephyper.benchmark.nas

============== ================ ========================================
      Neural Architecture Search Benchmarks ``deephyper.benchmark.nas``
------------------------------------------------------------------------
     Name            Type          Description
============== ================ ========================================
 ackleyReg      Regression       Generation of points in N dimensions corresponding to y=f(x) where f is https://www.sfu.ca/~ssurjano/ackley.html
 cifar10        Classification   https://www.cs.toronto.edu/~kriz/cifar.html
 dixonpriceReg  Regression       https://www.sfu.ca/~ssurjano/dixonpr.html
 levyReg        Regression       Generation of points in N dimensions corresponding to y=f(x) where f is https://www.sfu.ca/~ssurjano/levy.html
 linearReg      Regression       Generation of points in N dimensions corresponding to y=x
 mnistNas       Classification   http://yann.lecun.com/exdb/mnist/
 polynome2Reg   Regression       Generation of points in N dimensions corresponding to y=sum(x_i^2)
 saddleReg      Regression       https://en.wikipedia.org/wiki/Saddle_point
============== ================ ========================================