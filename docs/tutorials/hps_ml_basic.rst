.. _tutorial-hps-machine-learning-basic:

Hyperparameter Search for Machine Learning (Basic)
**************************************************

In this tutorial, we will show how to tune the hyperparameters of the `Random Forest (RF) classifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_
in `scikit-learn <https://scikit-learn.org/stable/>`_ for the Airlines data set.

Let us start by creating a DeepHyper project and a problem for our application:

.. code-block:: console
    :caption: bash

    $ deephyper start-project dhproj
    $ cd dhproj/dhproj/
    $ deephyper new-problem hps rf_tuning
    $ cd rf_tuning/

Create a script to test the accuracy of the baseline model:

.. literalinclude:: content_hps_ml_basic/test_config.py
    :linenos:
    :caption: rf_tuning/test_config.py
    :name: rf_tuning-test_config

Run the script and record the training, validation, and test accuracy as follows:

.. code-block:: console
    :caption: bash

    $ python -i test_config.py
    >>> test_config({})

Running the script will give the the following output:

.. code-block:: python
    :caption: [Out]

    Accuracy on Training: 0.879
    Accuracy on Validation: 0.621
    Accuracy on Testing: 0.620

The accuracy values show that the RandomForest classifier with default hyperparameters results in overfitting and thus poor generalization
(high accuracy on training data but not on the validation and test data).

Next, we optimize the hyperparameters of the RandomForest classifier to address the overfitting problem and improve the accuracy on the vaidation and test data.
Create ``load_data.py`` file to load and return training and validation data:

.. literalinclude:: content_hps_ml_basic/load_data.py
    :linenos:
    :caption: rf_tuning/load_data.py
    :name: rf_tuning-load_data

.. note::
    Subsampling with ``X_train, y_train = resample(X_train, y_train, n_samples=int(1e4))`` can be useful if you want to speed-up your search. By subsampling the training time will reduce.

To test this code:

.. code-block:: console
    :caption: bash

    $ python load_data.py

The expected output is:

.. code-block:: python
    :caption: [Out]

    X_train shape: (10000, 7)
    y_train shape: (10000,)
    X_valid shape: (119258, 7)
    y_valid shape: (119258,)

Create ``model_run.py`` file to train and evaluate the RF model. This function has to return a scalar value (typically, validation accuracy),
which will be maximized by the search algorithm.

.. literalinclude:: content_hps_ml_basic/model_run.py
    :linenos:
    :caption: rf_tuning/model_run.py
    :name: rf_tuning-model_run

Create ``problem.py`` to define the search space of hyperparameters for the RF model:

.. literalinclude:: content_hps_ml_basic/problem.py
    :linenos:
    :caption: rf_tuning/problem.py
    :name: rf_tuning-problem

Run the ``problem.py`` with ``$ python problem.py`` in your shell. The output will be:

.. code-block:: python
    :caption: [Out]

    Configuration space object:
        Hyperparameters:
            criterion, Type: Categorical, Choices: {gini, entropy}, Default: gini
            max_depth, Type: UniformInteger, Range: [1, 50], Default: 26
            min_samples_split, Type: UniformInteger, Range: [2, 10], Default: 6
            n_estimators, Type: UniformInteger, Range: [10, 300], Default: 155


        Starting Point:
        {0: {'criterion': 'gini',
            'max_depth': 50,
            'min_samples_split': 2,
            'n_estimators': 100}}


Run the search for 20 model evaluations using the following command line:

.. code-block:: console
    :caption: bash

    $ deephyper hps ambs --problem dhproj.rf_tuning.problem.Problem --run dhproj.rf_tuning.model_run.run --max-evals 20 --evaluator subprocess --n-jobs 4

Once the search is over, the ``results.csv`` file contains the hyperparameters configurations evaluated during the search and their corresponding objective value (validation accuracy).
Create ``test_best_config.py`` as given belwo. It will extract the best configuration from the ``results.csv`` and run RF with it.

.. code-block:: python
    :caption: rf_tuning/test_best_config.py

    import pandas as pd
    from test_config import test_config

    df = pd.read_csv("results.csv")
    best_config = df.iloc[df.objective.argmax()][:-2].to_dict()
    test_config(best_config)

Compared to the default configuration, we can see the accuracy improvement in the validation and test data sets.

.. code-block:: python
    :caption: [Out]

    Accuracy on Training: 0.748
    Accuracy on Validation: 0.666
    Accuracy on Testing: 0.666

