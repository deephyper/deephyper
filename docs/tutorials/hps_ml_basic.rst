.. _tutorial-hps-machine-learning-basic:

Hyperparameter Search for Machine Learning (Basic)
**************************************************


In this tutorial we will show how to search the Hyperparameters of the Random Forest (RF) model for the Arlines data set.

Let us start by creating a DeepHyper project and a problem for our application:

.. code-block:: console
    :caption: bash

    $ deephyper start-project dhproj
    $ cd dhproj/dhproj/
    $ deephyper new-problem hps rf_tuning
    $ cd rf_tuning/

Now we can create a script to test our model performance:

.. literalinclude:: content_hps_ml_basic/test_config.py
    :linenos:
    :caption: rf_tuning/test_config.py
    :name: rf_tuning-test_config

We can execute this fonction with:

.. code-block:: console
    :caption: bash

    $ python -i test_config.py
    >>> test_config({})

which gives the the following output:

.. code-block:: python
    :caption: [Out]

    Accuracy on Training: 0.879
    Accuracy on Validation: 0.621
    Accuracy on Testing: 0.620

We can clearly see that our RandomForest classifier is overfitting the training data set by looking at the difference between training and validation/testing accuracies. Now that we have this baseline performence we can optimize the hyperparameters of our model. Let us define the ``load_data`` to return training and validation data:

.. literalinclude:: content_hps_ml_basic/load_data.py
    :linenos:
    :caption: rf_tuning/load_data.py
    :name: rf_tuning-load_data

To test this code:

.. code-block:: console
    :caption: bash

    $ python load_data.py

The expected output is:

.. code-block:: python
    :caption: [Out]

    X_train shape: (260816, 54)
    y_train shape: (260816,)
    X_valid shape: (128462, 54)
    y_valid shape: (128462,)

Now, by taking inspiration from our ``test_config`` function we can create the ``run`` function where the model will be evaluated. This function has to return a scalar value which will be maximized by the search algorithm.

.. literalinclude:: content_hps_ml_basic/model_run.py
    :linenos:
    :caption: rf_tuning/model_run.py
    :name: rf_tuning-model_run

The last piece of code that we need is the search space of hyperparameters that we want to explore, defined in ``problem.py``:

.. literalinclude:: content_hps_ml_basic/problem.py
    :linenos:
    :caption: rf_tuning/problem.py
    :name: rf_tuning-problem

You can run the ``problem.py`` with ``$ python problem.py`` in your shell and should expect the corresponding outpout:

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


You can now run the search for 20 iterations by executing:

.. code-block:: console
    :caption: bash

    $ deephyper hps ambs --problem dhproj.rf_tuning.problem.Problem --run dhproj.rf_tuning.model_run.run --max-evals 20 --evaluator subprocess --n-jobs 4


Once the search has finished, the ``results.csv`` file contains the hyperparameters configurations tried during the search and their corresponding objective value.

.. code-block:: python
    :caption: rf_tuning/test_best_config.py

    import pandas as pd
    from test_config import test_config

    df = pd.read_csv("results.csv")
    best_config = df.iloc[df.objective.argmax()][:-2].to_dict()
    test_config(best_config)

Which gives as a good performance improvement! Now with a 0.66 accuracy on the test set when it was before 0.62. We can also observe a reduction of the overfitting issue:

.. code-block:: python
    :caption: [Out]

    Accuracy on Training: 0.748
    Accuracy on Validation: 0.666
    Accuracy on Testing: 0.666