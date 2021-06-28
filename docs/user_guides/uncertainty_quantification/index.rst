Uncertainty Quantification (UQ)
*******************************

Create the load data function.

.. literalinclude:: load_data.py
    :linenos:
    :caption: cubic/load_data.py
    :name: cubic-load_data

Create the search space with ``SpaceFactory`` and prepare the ``create_search_space`` function.

.. literalinclude:: space.py
    :linenos:
    :caption: cubic/space.py
    :name: cubic-space


Define the problem, here we do a joint HPS + NAS search.

.. literalinclude:: problem.py
    :linenos:
    :caption: cubic/problem.py
    :name: cubic-problem