
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "examples/plot_profile_worker_utilization.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_examples_plot_profile_worker_utilization.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_plot_profile_worker_utilization.py:


Profile the Worker Utilization
==============================

**Author(s)**: Romain Egele.

This example demonstrates the advantages of parallel evaluations over serial
evaluations. We start by defining an artificial black-box ``run``-function by
using the Ackley function:

.. image:: https://www.sfu.ca/~ssurjano/ackley.png
  :width: 400
  :alt: Ackley Function in 2D

We will use the ``time.sleep`` function to simulate a budget of 2 secondes of
execution in average which helps illustrate the advantage of parallel
evaluations. The ``@profile`` decorator is useful to collect starting/ending
time of the ``run``-function execution which help us know exactly when we are
inside the black-box. This decorator is necessary when profiling the worker
utilization. When using this decorator, the ``run``-function will return a
dictionnary with 2 new keys ``"timestamp_start"`` and ``"timestamp_end"``.
The ``run``-function is defined in a separate module because of
the "multiprocessing" backend that we are using in this example.

.. literalinclude:: ../../examples/black_box_util.py
   :language: python
   :emphasize-lines: 19-28
   :linenos:

After defining the black-box we can continue with the definition of our main script:

.. GENERATED FROM PYTHON SOURCE LINES 33-46

.. code-block:: Python


    import black_box_util as black_box
    import matplotlib.pyplot as plt

    from deephyper.analysis import figure_size
    from deephyper.analysis.hpo import (
        plot_search_trajectory_single_objective_hpo,
        plot_worker_utilization,
    )
    from deephyper.evaluator import Evaluator
    from deephyper.evaluator.callback import TqdmCallback
    from deephyper.hpo import CBO, HpProblem








.. GENERATED FROM PYTHON SOURCE LINES 47-50

Then we define the variable(s) we want to optimize. For this problem we
optimize Ackley in a 2-dimensional search space, the true minimul is
located at ``(0, 0)``.

.. GENERATED FROM PYTHON SOURCE LINES 50-58

.. code-block:: Python


    nb_dim = 2
    problem = HpProblem()
    for i in range(nb_dim):
        problem.add_hyperparameter((-32.768, 32.768), f"x{i}")
    problem






.. rst-class:: sphx-glr-script-out

 .. code-block:: none


    Configuration space object:
      Hyperparameters:
        x0, Type: UniformFloat, Range: [-32.768, 32.768], Default: 0.0
        x1, Type: UniformFloat, Range: [-32.768, 32.768], Default: 0.0




.. GENERATED FROM PYTHON SOURCE LINES 59-60

Then we define a parallel search.

.. GENERATED FROM PYTHON SOURCE LINES 60-81

.. code-block:: Python


    if __name__ == "__main__":
        timeout = 20
        num_workers = 4
        results = {}

        evaluator = Evaluator.create(
            black_box.run_ackley,
            method="process",
            method_kwargs={
                "num_workers": num_workers,
                "callbacks": [TqdmCallback()],
            },
        )
        search = CBO(
            problem,
            evaluator,
            random_state=42,
        )
        results = search.search(timeout=timeout)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    WARNING:root:Results file already exists, it will be renamed to /Users/romainegele/Documents/Argonne/deephyper/examples/results_20241125-183504.csv
    0it [00:00, ?it/s]    1it [00:00, 6423.13it/s, failures=0, objective=-19.8]    2it [00:00,  6.45it/s, failures=0, objective=-19.8]      2it [00:00,  6.45it/s, failures=0, objective=-19.8]    3it [00:00,  6.45it/s, failures=0, objective=-19.8]    4it [00:00,  5.37it/s, failures=0, objective=-19.8]    4it [00:00,  5.37it/s, failures=0, objective=-19.8]    5it [00:01,  2.74it/s, failures=0, objective=-19.8]    5it [00:01,  2.74it/s, failures=0, objective=-19.8]    6it [00:01,  3.38it/s, failures=0, objective=-19.8]    6it [00:01,  3.38it/s, failures=0, objective=-19.8]    7it [00:02,  2.19it/s, failures=0, objective=-19.8]    7it [00:02,  2.19it/s, failures=0, objective=-19.8]    8it [00:02,  2.33it/s, failures=0, objective=-19.8]    8it [00:02,  2.33it/s, failures=0, objective=-15.4]    9it [00:03,  2.03it/s, failures=0, objective=-15.4]    9it [00:03,  2.03it/s, failures=0, objective=-15.4]    10it [00:04,  1.92it/s, failures=0, objective=-15.4]    10it [00:04,  1.92it/s, failures=0, objective=-15.4]    11it [00:04,  2.43it/s, failures=0, objective=-15.4]    11it [00:04,  2.43it/s, failures=0, objective=-15.4]    12it [00:05,  1.89it/s, failures=0, objective=-15.4]    12it [00:05,  1.89it/s, failures=0, objective=-12.6]    13it [00:06,  1.45it/s, failures=0, objective=-12.6]    13it [00:06,  1.45it/s, failures=0, objective=-12.6]    14it [00:06,  1.94it/s, failures=0, objective=-12.6]    14it [00:06,  1.94it/s, failures=0, objective=-12.6]    15it [00:06,  2.34it/s, failures=0, objective=-12.6]    15it [00:06,  2.34it/s, failures=0, objective=-12.6]    16it [00:06,  2.57it/s, failures=0, objective=-12.6]    16it [00:06,  2.57it/s, failures=0, objective=-12.6]    17it [00:07,  1.79it/s, failures=0, objective=-12.6]    17it [00:07,  1.79it/s, failures=0, objective=-6.37]    18it [00:08,  1.55it/s, failures=0, objective=-6.37]    18it [00:08,  1.55it/s, failures=0, objective=-6.37]    19it [00:08,  1.79it/s, failures=0, objective=-6.37]    19it [00:08,  1.79it/s, failures=0, objective=-6.37]    20it [00:09,  2.25it/s, failures=0, objective=-6.37]    20it [00:09,  2.25it/s, failures=0, objective=-6.37]    21it [00:09,  1.72it/s, failures=0, objective=-6.37]    21it [00:09,  1.72it/s, failures=0, objective=-6.37]    22it [00:10,  1.79it/s, failures=0, objective=-6.37]    22it [00:10,  1.79it/s, failures=0, objective=-6.37]    23it [00:10,  1.95it/s, failures=0, objective=-6.37]    23it [00:10,  1.95it/s, failures=0, objective=-6.37]    24it [00:11,  1.76it/s, failures=0, objective=-6.37]    24it [00:11,  1.76it/s, failures=0, objective=-6.37]    25it [00:12,  1.58it/s, failures=0, objective=-6.37]    25it [00:12,  1.58it/s, failures=0, objective=-6.37]    26it [00:12,  1.85it/s, failures=0, objective=-6.37]    26it [00:12,  1.85it/s, failures=0, objective=-6.37]    27it [00:12,  2.37it/s, failures=0, objective=-6.37]    27it [00:12,  2.37it/s, failures=0, objective=-6.37]    28it [00:13,  1.71it/s, failures=0, objective=-6.37]    28it [00:13,  1.71it/s, failures=0, objective=-6.37]    29it [00:14,  1.65it/s, failures=0, objective=-6.37]    29it [00:14,  1.65it/s, failures=0, objective=-6.37]    30it [00:15,  1.53it/s, failures=0, objective=-6.37]    30it [00:15,  1.53it/s, failures=0, objective=-6.37]    31it [00:15,  1.97it/s, failures=0, objective=-6.37]    31it [00:15,  1.97it/s, failures=0, objective=-6.37]    32it [00:15,  1.95it/s, failures=0, objective=-6.37]    32it [00:15,  1.95it/s, failures=0, objective=-6.37]    33it [00:16,  1.84it/s, failures=0, objective=-6.37]    33it [00:16,  1.84it/s, failures=0, objective=-6.37]    34it [00:16,  1.96it/s, failures=0, objective=-6.37]    34it [00:16,  1.96it/s, failures=0, objective=-6.37]    35it [00:18,  1.15it/s, failures=0, objective=-6.37]    35it [00:18,  1.15it/s, failures=0, objective=-6.37]    36it [00:18,  1.15it/s, failures=0, objective=-6.37]    37it [00:18,  1.15it/s, failures=0, objective=-6.37]



.. GENERATED FROM PYTHON SOURCE LINES 82-83

Finally, we plot the results from the collected DataFrame.

.. GENERATED FROM PYTHON SOURCE LINES 83-107

.. code-block:: Python


    if __name__ == "__main__":
        t0 = results["m:timestamp_start"].iloc[0]
        results["m:timestamp_start"] = results["m:timestamp_start"] - t0
        results["m:timestamp_end"] = results["m:timestamp_end"] - t0
        tmax = results["m:timestamp_end"].max()

        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            figsize=figure_size(width=600),
        )

        plot_search_trajectory_single_objective_hpo(
            results, mode="min", x_units="seconds", ax=axes[0]
        )

        plot_worker_utilization(
            results, num_workers=num_workers, profile_type="start/end", ax=axes[1]
        )

        plt.tight_layout()
        plt.show()



.. image-sg:: /examples/images/sphx_glr_plot_profile_worker_utilization_001.png
   :alt: plot profile worker utilization
   :srcset: /examples/images/sphx_glr_plot_profile_worker_utilization_001.png
   :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    /Users/romainegele/Documents/Argonne/deephyper/examples/plot_profile_worker_utilization.py:106: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      plt.show()





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 24.680 seconds)


.. _sphx_glr_download_examples_plot_profile_worker_utilization.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_profile_worker_utilization.ipynb <plot_profile_worker_utilization.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_profile_worker_utilization.py <plot_profile_worker_utilization.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_profile_worker_utilization.zip <plot_profile_worker_utilization.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
