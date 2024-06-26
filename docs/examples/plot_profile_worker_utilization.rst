
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "examples/plot_profile_worker_utilization.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_examples_plot_profile_worker_utilization.py>`
        to download the full example code

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_plot_profile_worker_utilization.py:


Profile the Worker Utilization
==============================

**Author(s)**: Romain Egele.

This example demonstrates the advantages of parallel evaluations over serial evaluations. We start by defining an artificial black-box ``run``-function by using the Ackley function:

.. image:: https://www.sfu.ca/~ssurjano/ackley.png
  :width: 400
  :alt: Ackley Function in 2D

We will use the ``time.sleep`` function to simulate a budget of 2 secondes of execution in average which helps illustrate the advantage of parallel evaluations. The ``@profile`` decorator is useful to collect starting/ending time of the ``run``-function execution which help us know exactly when we are inside the black-box. This decorator is necessary when profiling the worker utilization. When using this decorator, the ``run``-function will return a dictionnary with 2 new keys ``"timestamp_start"`` and ``"timestamp_end"``. The ``run``-function is defined in a separate module because of the "multiprocessing" backend that we are using in this example.

.. literalinclude:: ../../examples/black_box_util.py
   :language: python
   :emphasize-lines: 19-28 
   :linenos:

After defining the black-box we can continue with the definition of our main script:

.. GENERATED FROM PYTHON SOURCE LINES 23-29

.. code-block:: default

    import black_box_util as black_box

    from deephyper.analysis._matplotlib import update_matplotlib_rc

    update_matplotlib_rc()








.. GENERATED FROM PYTHON SOURCE LINES 30-31

Then we define the variable(s) we want to optimize. For this problem we optimize Ackley in a 2-dimensional search space, the true minimul is located at ``(0, 0)``.

.. GENERATED FROM PYTHON SOURCE LINES 31-41

.. code-block:: default

    from deephyper.problem import HpProblem


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




.. GENERATED FROM PYTHON SOURCE LINES 42-43

Then we define a parallel search.

.. GENERATED FROM PYTHON SOURCE LINES 43-63

.. code-block:: default

    if __name__ == "__main__":
        from deephyper.evaluator import Evaluator
        from deephyper.evaluator.callback import TqdmCallback
        from deephyper.search.hps import CBO

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
        search = CBO(problem, evaluator, random_state=42)
        results = search.search(timeout=timeout)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none


    0it [00:00, ?it/s]
    1it [00:00, 7037.42it/s, failures=0, objective=-19.8]
    2it [00:00,  2.10it/s, failures=0, objective=-19.8]  
    2it [00:00,  2.10it/s, failures=0, objective=-19.8]
    3it [00:01,  2.27it/s, failures=0, objective=-19.8]
    3it [00:01,  2.27it/s, failures=0, objective=-19.8]
    4it [00:01,  2.46it/s, failures=0, objective=-19.8]
    4it [00:01,  2.46it/s, failures=0, objective=-19.8]
    5it [00:01,  2.46it/s, failures=0, objective=-19.8]
    6it [00:03,  1.78it/s, failures=0, objective=-19.8]
    6it [00:03,  1.78it/s, failures=0, objective=-19.8]
    7it [00:03,  1.78it/s, failures=0, objective=-15.4]
    8it [00:03,  2.55it/s, failures=0, objective=-15.4]
    8it [00:03,  2.55it/s, failures=0, objective=-15.4]
    9it [00:03,  2.55it/s, failures=0, objective=-15.4]
    10it [00:04,  2.63it/s, failures=0, objective=-15.4]
    10it [00:04,  2.63it/s, failures=0, objective=-15.4]
    11it [00:04,  2.29it/s, failures=0, objective=-15.4]
    11it [00:04,  2.29it/s, failures=0, objective=-15.4]
    12it [00:05,  2.21it/s, failures=0, objective=-15.4]
    12it [00:05,  2.21it/s, failures=0, objective=-15.4]
    13it [00:05,  2.69it/s, failures=0, objective=-15.4]
    13it [00:05,  2.69it/s, failures=0, objective=-12.6]
    14it [00:06,  1.70it/s, failures=0, objective=-12.6]
    14it [00:06,  1.70it/s, failures=0, objective=-12.6]
    15it [00:07,  1.74it/s, failures=0, objective=-12.6]
    15it [00:07,  1.74it/s, failures=0, objective=-12.6]
    16it [00:07,  2.23it/s, failures=0, objective=-12.6]
    16it [00:07,  2.23it/s, failures=0, objective=-12.6]
    17it [00:07,  2.23it/s, failures=0, objective=-12.6]
    18it [00:08,  1.73it/s, failures=0, objective=-12.6]
    18it [00:08,  1.73it/s, failures=0, objective=-12.6]
    19it [00:08,  1.73it/s, failures=0, objective=-5.88]
    20it [00:09,  2.40it/s, failures=0, objective=-5.88]
    20it [00:09,  2.40it/s, failures=0, objective=-5.62]
    21it [00:09,  2.32it/s, failures=0, objective=-5.62]
    21it [00:09,  2.32it/s, failures=0, objective=-5.62]
    22it [00:09,  2.57it/s, failures=0, objective=-5.62]
    22it [00:09,  2.57it/s, failures=0, objective=-5.62]
    23it [00:10,  1.91it/s, failures=0, objective=-5.62]
    23it [00:10,  1.91it/s, failures=0, objective=-5.62]
    24it [00:11,  2.14it/s, failures=0, objective=-5.62]
    24it [00:11,  2.14it/s, failures=0, objective=-5.62]
    25it [00:11,  1.73it/s, failures=0, objective=-5.62]
    25it [00:11,  1.73it/s, failures=0, objective=-5.62]
    26it [00:12,  2.18it/s, failures=0, objective=-5.62]
    26it [00:12,  2.18it/s, failures=0, objective=-5.62]
    27it [00:12,  1.98it/s, failures=0, objective=-5.62]
    27it [00:12,  1.98it/s, failures=0, objective=-5.62]
    28it [00:12,  2.35it/s, failures=0, objective=-5.62]
    28it [00:12,  2.35it/s, failures=0, objective=-5.62]
    29it [00:14,  1.60it/s, failures=0, objective=-5.62]
    29it [00:14,  1.60it/s, failures=0, objective=-5.62]
    30it [00:14,  1.67it/s, failures=0, objective=-5.62]
    30it [00:14,  1.67it/s, failures=0, objective=-5.62]
    31it [00:14,  1.98it/s, failures=0, objective=-5.62]
    31it [00:14,  1.98it/s, failures=0, objective=-5.62]
    32it [00:15,  2.39it/s, failures=0, objective=-5.62]
    32it [00:15,  2.39it/s, failures=0, objective=-5.62]
    33it [00:16,  1.66it/s, failures=0, objective=-5.62]
    33it [00:16,  1.66it/s, failures=0, objective=-5.62]
    34it [00:16,  1.84it/s, failures=0, objective=-5.62]
    34it [00:16,  1.84it/s, failures=0, objective=-5.62]
    35it [00:16,  1.96it/s, failures=0, objective=-5.62]
    35it [00:16,  1.96it/s, failures=0, objective=-5.62]



.. GENERATED FROM PYTHON SOURCE LINES 64-65

Finally, we plot the results from the collected DataFrame.

.. GENERATED FROM PYTHON SOURCE LINES 65-118

.. code-block:: default

    if __name__ == "__main__":
        import matplotlib.pyplot as plt
        import numpy as np

        def compile_profile(df):
            """Take the results dataframe as input and return the number of jobs running at a given timestamp."""
            history = []

            for _, row in df.iterrows():
                history.append((row["m:timestamp_start"], 1))
                history.append((row["m:timestamp_end"], -1))

            history = sorted(history, key=lambda v: v[0])
            nb_workers = 0
            timestamp = [0]
            n_jobs_running = [0]
            for time, incr in history:
                nb_workers += incr
                timestamp.append(time)
                n_jobs_running.append(nb_workers)

            return timestamp, n_jobs_running

        t0 = results["m:timestamp_start"].iloc[0]
        results["m:timestamp_start"] = results["m:timestamp_start"] - t0
        results["m:timestamp_end"] = results["m:timestamp_end"] - t0
        tmax = results["m:timestamp_end"].max()

        plt.figure()

        plt.subplot(2, 1, 1)
        plt.scatter(results["m:timestamp_end"], results.objective)
        plt.plot(results["m:timestamp_end"], results.objective.cummax())
        plt.xlabel("Time (sec.)")
        plt.ylabel("Objective")
        plt.grid()
        plt.xlim(0, tmax)

        plt.subplot(2, 1, 2)
        x, y = compile_profile(results)
        y = np.asarray(y) / num_workers * 100

        plt.step(
            x,
            y,
            where="pre",
        )
        plt.ylim(0, 100)
        plt.xlim(0, tmax)
        plt.xlabel("Time (sec.)")
        plt.ylabel("Worker Utilization (\%)")
        plt.tight_layout()
        plt.show()



.. image-sg:: /examples/images/sphx_glr_plot_profile_worker_utilization_001.png
   :alt: plot profile worker utilization
   :srcset: /examples/images/sphx_glr_plot_profile_worker_utilization_001.png
   :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 22.543 seconds)


.. _sphx_glr_download_examples_plot_profile_worker_utilization.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example




    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_profile_worker_utilization.py <plot_profile_worker_utilization.py>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_profile_worker_utilization.ipynb <plot_profile_worker_utilization.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
