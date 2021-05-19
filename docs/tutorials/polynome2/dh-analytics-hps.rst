
Deephyper analytics
===================

We will use the ``deephyper-analytics`` command line tool to investigate the results.

.. note::

  See the :ref:`analytics-local-install` installation instructions of ``deephyper-analytics``.

Run:

.. code-block:: console
    :caption: bash

    deephyper-analytics notebook --type hps --output dh-analytics-hps.ipynb results.csv

Then start ``jupyter``:

.. code-block:: console
    :caption: bash

    jupyter notebook

Open the ``dh-analytics-hps`` notebook and run it:

**path to data file**: polynome2/results.csv

for customization please see:
https://matplotlib.org/api/matplotlib\_configuration\_api.html

Setup & data loading
--------------------

.. code:: ipython3

    path_to_data_file = 'polynome2/results.csv'

.. code:: ipython3

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from pprint import pprint
    from datetime import datetime
    from tqdm import tqdm
    from IPython.display import display, Markdown

    width = 21
    height = 13

    matplotlib.rcParams.update({
        'font.size': 21,
        'figure.figsize': (width, height),
        'figure.facecolor': 'white',
        'savefig.dpi': 72,
        'figure.subplot.bottom': 0.125,
        'figure.edgecolor': 'white',
        'xtick.labelsize': 21,
        'ytick.labelsize': 21})

    df = pd.read_csv(path_to_data_file)

    display(Markdown(f'The search did _{df.count()[0]}_ **evaluations**.'))

    df.head()



The search did *88* **evaluations**.




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>activation</th>
          <th>lr</th>
          <th>units</th>
          <th>objective</th>
          <th>elapsed_sec</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>NaN</td>
          <td>0.010000</td>
          <td>10</td>
          <td>-67.720345</td>
          <td>4.683628</td>
        </tr>
        <tr>
          <th>1</th>
          <td>sigmoid</td>
          <td>0.210479</td>
          <td>78</td>
          <td>-47.973845</td>
          <td>7.850657</td>
        </tr>
        <tr>
          <th>2</th>
          <td>sigmoid</td>
          <td>0.849683</td>
          <td>18</td>
          <td>-7.910984</td>
          <td>11.379633</td>
        </tr>
        <tr>
          <th>3</th>
          <td>tanh</td>
          <td>0.951716</td>
          <td>19</td>
          <td>-2.596602</td>
          <td>16.031375</td>
        </tr>
        <tr>
          <th>4</th>
          <td>sigmoid</td>
          <td>0.898754</td>
          <td>74</td>
          <td>-21.409714</td>
          <td>19.312386</td>
        </tr>
      </tbody>
    </table>
    </div>



Statistical summary
-------------------

.. code:: ipython3

    df.describe()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>lr</th>
          <th>units</th>
          <th>objective</th>
          <th>elapsed_sec</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>count</th>
          <td>100.000000</td>
          <td>100.00000</td>
          <td>100.000000</td>
          <td>100.000000</td>
        </tr>
        <tr>
          <th>mean</th>
          <td>0.861301</td>
          <td>13.12000</td>
          <td>-3.468272</td>
          <td>188.652953</td>
        </tr>
        <tr>
          <th>std</th>
          <td>0.112005</td>
          <td>10.78746</td>
          <td>11.586969</td>
          <td>116.032871</td>
        </tr>
        <tr>
          <th>min</th>
          <td>0.010000</td>
          <td>1.00000</td>
          <td>-74.376173</td>
          <td>4.683628</td>
        </tr>
        <tr>
          <th>25%</th>
          <td>0.861376</td>
          <td>7.75000</td>
          <td>-2.011465</td>
          <td>87.576996</td>
        </tr>
        <tr>
          <th>50%</th>
          <td>0.871134</td>
          <td>11.50000</td>
          <td>-0.092576</td>
          <td>178.604464</td>
        </tr>
        <tr>
          <th>75%</th>
          <td>0.876806</td>
          <td>15.00000</td>
          <td>0.494384</td>
          <td>288.718287</td>
        </tr>
        <tr>
          <th>max</th>
          <td>0.997793</td>
          <td>78.00000</td>
          <td>0.746590</td>
          <td>399.764441</td>
        </tr>
      </tbody>
    </table>
    </div>



Search trajectory
-----------------

.. code:: ipython3

    plt.plot(df.elapsed_sec, df.objective)
    plt.ylabel('Objective')
    plt.xlabel('Time (s.)')
    plt.xlim(0)
    plt.grid()
    plt.show()



.. image:: polynome2/output_6_0.png


Pairplots
---------

.. code:: ipython3

    not_include = ['elapsed_sec']
    sns.pairplot(df.loc[:, filter(lambda n: n not in not_include, df.columns)],
                    diag_kind="kde", markers="+",
                    plot_kws=dict(s=50, edgecolor="b", linewidth=1),
                    diag_kws=dict(shade=True))
    plt.show()



.. image:: polynome2/output_8_0.png


.. code:: ipython3

    corr = df.loc[:, filter(lambda n: n not in not_include, df.columns)].corr()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True))
    plt.show()



.. image:: polynome2/output_9_0.png


Best objective
--------------

.. code:: ipython3

    i_max = df.objective.idxmax()
    df.iloc[i_max]




.. parsed-literal::

    activation         relu
    lr             0.882041
    units                21
    objective       0.74659
    elapsed_sec     394.818
    Name: 98, dtype: object



.. code:: ipython3

    dict(df.iloc[i_max])




.. parsed-literal::

    {'activation': 'relu',
     'lr': 0.8820413612862609,
     'units': 21,
     'objective': 0.7465898108482361,
     'elapsed_sec': 394.81818103790283}


