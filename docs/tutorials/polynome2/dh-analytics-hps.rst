
Deephyper analytics - hyperparameter search study
=================================================

**path to data file**: /Users/romainegele/polynome2/results.csv

for customization please see:
https://matplotlib.org/api/matplotlib\_configuration\_api.html

Setup & Data loading
--------------------

.. code:: ipython3

    path_to_data_file = '/Users/romainegele/polynome2/results.csv'

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

    width = 15
    height = 10

    matplotlib.rcParams.update({
        'font.size': 22,
        'figure.figsize': (width, height),
        'figure.facecolor': 'white',
        'savefig.dpi': 72,
        'figure.subplot.bottom': 0.125,
        'figure.edgecolor': 'white',
        'xtick.labelsize': 20,
        'ytick.labelsize': 20})

    df = pd.read_csv(path_to_data_file)

    display(Markdown(f'The search did _{df.count()[0]}_ **evaluations**.'))

    df.head()



The search did *15* **evaluations**.




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
          <th>num_units</th>
          <th>objective</th>
          <th>elapsed_sec</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>10</td>
          <td>0.000225</td>
          <td>8.074761</td>
        </tr>
        <tr>
          <th>1</th>
          <td>61</td>
          <td>0.981812</td>
          <td>85.334238</td>
        </tr>
        <tr>
          <th>2</th>
          <td>43</td>
          <td>0.961050</td>
          <td>159.153082</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>-2.439434</td>
          <td>172.763155</td>
        </tr>
        <tr>
          <th>4</th>
          <td>35</td>
          <td>0.891610</td>
          <td>241.104031</td>
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
          <th>num_units</th>
          <th>objective</th>
          <th>elapsed_sec</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>count</th>
          <td>15.00000</td>
          <td>15.000000</td>
          <td>15.000000</td>
        </tr>
        <tr>
          <th>mean</th>
          <td>65.00000</td>
          <td>0.682333</td>
          <td>451.246124</td>
        </tr>
        <tr>
          <th>std</th>
          <td>31.59792</td>
          <td>0.899678</td>
          <td>293.341006</td>
        </tr>
        <tr>
          <th>min</th>
          <td>1.00000</td>
          <td>-2.439434</td>
          <td>8.074761</td>
        </tr>
        <tr>
          <th>25%</th>
          <td>46.00000</td>
          <td>0.968106</td>
          <td>206.933593</td>
        </tr>
        <tr>
          <th>50%</th>
          <td>82.00000</td>
          <td>0.984586</td>
          <td>423.426936</td>
        </tr>
        <tr>
          <th>75%</th>
          <td>90.50000</td>
          <td>0.985981</td>
          <td>686.575820</td>
        </tr>
        <tr>
          <th>max</th>
          <td>95.00000</td>
          <td>0.986564</td>
          <td>924.619735</td>
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


.. parsed-literal::

    /anaconda3/lib/python3.7/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval



.. image:: polynome2/output_8_1.png


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

    num_units       90.000000
    objective        0.986564
    elapsed_sec    924.619735
    Name: 14, dtype: float64



.. code:: ipython3

    dict(df.iloc[i_min])




.. parsed-literal::

    {'num_units': 1.0,
     'objective': -2.4394338798522948,
     'elapsed_sec': 172.7631549835205}


