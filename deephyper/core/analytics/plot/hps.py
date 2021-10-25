"""
The hyperparameter search study is designed to get some visualisation and results processing after running an hyperparameter search. It will provide you an overview of evaluations done during the search as well as a search trajectory with respect to your objective value. Finally you will find the best set of hyperparameters found by the search.

An example use of this command line can be as follow::

    $ deephyper-analytics notebook --type hps --output mynotebook.ipynb ../../database/testdb/data/TEST/test_de03094c/results.csv
"""

import os
from deephyper.core.plot.jn_loader import NbEdit

HERE = os.path.dirname(os.path.abspath(__file__))


def hps_analytics(path_to_data_file, output_file):
    path_to_data_file = os.path.abspath(path_to_data_file)
    editor = NbEdit(os.path.join(HERE, 'stub/hps_analytics.ipynb'),
                    path_to_save=output_file)

    try:
        venv_name = os.environ.get('VIRTUAL_ENV').split('/')[-1]
        editor.setkernel(venv_name)
    except Exception:
        pass

    editor.edit(0, "{{path_to_data_file}}", path_to_data_file)

    editor.edit(1, "{{path_to_data_file}}", f"'{path_to_data_file}'")

    editor.write()

    editor.execute()
