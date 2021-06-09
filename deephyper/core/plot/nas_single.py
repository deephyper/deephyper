"""Neural architecture search single study.

usage::

    $ deephyper-analytics notebook --type nas --output mynotebook.ipynb logdir/
"""

import os
from deephyper.core.plot.jn_loader import NbEdit

HERE = os.path.dirname(os.path.abspath(__file__))


def single_analytics(path_to_logdir, nb_name):
    editor = NbEdit(os.path.join(HERE, 'stub/nas-single.ipynb'),
    path_to_save=f"{nb_name}.ipynb")

    try:
        venv_name = os.environ.get('VIRTUAL_ENV').split('/')[-1]
        editor.setkernel(venv_name)
    except:
        pass

    editor.edit(0, "{{path_to_logdir}}", path_to_logdir)

    editor.edit(1, "{{path_to_logdir}}", f"'{path_to_logdir}'")

    editor.write()

    editor.execute()
