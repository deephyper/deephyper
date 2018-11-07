Hyper Parameter Search
**********************

.. image:: ../_static/img/search.png
   :scale: 50 %
   :alt: search
   :align: center

All searches can be used directly with the command line or inside an other python file.
To print the parameters of a search just run ``python search_script.py --help``. For example with AMBS run ``python ambs.py --help``.

Asynchronous Model-Base Search (AMBS)
=====================================

You can download deephyper paper :download:`here <../downloads/deephyper_final.pdf>`

Environment variable to access the search on Theta: ``DH_AMBS``

Arguments :

* ``learner``

    * ``RF`` : Random Forest (default)
    * ``ET`` :
    * ``GBRT`` :
    * ``DUMMY`` :
    * ``GP`` :

* ``liar-strategy``

    * ``cl_max`` : (default)
    * ``cl_min`` :
    * ``cl_mean`` :

* ``acq-func`` : Acquisition function

    * ``LCB`` :
    * ``EI`` :
    * ``PI`` :
    * ``gp_hedge`` : (default)

.. autoclass:: deephyper.search.AMBS
  :members:
