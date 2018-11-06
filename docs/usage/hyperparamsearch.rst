Hyper Parameter Search
**********************

.. image:: ../_static/img/search.png
   :scale: 50 %
   :alt: search
   :align: right

Asynchronous Model-Base Search (AMBS)
=====================================

You can download deephyper paper :download:`here <../downloads/deephyper_final.pdf>`

Environment variable of the search : ``DH_AMBS``

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
