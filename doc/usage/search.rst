Search
******

Hyper Parameter Search
======================

Asynchronous Model-Base Search (AMBS)
-------------------------------------

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

Neural Architecture Search
==========================

Coming soon...
