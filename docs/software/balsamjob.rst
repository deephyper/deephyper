.. _balsamjob_spec:

Configuring model execution with Balsam
*******************************************

When using the Balsam evaluator or ``balsam-submit`` shortcut to
launch DeepHyper HPS jobs, the default assumption is that the
``--run`` argument is a callable that trains your model and returns the
objective of maximization.
This is a convenient abstraction for many simple use cases, because 
DeepHyper fully automates: 

    * passing hyperparameters from DeepHyper to your model
    * returning the objective from a trained model to DeepHyper
    * wrapping your model in an executable "runner" script

Unfortunately, there are cases where the default execution model does
not apply. For instance, you may wish to launch a Singularity container
that performs data-parallel model training on several nodes.  You may
also need to vary the number of MPI ranks according to a local batch-size hyperparmeter
in your search.  In another scenario, you might have a model implemented
in Fortran that isn't trivially importable by DeepHyper.

The underlying Balsam evaluator is sufficiently flexible to handle these
complex use cases. The price to pay for this flexibility is that your
code becomes responsible for addressing the preceeding bullet points.

We illustrate how to control Balsam model evaluation tasks below.

Creating an execution wrapper
==============================

Decorating the run function
============================

Parsing the objective
======================