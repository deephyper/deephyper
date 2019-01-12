<p align="center">
![Alt text](docs/images/deephyper.png?raw=true "DeepHyper")
</p>

# What is DeepHyper?

DeepHyper is a Python package and infrastructure that targets
experimental research in DL search methods, scalability, and
portability across HPC systems. It comprises three modules:
benchmarks, a collection of extensible and diverse DL hyperparameter search problems; search, a set
of search algorithms for DL hyperparameter search; and
evaluators, a common interface for evaluating hyperparameter
configurations on HPC platforms.

# Documentation

Deephyper documentation is on : [ReadTheDocs](https://deephyper.readthedocs.io)

# Directory structure

```
benchmarks/
    directory for problems
experiments/
    directory for saving the running the experiments and storing the results
search/
    directory for search applications
    hps/
        hyperparameter search applications
    nas/
        neural architecture search applications
```

# Install instructions

```
cd deephyper
pip install -e .
```
# How do I learn more?

* Documentation: https://deephyper.readthedocs.io

* GitHub repository: https://github.com/deephyper/deephyper

# Who is responsible?

The core DeepHyper team is at Argonne National Laboratory:

* Prasanna Balaprakash <pbalapra@anl.gov>, Lead and founder
* Romain Egele <regele@anl.gov>
* Misha Salim <msalim@anl.gov>
* Venkat Vishwanath <venkat@anl.gov>
* Stefan Wild <wild@anl.gov>

Modules, patches (code, documentation, etc.) contributed by:

* Elise Jennings <ejennings@anl.gov>
* Dipendra Kumar Jha <dipendrajha2018@u.northwestern.edu>

# How can I participate?

Questions, comments, feature requests, bug reports, etc. can be directed to:

* Our mailing list: *deephyper@groups.io* or https://groups.io/g/deephyper

* Issues on GitHub

Patches are much appreciated on the software itself as well as documentation.
Optionally, please include in your first patch a credit for yourself in the
list above.

# Acknowledgements

* Scalable Data-Efficient Learning for Scientific Domains, U.S. Department of Energy 2018 Early Career Award funded by the Advanced Scientific Computing Research program within the DOE Office of Science (2018--Present)
* Argonne Leadership Computing Facility (2018--Present)
* SLIK-D: Scalable Machine Learning Infrastructures for Knowledge Discovery, Argonne Computing, Environment and Life Sciences (CELS) Laboratory Directed Research and Development (LDRD) Program (2016--2018)

# Copyright and license

TBD

