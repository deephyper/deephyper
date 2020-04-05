# Changelog - DeepHyper v0.1.3

## Command line interface for Balsam Job submission

`deephyper balsam-submit` performs all three steps of creating a DeepHyper App,
adding a BalsamJob to run a search, and submitting a launcher job for
execution. The tool checks for existing workflows with a conflicting name, and
ensures that the provided Problem and run functions are importable.

For instance:

```bash
deephyper balsam-submit hps test2 -p foo/foo/problem.py -r foo/foo/run.py -t 60 -q debug-cache-quad -n 4 -A datascience -j mpi
```

## Customizing Balsam execution of model runs

In order to directly control how BalsamJobs are created to evaluate each model in DeepHyper, users can provide a *BalsamJob spec* in place of the actual run function.  The run function is decorated with `deephyper.benchmarks.balsamjob_spec` and must return a BalsamJob.  It is up to the user to ensure that the necessary App is in place and the objective is parsed correctly.

## Updated command line

## Multiple losses available for Neural Architecture Search
