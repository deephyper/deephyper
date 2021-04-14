#!/bin/bash

DB_NAME=thetadb
PROJECT_NAME=datascience

balsam init $DB_NAME
source balsamactivate $DB_NAME
balsam app --name AMBS --exec "$(which python) -m deephyper.search.hps.ambs"

deephyper balsam-submit hps theta_test \
    -p deephyper.benchmark.hps.polynome2.Problem \
    -r deephyper.benchmark.hps.polynome2.run \
    -t 15 -q debug-cache-quad -n 2 -A $PROJECT_NAME -j mpi