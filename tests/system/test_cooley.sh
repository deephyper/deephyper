#!/bin/bash +x

DB_NAME=cooleydb
PROJECT_NAME=datascience

balsam init $DB_NAME
source balsamactivate $DB_NAME
balsam app --name AMBS --exec "$(which python) -m deephyper.search.hps.ambs"

deephyper balsam-submit hps theta_test \
    -p deephyper.benchmark.hps.polynome2.Problem \
    -r deephyper.benchmark.hps.polynome2.run \
    -t 15 -q debug -n 2 -A $PROJECT_NAME -j serial --num-evals-per-node 2