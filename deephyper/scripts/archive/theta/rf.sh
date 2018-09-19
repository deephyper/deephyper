#!/bin/bash -x
#COBALT -A datascience
#COBALT -n $nodes
#COBALT -q $queue
#COBALT -t $time_min
#COBALT --attrs ssds=required:ssd_size=128

NUM_WORKERS=$$(( $nodes - 2 ))

DEEPHYPER_TOP=$DEEPHYPER_TOP
DATABASE_TOP=$DATABASE_TOP
BALSAM_PATH=$BALSAM_PATH

WALLMINUTES=$$(( $time_minutes - 10 ))
MAXEVALS=$max_evals
STAGE_IN_DIR=$STAGE_IN_DIR
source ~/.bash_profile
source activate $DEEPHYPER_ENV_NAME

JOBNAME=$benchmark_$nodes_rf
DBPATH=$DATABASE_TOP/$$JOBNAME

balsam init $$DBPATH
export BALSAM_DB_PATH=$$DBPATH

balsam rm apps --all --force
balsam rm jobs --all --force

SEARCH_APP_PATH=$DEEPHYPER_TOP/search/amls.py
ARGS="--max-evals $max_evals --benchmark $benchmark --num-workers $$NUM_WORKERS --learner RF --stage_in_destination=$STAGE_IN_DIR"
balsam app --name search --desc 'run AMLS: RF' --executable $SEARCH_APP_PATH
