#!/bin/bash -x
#COBALT -A datascience
#COBALT -n 128
#COBALT -q default
#COBALT -t 02:00:00
#COBALT --attrs ssds=required:ssd_size=128

source ~/.bash_profile
source activate balsam

BNAME=$1
NNODES=$2
METHOD=$3


JOBNAME=$BNAME_$NNODE_$METHOD
WORKFLOWNAME=$BNAME_$NNODE_$METHOD
WALLMINUTES=110
MAXEVALS=100000000
BENCHMARK=$BNAME
SEARCH_APP_PATH=/projects/datascience/pbalapra/deephyper/deephyper/search/$METHOD.py
NUM_WORKERS=$(( $COBALT_JOBSIZE-1 ))

balsam init /projects/datascience/pbalapra/deephyper/database/$WORKFLOWNAME
export BALSAM_DB_PATH=/projects/datascience/pbalapra/deephyper/database/$WORKFLOWNAME

balsam rm apps --all --force
balsam app --name search --desc 'run search' --executable $SEARCH_APP_PATH
balsam rm jobs --all --force


ARGS="--max-evals $MAXEVALS --benchmark $BENCHMARK --num-workers $NUM_WORKERS"

balsam job --name $JOBNAME --workflow $WORKFLOWNAME --application search --wall-minutes $WALLMINUTES  --num-nodes 1 --ranks-per-node 1 --args "$ARGS" --yes --threads-per-rank 64 --threads-per-core 1

balsam dbserver --reset /projects/datascience/pbalapra/deephyper/hpc-edge-service/default_balsamdb/
balsam dbserver
sleep 1
balsam launcher --wf-name $WORKFLOWNAME --max-ranks-per-node 1

