#!/bin/bash -x
#COBALT -A datascience
#COBALT -n 128
#COBALT -q default
#COBALT -t 02:00:00
#COBALT --attrs ssds=required:ssd_size=128

source ~/.bash_profile
source activate dl-hps

BNAME=$1
NNODES=$2
METHOD=$3

WALLMINUTES=110
MAXEVALS=100000000
BENCHMARK=$BNAME

SEARCH_APP_PATH=/home/msalim/workflows/deephyper/search/$METHOD.py
NUM_WORKERS=$(( $COBALT_JOBSIZE-1 ))

JOBNAME="$BNAME"_"$NNODES"_"$METHOD"
WORKFLOWNAME="$BNAME"_"$NNODES"_"$METHOD"

echo "Creating new job:" $JOBNAME

balsam init /projects/datascience/msalim/deephyper/database/$WORKFLOWNAME
export BALSAM_DB_PATH=/projects/datascience/msalim/deephyper/database/$WORKFLOWNAME

balsam rm apps --all --force
balsam app --name search --desc 'run search' --executable $SEARCH_APP_PATH
balsam rm jobs --all --force

ARGS="--max-evals $MAXEVALS --benchmark $BENCHMARK --num-workers $NUM_WORKERS"

balsam job --name $JOBNAME --workflow $WORKFLOWNAME --application search --wall-minutes $WALLMINUTES  --num-nodes 1 --ranks-per-node 1 --args "$ARGS" --yes --threads-per-rank 64 --threads-per-core 1

NEW_ID=$(balsam ls | grep CREATED | awk '{print $1}' | cut -d '-' -f 1)
balsam modify jobs $NEW_ID --attr state --value PREPROCESSED

balsam dbserver --reset /projects/datascience/msalim/deephyper/database/$WORKFLOWNAME
balsam dbserver
sleep 1
aprun -n $COBALT_JOBSIZE -N 1 -cc none python /home/msalim/hpc-edge-service/balsam/launcher/mpi_ensemble_pull.py
