#!/bin/bash -x
#COBALT -A datascience
#COBALT -n 4
#COBALT -q debug
#COBALT -t 00:30:00

# User-specific paths and names go here (NO TRAILING SLASHES):
DEEPHYPER_ENV_NAME="deephyper-cooley"
DEEPHYPER_TOP=/home/msalim/workflows/deephyper
DATABASE_TOP=/projects/datascience/msalim/deephyper/database-cooley
BALSAM_PATH=/home/msalim/hpc-edge-service

# Set Hyperband options
WALLMINUTES=20   # should be about 10 min less than COBALT requested time
SAVED_MODEL_DIR=/projects/datascience/msalim/deephyper/saved_models # shared directory accessible by all compute nodes
STAGE_IN_DIR=""

# DO NOT CHANGE ANYTHING BELOW HERE:
# ----------------------------------
if [ $# -ne 1 ] 
then
    echo "Please provide one argument: benchmark_name (e.g. dummy2.regression or b2.babi_memnn)"
    exit 1
fi

source /soft/libraries/anaconda/bin/activate $DEEPHYPER_ENV_NAME

# Job naming
BNAME=$1
METHOD="hyperband"

NNODES=`cat $COBALT_NODEFILE | wc -l`
COBALT_JOBSIZE=$NNODES
NUM_WORKERS=$(( 2*$NNODES - 2 ))
JOBNAME="$BNAME"_"$NNODES"_"$METHOD"
WORKFLOWNAME="$BNAME"_"$NNODES"_"$METHOD"
echo "Creating new job:" $JOBNAME

# Set up Balsam DB: ensure it is clear
# ------------------------------------
DBPATH=$DATABASE_TOP/$WORKFLOWNAME
balsam init $DBPATH
export BALSAM_DB_PATH=$DBPATH

balsam rm apps --all --force
balsam rm jobs --all --force

# Register search app
#---------------------
SEARCH_APP_PATH=$DEEPHYPER_TOP/search/hyperband.py
ARGS="--benchmark $BNAME --num-workers $NUM_WORKERS --model_path=$SAVED_MODEL_DIR --stage_in_destination=$STAGE_IN_DIR"
balsam app --name search --desc 'run Hyperband' --executable $SEARCH_APP_PATH

# Create job and mark as PREPROCESSED so it can be picked up immediately by mpi_ensemble module of Balsam
# -------------------------------------------------------------------------------------------------------------
balsam job --name $JOBNAME --workflow $WORKFLOWNAME --application search --wall-minutes $WALLMINUTES  --num-nodes 1 --ranks-per-node 1 --args "$ARGS" --yes --threads-per-rank 64 --threads-per-core 1

NEW_ID=$(balsam ls | grep CREATED | awk '{print $1}' | cut -d '-' -f 1)
balsam modify jobs $NEW_ID --attr state --value PREPROCESSED

# Start up Balsam DB server
# -------------------------
balsam dbserver --reset $DBPATH
balsam dbserver
sleep 1
mpirun -f $COBALT_NODEFILE -n $(( $COBALT_JOBSIZE*2 )) --ppn 2 python $BALSAM_PATH/balsam/launcher/mpi_ensemble_pull.py --time-limit-min=$(( $WALLMINUTES+5 )) --gpus-per-node=2
balsam dbserver --stop
