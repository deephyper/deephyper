#!/bin/bash -x
#COBALT -A {{ project }}
#COBALT -n {{ nodes }}
#COBALT -q {{ queue }}
#COBALT -t {{ time_minutes }}

mkdir infos && cd infos

ACTIVATE_PYTHON_ENV="{{ activation_script }}"
echo "Script to activate Python env: $ACTIVATE_PYTHON_ENV"
source $ACTIVATE_PYTHON_ENV

{{ script_launch_ray_cluster }}

deephyper {{ mode }} {{ search }} --evaluator {{ evaluator }} --ray-address auto \
    --problem {{ problem }} \
    --run {{ run }} \
    --max-evals {{ max_evals }} \
    --num-cpus-per-task {{ num_cpus_per_task }} \
    --num-gpus-per-task {{ num_gpus_per_task }} \
    {{ other_search_arguments }}

ray stop