#!/bin/bash

# delete the api/ folder if exists
if [ ! -d "api/" ]
then
    mkdir api/
else
    rm api/*.rst
fi

# settings for automodule options
export SPHINX_APIDOC_OPTIONS="members,undoc-members,show-inheritance,inherited-members"

# generate the API documentation
export OUTPUT_PATH=api/
export MODULE_PATH=../deephyper
sphinx-apidoc --templatedir templates/api --module-first -e \
    -o $OUTPUT_PATH \
    $MODULE_PATH \
    ../deephyper/benchmark \
    ../deephyper/contrib \
    ../deephyper/baseline \
    ../deephyper/layers \
    ../deephyper/core/logs \
    ../deephyper/core/exceptions \
    ../deephyper/core/plot/hps* \
    ../deephyper/core/plot/jn_loader* \
    ../deephyper/core/plot/multi* \
    ../deephyper/core/plot/nas_single* \
    ../deephyper/core/plot/post_train* \
    ../deephyper/core/parser*


rm api/deephyper.rst
rm api/modules.rst