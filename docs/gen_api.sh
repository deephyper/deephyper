#!/bin/bash

# list of subpackages to ignore for the API
ignoreDoc="benchmark baseline contrib layers"

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
sphinx-apidoc -o api/ ../deephyper --templatedir templates/api --module-first -e

# delete some subpackages
for package in $ignoreDoc
do
    rm api/deephyper.$package*
done