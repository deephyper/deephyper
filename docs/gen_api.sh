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

# generate the API documentation
sphinx-apidoc -o api/ ../deephyper --templatedir templates/api --module-first

# delete some subpackages
for package in $ignoreDoc
do
    rm api/deephyper.$package*
done