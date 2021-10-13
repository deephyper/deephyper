#!/bin/bash

ignoreDoc="benchmark baseline contrib layers"

if [ ! -d "api/" ]
then
    mkdir api/
else
    rm api/*.rst
fi

sphinx-apidoc -o api/ ../deephyper --templatedir templates/api --module-first

for package in $ignoreDoc
do
    rm api/deephyper.$package*
done