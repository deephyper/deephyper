#!/bin/bash

export PYTESTS_DIR=$PWD/tests
export DOCTESTS_DIR=$PWD/docs

cd $DOCTESTS_DIR
make doctest

cd $PYTESTS_DIR
pytest --cov=./