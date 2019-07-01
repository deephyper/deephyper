#!/bin/bash

export PYTESTS_DIR=$PWD/tests
export DOCTESTS_DIR=$PWD/docs

cd $PYTESTS_DIR
coverage run -m pytest

cd $DOCTESTS_DIR
make doctest