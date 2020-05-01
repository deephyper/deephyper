#!/bin/bash

python -m deephyper.search.nas.ppo --problem deephyper.benchmark.nas.linearReg.Problem --evaluator subprocess --max-evals 5
