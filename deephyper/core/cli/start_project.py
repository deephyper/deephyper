import argparse
import os
import sys

def add_subparser(subparsers):
    subparser_name = 'startproject'
    function_to_call = main

    subparser = subparsers.add_parser(
        subparser_name, help='Set up a new project folder for DeepHyper benchmarks')

    subparser.add_argument('path', type=str, help='Path to the new project directory')
    subparser.set_defaults(func=function_to_call)

def main(path, *args, **kwargs):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'setup.py'), 'w') as fp:
        fp.write(f"from setuptools import setup, find_packages\n\nsetup(\n    name='{os.path.basename(path)}',\n    packages=find_packages(),\n    install_requires=[]\n)")
    with open(os.path.join(path, '__init__.py'), 'w') as fp:
        pass
    with open(os.path.join(path, '.deephyper'), 'w') as fp:
        pass