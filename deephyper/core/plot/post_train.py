"""Deephyper analytics - post-training study

usage:

::

    $ deephyper-analytics post mybalsamdb/data/workflow_folder

"""

import os
import argparse
from deephyper.core.plot.jn_loader import NbEdit

HERE = os.path.dirname(os.path.abspath(__file__))


def post_train_analytics(path_to_data_folder):
    editor = NbEdit(os.path.join(HERE, 'stub/post_train.ipynb'), path_to_save="dh-analytics-post.ipynb")

    venv_name = os.environ.get('VIRTUAL_ENV').split('/')[-1]
    editor.setkernel(venv_name)

    editor.edit(0, "{{path_to_data_folder}}", path_to_data_folder)

    editor.edit(1, "{{path_to_data_folder}}", f"'{path_to_data_folder}'")

    editor.write()

    editor.execute()


def add_subparser(subparsers):
    subparser_name = 'post'
    function_to_call = main

    parser_parse = subparsers.add_parser(
        subparser_name, help='Tool to generate analytics from a post-training experiment (jupyter notebook).')
    parser_parse.add_argument(
        'path', type=str, help=f'Path to the workflow folder of the experiment. The workflow folder is located in "database/data/workflow_folders", it is the folder containing all task folders.')

    return subparser_name, function_to_call


def main(path, *args, **kwargs):
    post_train_analytics(path_to_data_folder=path)
