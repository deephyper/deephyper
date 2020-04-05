import argparse
import os
import glob
import sys
from jinja2 import Template

def add_subparser(subparsers):
    subparser_name = 'new-problem'
    function_to_call = main

    subparser = subparsers.add_parser(
        subparser_name, help='Tool to init an hyper-parameter search package or an hyper-parameter search problem folder.')

    subparser.add_argument('mode', type=str, choices=['nas', 'hps'], help='NAS or HPS problem')
    subparser.add_argument('name', type=str, help='Name of the problem directory to create')
    subparser.set_defaults(func=function_to_call)

def main(mode, name, *args, **kwargs):
    prob_name = name
    current_path = os.getcwd()
    project_path = os.path.dirname(current_path)
    assert os.path.exists(os.path.join(project_path, 'setup.py')), 'No setup.py in current directory'
    assert os.path.exists(os.path.join(project_path, '.deephyper')), 'Not inside a deephyper project directory'
    assert '/' not in prob_name, f'Problem name must not contain "/"'
    assert prob_name.isidentifier(), f'{prob_name} is not a valid Python identifier'

    os.mkdir(prob_name)
    with open(os.path.join(prob_name, '__init__.py'), 'w') as fp:
        pass
    render_files(mode, prob_name)

def render_files(mode, prob_name):
    package = os.path.basename(os.getcwd())
    print("package: ", package)
    templates_pattern = os.path.join(
        os.path.dirname(__file__),
        'templates',
        mode,
        '*.tmpl'
    )
    for template_name in glob.glob(templates_pattern):
        template = Template(open(template_name).read())
        py_name = os.path.basename(
            template_name.rstrip('.tmpl')
        )
        with open(os.path.join(prob_name, py_name), 'w') as fp:
            fp.write(
                template.render(
                    pckg=package,
                    pb_folder=prob_name,
                )
            )
            print("Created", fp.name)