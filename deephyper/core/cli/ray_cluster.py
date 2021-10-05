import os
import socket
import getpass
import stat
import sys
import ipaddress
from pprint import pformat

from jinja2 import Template

from deephyper.core.cli.nodelist import expand_nodelist

MODULE_PATH = os.path.dirname(os.path.abspath(__file__))


def add_subparser(subparsers):
    subparser_name = "ray-cluster"
    function_to_call = main

    subparser_ray_cluster = subparsers.add_parser(
        subparser_name, help="Manipulate a Ray cluster."
    )
    subparsers_ray_cluster = subparser_ray_cluster.add_subparsers()

    # config
    subparser_config = subparsers_ray_cluster.add_parser(
        "config",
        help="Generate a Ray cluster YAML configuration based on the current context.",
    )
    subparser_config.add_argument("--head-node-ip")
    subparser_config.add_argument("--worker-nodes-ips", nargs="*", default=[])
    subparser_config.add_argument("--num-cpus", type=int)
    subparser_config.add_argument("--num-gpus", type=int, default=None)
    subparser_config.add_argument(
        "-o",
        "--output",
        default="ray-config.yaml",
        required=False,
        help="Name of the YAML configuration file created.",
    )
    subparser_config.add_argument(
        "--init",
        required=True,
        help="Initialization script source before starting the Ray servers.",
    )
    subparser_config.add_argument("-v", "--verbose", action="store_true")

    subparser_ray_cluster.set_defaults(func=function_to_call)


def main(**kwargs):

    if sys.argv[2] == "config":
        do_config(**kwargs)


def validate_ip(ip):
    """Check if the input is a valid ip address. If it is not it will assume being a valid domain and try to retrive a valid ip from it.

    Args:
        ip (str): a valid ip address or a valid domain.

    Returns:
        str: a valid ip address
    """
    try:
        ipaddress.ip_address(ip)
        return ip
    except ValueError:
        return socket.gethostbyname(ip)

def do_config(head_node_ip, worker_nodes_ips, num_cpus, num_gpus, output, init, verbose, **kwargs):

    # check if output has ".yaml"
    if output[-5:] != ".yaml":
        output += ".yaml"

    # Detection of the host
    config = {
        "init_script": os.path.abspath(init),
        "head_node_ip": head_node_ip,
        "worker_nodes_ips": worker_nodes_ips,
        "username": getpass.getuser(),
    }
    config["num_workers"] = len(config["worker_nodes_ips"])

    # managed requested ressources for each node
    ressources = f"--num-cpus {num_cpus}"
    if num_gpus is not None:
        ressources += f" --num-gpus {num_gpus}"
    config["ressources"] = ressources

    if verbose:
        print("Configuration will be created with: ")
        print(pformat(config))

    # Load configuration template
    ray_config_template_path = os.path.join(
        MODULE_PATH, "job-templates-ray", "ray-config.yaml.tmpl"
    )
    with open(ray_config_template_path, "r") as f:
        template = Template(f.read())

    with open(output, "w") as fp:
        fp.write(template.render(**config))

    if verbose:
        print("Created Ray configuration: ", fp.name)