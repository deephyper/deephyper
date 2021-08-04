import sys
import os
import importlib
import traceback

from deephyper.core.exceptions.loading import GenericLoaderError

bcolors = {
    "HEADER": "\033[95m",
    "OKBLUE": "\033[94m",
    "OKGREEN": "\033[92m",
    "WARNING": "\033[93m",
    "FAIL": "\033[91m",
    "ENDC": "\033[0m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m",
}


def banner(message, color="HEADER"):
    """Print a banner with message

    Args:
        message (str): The message to be printed
        color (str, optional): The color of the banner in bcolors. Defaults to "HEADER".
    """

    header = "*" * (len(message) + 4)
    msg = f" {header}\n   {message}\n {header}"
    if sys.stdout.isatty():
        print(bcolors.get(color), msg, bcolors["ENDC"], sep="")
    else:
        print(msg)


def load_attr(str_full_module):
    """
    Args:
        - str_full_module: (str) correspond to {module_name}.{attr}
    Return: the loaded attribute from a module.
    """
    if type(str_full_module) == str:
        split_full = str_full_module.split(".")
        str_module = ".".join(split_full[:-1])
        str_attr = split_full[-1]
        module = importlib.import_module(str_module)
        return getattr(module, str_attr)
    else:
        return str_full_module

def load_from_script(fname, attr):
    dirname, basename = os.path.split(fname)
    sys.path.insert(0, dirname)
    module_name = os.path.splitext(basename)[0]
    module = importlib.import_module(module_name)
    return getattr(module, attr)

def generic_loader(target: str, attribute=None):
    """Load attribute from target module

    Args:
        target (str or Object): either path to python file, or dotted Python package name.
        attribute (str): name of the attribute to load from the target module.

    Raises:
        GenericLoaderError: Raised when the generic_loader function is failing.

    Returns:
        Object: the loaded attribute.
    """
    if not isinstance(target, str):
        return target

    if os.path.isfile(os.path.abspath(target)):
        target_file = os.path.abspath(target)
        try:
            attr = load_from_script(target_file, attribute)
        except:
            trace_source = traceback.format_exc()
            raise GenericLoaderError(target, attribute, trace_source)
    else:
        try:
            attr = load_attr(target)
        except:
            attribute = target.split(".")[-1]
            trace_source = traceback.format_exc()
            raise GenericLoaderError(target, attribute, trace_source)

    return attr