"""Utilies for evaluator module."""


def test_ipython_interpretor() -> bool:
    """Test if the current Python interpretor is IPython or not.

    Suggested by
    https://stackoverflow.com/questions/15411967/
    how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    """
    # names of shells/modules using jupyter
    notebooks_shells = ["ZMQInteractiveShell"]
    notebooks_modules = ["google.colab._shell"]

    try:
        shell_name = get_ipython().__class__.__name__  # type: ignore
        shell_module = get_ipython().__class__.__module__  # type: ignore

        if shell_name in notebooks_shells or shell_module in notebooks_modules:
            return True  # Jupyter notebook or qtconsole
        elif shell_name == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)

    except NameError:
        return False  # Probably standard Python interpreter
