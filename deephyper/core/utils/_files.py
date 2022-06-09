import pathlib


def ensure_dh_folder_exists():
    """Creates a ``".deephyper"`` directory in the user home directory."""
    pathlib.Path("~/.deephyper").mkdir(parents=False, exist_ok=True)
