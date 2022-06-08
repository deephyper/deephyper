import pathlib


def ensure_dh_folder_exists():
    pathlib.Path("~/.deephyper").mkdir(parents=True, exist_ok=True)
