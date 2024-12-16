import pathlib


def ensure_dh_folder_exists():
    """Creates a ``".deephyper"`` directory in the user home directory."""
    home = pathlib.Path.home()
    deephyper_dir = home.joinpath(".deephyper")
    deephyper_dir.mkdir(parents=False, exist_ok=True)
    return deephyper_dir.as_posix()
