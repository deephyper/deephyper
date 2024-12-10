import subprocess
import sys


def run(command, live_output=False):
    """Test command line interface.

    Args:
        command (str): the command line as a str.
    """
    command = command.split()
    try:
        if live_output:
            result = subprocess.run(
                command,
                check=True,
                capture_output=False,
                text=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
        else:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(e.stdout)
        print(e.stderr)
        raise e
