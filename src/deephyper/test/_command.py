import subprocess
import sys


def run(command: str, live_output: bool = False):
    """Test command line interface.

    Args:
        command (str): the command line as a string.
        live_output (bool, Optional): boolean that indicates if the STDOUT/STDERR streams from the launched subprocess should directly be redirected to the streams of the parent process.
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
