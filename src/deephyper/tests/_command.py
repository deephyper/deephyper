import subprocess
import sys


def run(command: str, live_output: bool = False, timeout=None):
    """Test command line interface.

    Args:
        command (str):
            The command line as a string.
        live_output (bool, Optional):
            Boolean that indicates if the STDOUT/STDERR streams from the
            launched subprocess should directly be redirected to the streams
            of the parent process.
        timeout (float): a timeout for the called command.
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
                timeout=timeout,
            )
        else:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        return result
    except subprocess.CalledProcessError as e:
        print(e.stdout, file=sys.stdout)
        print(e.stderr, file=sys.stderr)
        raise e
    except subprocess.TimeoutExpired as e:
        print(e.stdout, file=sys.stdout)
        print(e.stderr, file=sys.stderr)
        raise e
