from io import StringIO
import sys


class CaptureSTD(list):
    """Capture the stdout and stderr of a block of code."""

    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self._stringio_stdout = StringIO()
        sys.stderr = self._stringio_stderr = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(
            (
                self._stringio_stdout.getvalue().splitlines(),
                self._stringio_stdout.getvalue().splitlines(),
            )
        )
        del self._stringio_stdout  # free up some memory
        del self._stringio_stderr  # free up some memory
        sys.stdout = self._stdout
        sys.stderr = self._stderr
