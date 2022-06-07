import logging
import asyncio
import sys
import inspect
import re
import json
import os

from deephyper.evaluator._evaluator import Evaluator
from deephyper.evaluator._encoder import Encoder

logger = logging.getLogger(__name__)


def encode_dict(d: dict):
    return json.loads(json.dumps(d, cls=Encoder))


class SubprocessEvaluator(Evaluator):
    """This evaluator uses the ``asyncio.create_subprocess_exec`` as backend.

    Args:
        run_function (callable): functions to be executed by the ``Evaluator``.
        num_workers (int, optional): Number of parallel processes used to compute the ``run_function``. Defaults to 1.
        callbacks (list, optional): A list of callbacks to trigger custom actions at the creation or completion of jobs. Defaults to None.
    """

    def __init__(
        self,
        run_function,
        num_workers: int = 1,
        callbacks: list = None,
        run_function_kwargs: dict = None,
    ):
        super().__init__(run_function, num_workers, callbacks, run_function_kwargs)
        self.sem = asyncio.Semaphore(num_workers)

        if hasattr(run_function, "__name__") and hasattr(run_function, "__module__"):
            logger.info(
                f"Subprocess Evaluator will execute {self.run_function.__name__}() from module {self.run_function.__module__}"
            )
        else:
            logger.info(f"Subprocess Evaluator will execute {self.run_function}")

    def _encode(self, job):
        return encode_dict(job.config)

    async def execute(self, job):
        async with self.sem:

            # Retrieve the path of the module holding the user-defined function given to the async evaluator.
            # Pass this module path to the subprocess to add to its python path and use.
            script_file = inspect.getfile(sys.modules[self.run_function.__module__])
            module_path = os.path.dirname(script_file)
            module_name = os.path.basename(script_file)[:-3]
            # Code that will run on the subprocess.
            code = f"import sys; sys.path.insert(1, '{module_path}'); from {module_name} import {self.run_function.__name__}; print('DH-OUTPUT:' + str({self.run_function.__name__}({self._encode(job)}, **{encode_dict(self.run_function_kwargs)})))"
            logger.debug(f"executing:  {code}")
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            # Retrieve the stdout byte array from the (stdout, stderr) tuple returned from the subprocess.
            stdout, stderr = await proc.communicate()
            # Search through the byte array using a regular expression and collect the return value of the user-defined function.
            try:
                retval_bytes = re.search(b"DH-OUTPUT:(.+)\n", stdout).group(1)
            except AttributeError:
                error = stderr.decode("utf-8")
                raise RuntimeError(
                    f"{error}\n\n Could not collect any result from the run_function in the main process because an error happened in the subprocess."
                )
            # Finally, parse whether the return value from the user-defined function is a scalar, a list, or a dictionary.
            retval = retval_bytes.replace(
                b"'", b'"'
            )  # For dictionaries, replace single quotes with double quotes!
            sol = json.loads(retval)

            await proc.wait()

            job.result = sol

        return job
