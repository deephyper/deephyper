import logging
import asyncio
import time
import sys
import inspect
import re
import json

from deephyper.evaluator.async_evaluate import AsyncEvaluator

logger = logging.getLogger(__name__)


class SubprocessEvaluator(AsyncEvaluator):

    def __init__(self, run_function, method, num_workers=1):
        super().__init__(run_function, method, num_workers)
        self.sem = asyncio.Semaphore(num_workers)
        logger.info(
            f"Subprocess Evaluator will execute {self.run_function.__name__}() from module {self.run_function.__module__}"
        )

    async def execute(self, job):
        async with self.sem:
            job.status = job.RUNNING
            start_time = time.time()

            # Retrieve the path of the module holding the user-defined function given to the async evaluator.
            # Pass this module path to the subprocess to add to its python path and use.
            module_path = inspect.getfile(sys.modules[self.run_function.__module__])
            # Code that will run on the subprocess.
            code = f"import sys; sys.path.append('{module_path}');from {self.run_function.__module__} import {self.run_function.__name__}; print('DH-OUTPUT:' + str({self.run_function.__name__}({job.config})))"
            proc = await asyncio.create_subprocess_exec(
                sys.executable, '-c', code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE)
            
            # Retrieve the stdout byte array from the (stdout, stderr) tuple returned from the subprocess.
            byte_arr = (await proc.communicate())[0]

            # Search through the byte array using a regular expression and collect the return value of the user-defined function. 
            retval_bytes = re.search(b'DH-OUTPUT:(.+)\n', byte_arr).group(1)
            # Finally, parse whether the return value from the user-defined function is a scalar, a list, or a dictionary.
            retval = retval_bytes.replace(b"\'", b"\"") # For dictionaries, replace single quotes with double quotes!
            sol = json.loads(retval)

            await proc.wait()
            
            job.duration = time.time() - start_time
            job.status = job.DONE
            job.result = (job.config, sol)

        return job