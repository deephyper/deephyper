import sys
import time

from balsam.launcher.dag import BalsamJob
from balsam.core.models import END_STATES

class EvalFailed(Exception): pass
class TimeoutError(Exception): pass

class BalsamApplyResult:

    def __init__(self, pk):
        self.pk = pk
        self.result = None
        self._state = None
    
    def _read(self):
        job = BalsamJob.objects.get(pk=self.pk)
        output = job.read_file_in_workdir(f'{job.name}.out')
        y = None
        for line in output.split('\n'):
            if "OUTPUT:" in line.upper():
                y = float(line.split()[-1])
                break
        if y is None: 
            raise EvalFailed(f'Never found OUTPUT line for job {job.cute_id}')
        else:
            return y

    def _poll(self, timeout, period=1.0):
        if self._state is not None:
            return self._state

        if timeout is None:
            timeout = sys.float_info.max
        else:
            timeout = float(timeout)
            timeout = max(timeout, 0.1)

        start = time.time()
        elapsed = 0.0
        while elapsed < timeout:
            elapsed = time.time() - start
            _job = BalsamJob.objects.get(pk=self.pk)
            if _job.state in END_STATES:
                self._state = _job.state
                return self._state
            time.sleep(period)
        raise TimeoutError

    def get(self, timeout=None):
        if self.result is not None:
            return self.result
        state = self._poll(timeout)
        if state in ['RUN_DONE', 'JOB_FINISHED']:
            self.result = self._read()
            return self.result
        elif state in ['RUN_ERROR', 'FAILED']:
            raise EvalFailed(f'BalsamJob {self.pk} in failed state')

    def ready(self):
        try:
            state = self._poll(0)
        except TimeoutError:
            return False
        else:
            return True
