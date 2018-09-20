'''
>>> evaluator = create_evaluator_nas(config)
>>> apply_result = evaluator.apply_async(x)
>>> apply_result.get(timeout=None) --> return scalar or raise EvalFailed
>>> apply_result.ready() --> return False or True
'''

from deephyper.evaluators.balsam_async import (BalsamApplyResult,
                                               EvalFailed, TimeoutError)
from deephyper.evaluators.evaluate import (create_evaluator,
                                           create_evaluator_nas)


__all__ = '''
BalsamApplyResult
EvalFailed
TimeoutError
create_evaluator_nas
'''.split()
