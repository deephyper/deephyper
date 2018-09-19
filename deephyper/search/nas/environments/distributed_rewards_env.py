
import time
from multiprocessing.pool import ThreadPool
from tensorforce.environments import Environment
import numpy as np

POOL = ThreadPool(processes=10)

class DistributedRewardsEnvironment(Environment):

    def execute_async(self, async_func, args):
        '''
        Return an async reward.
        '''
        return POOL.apply_async(async_func, args)
