import time

import numpy as np
import tensorflow as tf
from mpi4py import MPI

import deephyper.search.nas.utils.common.tf_util as U
from deephyper.search import util
from deephyper.search.nas.utils._logging import JsonMessage as jm

TAG_UPDATE_START = 1
TAG_UPDATE_DONE  = 2

dh_logger = util.conf_logger('deephyper.baselines.common.mpi_adam_async')

class MpiAdamAsync(object):
    def __init__(self, var_list, *, beta1=0.9, beta2=0.999, epsilon=1e-08, scale_grad_by_procs=True, comm=None):
        self.var_list = var_list
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.scale_grad_by_procs = scale_grad_by_procs
        size = sum(U.numel(v) for v in var_list)
        self.m = np.zeros(size, 'float32')
        self.v = np.zeros(size, 'float32')
        self.t = 0
        self.setfromflat = U.SetFromFlat(var_list)
        self.getflat = U.GetFlat(var_list)
        self.comm = MPI.COMM_WORLD if comm is None else comm
        self.status = MPI.Status()

    def master_update(self):
        # Receive gradient from a worker
        t1 = time.time()
        ##
        update_info = self.comm.recv(source=MPI.ANY_SOURCE,
                                     tag=TAG_UPDATE_START,
                                     status=self.status)
        worker_source = self.status.Get_source()
        ##
        t2 = time.time()
        t = t2 - t1
        dh_logger.info(jm(type='receive_gradient', rank=self.comm.Get_rank(), duration=t, start_time=t1, end_time=t2, rank_worker_source=worker_source))

        t1 = time.time()
        ##
        workerg = update_info['workerg']
        stepsize = update_info['stepsize']
        if self.scale_grad_by_procs:
            workerg /= self.comm.Get_size() - 1 # one is the parameter server

        self.t += 1
        a = stepsize * np.sqrt(1 - self.beta2**self.t)/(1 - self.beta1**self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * workerg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (workerg * workerg)
        step = (- a) * self.m / (np.sqrt(self.v) + self.epsilon)
        update_vars = self.getflat() + step
        self.setfromflat(update_vars)
        ##
        t2 = time.time()
        t = t2 - t1
        dh_logger.info(jm(type='update_parameters', rank=self.comm.Get_rank(), duration=t, start_time=t1, end_time=t2, rank_worker_source=worker_source))

        t1 = time.time()
        ##
        self.comm.send(update_vars, dest=worker_source, tag=TAG_UPDATE_DONE)
        ##
        t2 = time.time()
        t = t2 - t1
        dh_logger.info(jm(type='send_parameters', rank=self.comm.Get_rank(), duration=t, start_time=t1, end_time=t2, rank_worker_dest=worker_source))

        return worker_source

    def worker_update(self, localg, stepsize):
        # Send local gradient to master
        update_info = dict(workerg=localg, stepsize=stepsize)

        t1 = time.time()
        ##
        self.comm.send(update_info, dest=0, tag=TAG_UPDATE_START)
        update_vars = self.comm.recv(source=0, tag=TAG_UPDATE_DONE, status=self.status)
        ##
        t2 = time.time()
        t = t2 - t1
        dh_logger.info(jm(type='receive_parameters', rank=self.comm.Get_rank(), duration=t, start_time=t1, end_time=t2, master_rank=0))

        t1 = time.time()
        ##
        self.setfromflat(update_vars)
        ##
        t2 = time.time()
        t = t2 - t1
        dh_logger.info(jm(type='setfromflat', rank=self.comm.Get_rank(), duration=t, start_time=t1, end_time=t2, master_rank=0))

    def sync(self):
        theta = self.getflat()
        self.comm.Bcast(theta, root=0)
        self.setfromflat(theta)
