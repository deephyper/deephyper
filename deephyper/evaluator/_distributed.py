import logging
import time
import pickle

from typing import List, Tuple

from deephyper.evaluator import Job

import mpi4py

# !To avoid initializing MPI when module is imported (MPI is optional)
mpi4py.rc.initialize = False
mpi4py.rc.finalize = True
from mpi4py import MPI  # noqa: E402

TAG_INIT = 20
TAG_DATA = 30


def distributed(backend: str):
    """Decorator transforming an Evaluator into a ``Distributed{Evaluator}``.

    For the decorator:

    Args:
        backend (str): Use ``"mpi"`` for pure MPI backend. Use ``"s4m"`` for Share4Me backend.

    For the returned evaluator:

    Args:
        comm: An MPI communicator. Defaults to ``None`` for ``MPI.COMM_WORLD``.
        share_freq (int): The frequency at which data should be shared between ranks of the distributed evaluator.
    """

    def wrapper(evaluator_class):

        if not (backend in ["mpi", "s4m"]):
            raise ValueError(f"Unknown backend={backend} for distributed Evaluator!")
        logging.info(
            f"Creating Distributed{evaluator_class.__name__} with backend='{backend}'."
        )

        if backend == "mpi":

            def __init__(self, *args, comm=None, share_freq=1, **kwargs):
                evaluator_class.__init__(self, *args, **kwargs)
                if not MPI.Is_initialized():
                    MPI.Init_thread()
                self.comm = comm if comm else MPI.COMM_WORLD

                # number of local jobs to evaluate before sharing with other ranks
                self.share_freq = share_freq
                # number of local jobs done since last sharing with other ranks
                self.num_local_done = 0

                self.size = self.comm.Get_size()
                self.rank = self.comm.Get_rank()
                self.num_total_workers = self.num_workers * self.size

        elif backend == "s4m":

            def __init__(self, *args, comm=None, share_freq=1, **kwargs):
                evaluator_class.__init__(self, *args, **kwargs)
                if not MPI.Is_initialized():
                    MPI.Init_thread()
                self.comm = comm if comm else MPI.COMM_WORLD

                # number of local jobs to evaluate before sharing with other ranks
                self.share_freq = share_freq
                # number of local jobs done since last sharing with other ranks
                self.num_local_done = 0

                self.size = self.comm.Get_size()
                self.rank = self.comm.Get_rank()
                self.num_total_workers = self.num_workers * self.size

                # The constructor is going to do some collective communication
                # across processes of the provided MPI communicator, so make
                # sure this call is done by all the processes at the same time.
                logging.info("Starting S4M service...")
                self._s4m_service = s4m.S4MService(self.comm, "verbs://")
                logging.info("S4M service running!")

                # Wait for all s4m services to be started
                logging.info("MPI Barrier...")
                self.comm.Barrier()
                logging.info("MPI Barrier done!")

        def _on_launch(self, job):
            """Called after a job is started."""
            job.rank = self.rank
            evaluator_class._on_launch(self, job)

        def _on_done(self, job):
            """Called after a job has completed."""
            evaluator_class._on_done(self, job)
            job.run_function = None
            self.num_local_done += 1

        def allgather(self, jobs: List[Job]) -> List[Job]:
            logging.info("Broadcasting to all...")
            t1 = time.time()
            all_data = self.comm.allgather(jobs)
            received_jobs = []

            for i, chunk in enumerate(all_data):
                if i != self.rank:
                    received_jobs.extend(chunk)

            n_received = len(received_jobs)

            self.jobs_done.extend(received_jobs)
            logging.info(
                f"Broadcast received {n_received} configurations in {time.time() - t1:.4f} sec."
            )
            return received_jobs

        if backend == "mpi":

            def broadcast(self, jobs: List[Job]):
                logging.info("Broadcasting jobs to all...")
                t1 = time.time()

                data = MPI.pickle.dumps(jobs)

                req_send = [
                    self.comm.Isend(data, dest=i, tag=TAG_DATA)
                    for i in range(self.size)
                    if i != self.rank
                ]
                MPI.Request.waitall(req_send)

                logging.info(f"Broadcasting to all done in {time.time() - t1:.4f} sec.")

            def receive(self) -> List[Job]:
                logging.info("Receiving jobs from any...")
                t1 = time.time()

                received_any = self.size > 1
                received_jobs = []
                while received_any:

                    received_any = False
                    req_recv = [
                        self.comm.irecv(source=i, tag=TAG_DATA)
                        for i in range(self.size)
                        if i != self.rank
                    ]

                    # asynchronous
                    for i, req in enumerate(req_recv):
                        try:
                            done, jobs = req.test()
                            if done:
                                received_any = True
                                received_jobs.extend(jobs)
                            else:
                                req.cancel()
                        except pickle.UnpicklingError:
                            logging.error(f"UnpicklingError for request {i}")

                self.jobs_done.extend(received_jobs)
                logging.info(
                    f"Received {len(received_jobs)} configurations in {time.time() - t1:.4f} sec."
                )
                return received_jobs

        elif backend == "s4m":
            import s4m

            def broadcast(self, jobs: List[Job]):
                logging.info("Broadcasting jobs to all...")
                t1 = time.time()

                data = MPI.pickle.dumps(jobs)

                self._s4m_service.broadcast(data)

                logging.info(f"Sending to all done in {time.time() - t1:.4f} sec.")

            def receive(self):
                logging.info("Receiving jobs from any...")
                t1 = time.time()

                # The receive function is non-blocking and will check
                # for available data sent by other processes. If data
                # is available, the function will return a pair (source, data)
                # where source is the rank that sent the data, and data is a
                # bytes object. If no data is available, the function will
                # return None.
                received_any = True
                received_jobs = []
                while received_any:
                    data = self._s4m_service.receive()
                    if data is None:
                        received_any = False
                    else:
                        source_rank, data = data
                        try:
                            jobs = MPI.pickle.loads(data)
                        except pickle.UnpicklingError:
                            logging.error(
                                f"UnpicklingError for request source {source_rank}"
                            )
                            continue
                        received_jobs.extend(jobs)

                self.jobs_done.extend(received_jobs)
                logging.info(
                    f"Received {len(received_jobs)} configurations in {time.time() - t1:.4f} sec."
                )
                return received_jobs

        def share(
            self, jobs: List[Job], sync_communication=False
        ) -> Tuple[List[Job], List[Job]]:

            if self.num_local_done % self.share_freq == 0:
                if sync_communication:
                    other_jobs = self.allgather(jobs)
                else:
                    self.broadcast(jobs)
                    other_jobs = self.receive()

            return jobs, other_jobs

        def gather(self, *args, sync_communication=False, **kwargs):
            jobs = evaluator_class.gather(self, *args, **kwargs)
            jobs, other_jobs = self.share(jobs, sync_communication)
            return jobs, other_jobs

        def dump_evals(self, *args, **kwargs):
            if self.rank == 0:
                evaluator_class.dump_evals(self, *args, **kwargs)

        cls_attrs = {
            "__init__": __init__,
            "_on_launch": _on_launch,
            "_on_done": _on_done,
            "allgather": allgather,
            "broadcast": broadcast,
            "receive": receive,
            "share": share,
            "gather": gather,
            "dump_evals": dump_evals,
        }

        distributed_evaluator_class = type(
            f"Distributed{evaluator_class.__name__}", (evaluator_class,), cls_attrs
        )

        return distributed_evaluator_class

    return wrapper
