import collections


def queued(evaluator_class):
    """Decorator transforming an Evaluator into a ``Queued{Evaluator}``. The ``run_function`` used with a ``Queued{Evaluator}`` needs to have a ``dequed`` keyword-argument where the dequed resources from the queue will be passed.

    Args:
        queue (list): A list of queued resources.
        queue_pop_per_task (int, optional): The number of resources popped out of the queue each time a task is submitted. Defaults to ``1``.
    """

    def __init__(
        self, *args, queue: list = None, queue_pop_per_task: int = 1, **kwargs
    ):
        evaluator_class.__init__(self, *args, **kwargs)

        self.queue = collections.deque(queue[:])
        self.queue_pop_per_task = queue_pop_per_task

    async def execute(self, job):

        dequed = [self.queue.popleft() for _ in range(self.queue_pop_per_task)]
        self.run_function_kwargs["dequed"] = dequed

        job = await evaluator_class.execute(self, job)
        setattr(job, "dequed", dequed)

        self.queue.extend(dequed)

        return job

    cls_attrs = {"__init__": __init__, "execute": execute}

    queued_evaluator_class = type(
        f"Queued{evaluator_class.__name__}", (evaluator_class,), cls_attrs
    )

    return queued_evaluator_class
