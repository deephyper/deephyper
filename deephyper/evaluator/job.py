import uuid

class Job:
    # Job status states.
    READY = 0
    RUNNING = 1
    DONE = 2

    def __init__(self, id, seed:str, config:dict, run_function, num_workers=1, num_cpus=1, num_gpus=1):
        self.id = id
        self.seed = seed
        self.config = config
        self.run_function = run_function
        self.duration = 0 # in seconds.
        self.elapsed_sec = 0 # in seconds
        self.status = self.READY
        self.result = None
        self.num_workers = num_workers
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus

    def __repr__(self) -> str:
        return "Job ID: " +  str(self.id) + "\n" + "Configuration: " + str(self.config) + "\n" + "Status: " + str(self.status)

    def __getitem__(self, index):
        return (self.config, self.result)[index]