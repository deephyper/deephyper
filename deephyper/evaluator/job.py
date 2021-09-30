class Job:
    # Job status states.
    READY = 0
    RUNNING = 1
    DONE = 2

    def __init__(self, id, config:dict, run_function):
        self.id = id
        self.config = config
        self.run_function = run_function
        self.duration = 0 # in seconds.
        self.elapsed_sec = 0 # in seconds
        self.status = self.READY
        self.result = None

    def __repr__(self) -> str:
        return f"Job(id={self.id}, status={self.status}, config={self.config})"

    def __getitem__(self, index):
        return (self.config, self.result)[index]