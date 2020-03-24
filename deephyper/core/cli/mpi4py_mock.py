class Comm:
    size = 1
    def Get_rank(self):
        return 0

class MPI:
    COMM_WORLD = Comm()
