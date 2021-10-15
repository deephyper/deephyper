class Comm:
    """
    :meta private:
    """
    size = 1
    def Get_rank(self):
        return 0

class MPI:
    """
    :meta private:
    """
    COMM_WORLD = Comm()
