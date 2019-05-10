from deephyper.search.nas.model.space.structure import KerasStructure

def create_struct_full_skipco(input_shape, output_shape, create_cell, num_cells):
    """
        Create a SequentialStructure object.

        Args:
            input_shape (tuple): shape of input tensor
            output_shape (tuple): shape of output tensor
            create_cell (function): function that create a cell, take one argument (inputs: list(None))
            num_cells (int): number of cells in the sequential structure

        Return:
            KerasStructure: the corresponding built structure.
    """

    network = KerasStructure(input_shape, output_shape)
    input_nodes = network.input_nodes

    func = lambda: create_cell(input_nodes)
    network.add_cell_f(func)

    func = lambda x: create_cell(x)
    for i in range(num_cells-1):
        network.add_cell_f(func, num=None)

    return network

def create_seq_struct(input_shape, output_shape, create_cell, num_cells):
    """
        Create a KerasStructure object.

        Args:
            input_tensor (tensor): a tensorflow tensor object
            create_cell (function): function that create a cell, take one argument (inputs: list(None))
            num_cells (int): number of cells in the sequential structure

        Return:
            KerasStructure: the corresponding built structure.
    """

    network = KerasStructure(input_shape, output_shape)
    input_nodes = network.input_nodes

    func = lambda: create_cell(input_nodes)
    network.add_cell_f(func)

    func = lambda x: create_cell(x)
    for _ in range(num_cells-1):
        network.add_cell_f(func, num=1)

    return network
