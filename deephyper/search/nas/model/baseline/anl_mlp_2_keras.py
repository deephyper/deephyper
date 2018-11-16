from deephyper.search.nas.model.space.structure import create_dense_cell_type2, create_seq_struct_full_skipco


def create_structure(input_shape=(2,), output_shape=(1,), num_cells=2):
    return create_seq_struct_full_skipco(
        input_shape,
        output_shape,
        create_dense_cell_type2,
        num_cells)
