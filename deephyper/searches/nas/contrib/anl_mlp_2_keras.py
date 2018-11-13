from deephyper.searches.nas.model.keras.structure import create_dense_cell_type2, create_seq_struct_full_skipco


def create_structure(input_shape, output_shape, num_cells):
    return create_seq_struct_full_skipco(input_shape, output_shape, create_dense_cell_type2, num_cells)
