from deephyper.layers.padding import Padding

# When loading models with: "model.load('file.h5', custom_objects=custom_objects)"
custom_objects = {"Padding": Padding}

