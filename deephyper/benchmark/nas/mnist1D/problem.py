from deephyper.problem import NaProblem
from deephyper.benchmark.nas.mnist1D.load_data import load_data
from deepspace.tabular import OneLayerSpace


#def create_search_space(input_shape=(728,), output_shape=(10,), **kwargs):
#    return OneLayerSpace()(input_shape, output_shape, regression=False, **kwargs)


Problem = NaProblem()

Problem.load_data(load_data)

Problem.search_space(OneLayerSpace)

Problem.hyperparameters(batch_size=32, learning_rate=0.1, optimizer="adam", num_epochs=10)

Problem.loss("categorical_crossentropy")

Problem.metrics(["acc"])

Problem.objective("val_acc")


# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == "__main__":
    print(Problem)
