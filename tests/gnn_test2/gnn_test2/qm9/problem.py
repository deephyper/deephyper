from deephyper.problem import NaProblem
from gnn_test2.qm9.load_data import load_data_qm9
from gnn_test2.qm9.search_space import create_search_space
from deephyper.search.nas.model.preprocessing import minmaxstdscaler

Problem = NaProblem(seed=2019)

Problem.load_data(load_data_qm9)

# Problem.preprocessing(minmaxstdscaler)

Problem.search_space(create_search_space, num_layers=2)

Problem.hyperparameters(
    batch_size=32,
    learning_rate=0.01,
    optimizer='adam',
    num_epochs=20,
    callbacks=dict(
        EarlyStopping=dict(
            monitor='val_r2', # or 'val_acc' ?
            mode='min',
            verbose=0,
            patience=5
        )
    )
)

Problem.loss('mse') # or 'categorical_crossentropy' ?

Problem.metrics(['r2']) # or 'acc' ?

Problem.objective('val_r2__last') # or 'val_acc__last' ?


# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == '__main__':
    print(Problem)