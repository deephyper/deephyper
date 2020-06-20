from deephyper.problem import NaProblem
from protonation.gnnproton.load_data import load_data
from protonation.gnnproton.search_space import create_search_space
from deephyper.search.nas.model.preprocessing import minmaxstdscaler
import tensorflow as tf

Problem = NaProblem(seed=2019)

Problem.load_data(load_data)

# Problem.preprocessing(minmaxstdscaler)


def my_loss_fn(y_true, y_pred):
    y_true = tf.math.abs(y_true)
    num_non_zero = tf.cast(tf.math.count_nonzero(y_true), tf.float32)
    custom_loss = tf.div(tf.reduce_mean(tf.squared_difference(y_true, y_pred)), num_non_zero)
    return custom_loss


Problem.search_space(create_search_space)

Problem.hyperparameters(
    batch_size=32,
    learning_rate=0.001,
    optimizer='adam',
    num_epochs=10,
    callbacks=dict(
        EarlyStopping=dict(
            monitor='val_loss',  # or 'val_acc' ?
            mode='min',
            verbose=0,
            patience=5
        )
    )
)

Problem.loss(my_loss_fn)  # or 'categorical_crossentropy' ?

Problem.metrics(['mse', 'mae', 'po', 'por2'])  # or 'acc' ?

Problem.objective('val_po')  # or 'val_acc__last' ?

# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == '__main__':
    print(Problem)