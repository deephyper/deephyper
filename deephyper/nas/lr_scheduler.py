import tensorflow as tf


def exponential_decay(epoch, lr):
    """Keep the learning rate constant for the first 10 epochs. Then, decay the learning
    rate exponentially."""

    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
