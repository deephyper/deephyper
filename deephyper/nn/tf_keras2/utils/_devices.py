import tensorflow as tf


def set_memory_growth_for_visible_gpus(enable=True):
    # GPU Configuration if available
    physical_devices = tf.config.list_physical_devices("GPU")
    for i in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[i], enable)
