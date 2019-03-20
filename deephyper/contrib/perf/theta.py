import os
import tensorflow as tf
from tensorflow.keras import backend as K


"""Performance settings for Theta

OMP_NUM_THREADS='62' should be set before importing tensorflow
"""

def get_session_conf():
    """Set env variables for better performance on Theta.

        Return:
                A tf.ConfigProto object with specific settings.
    """
    os.environ['KMP_BLOCKTIME'] = '0'
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'

    session_conf = tf.ConfigProto(
                intra_op_parallelism_threads=62
                )

    return session_conf

def set_perf_settings_for_keras():
    """Set a session with performance setting for keras backend.
    """
    if not(os.environ.get('HOST') is None) and 'theta' in os.environ.get('HOST'):
        session_conf = get_session_conf()
        session = tf.Session(config=session_conf)
        K.set_session(session)


