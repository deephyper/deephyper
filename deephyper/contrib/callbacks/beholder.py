import tensorflow as tf
from tensorboard.plugins.beholder import Beholder

class BeholderCB(tf.keras.callbacks.Callback):
    """Keras callback for tensorboard beholder plugin: https://github.com/tensorflow/tensorboard/tree/master/tensorboard/plugins/beholder

    Args:
        logdir (str): path to the tensorboard log directory.
        sess: tensorflow session.
    """

    def __init__(self,logdir, sess):
        super(BeholderCB, self).__init__()
        self.beholder=Beholder(logdir=logdir)
        self.session=sess

    def on_epoch_end(self, epoch, logs=None):
        super(BeholderCB, self).on_epoch_end(epoch, logs)
        self.beholder.update(session=self.session)