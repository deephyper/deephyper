'''
 * @Author: romain.egele
 * @Date: 2018-06-19 17:02:17
 * @Last Modified by:   romain.egele
 * @Last Modified time: 2018-06-19 17:02:17
'''
import tensorflow as tf
import os


class Summary:
    def __init__(self, config, model):
        self.is_configured = False
        if (config.get('accuracy')):
            # Create a summary to monitor accuracy tensor
            tf.summary.scalar("accuracy", model.accuracy)
            self.is_configured = True

        if (config.get('loss')):
            # Create a summary to monitor loss tensor
            tf.summary.scalar("loss", model.loss)
            self.is_configured = True

        # Merge all summaries into a single op
        self.merged_summary_op = tf.summary.merge_all()

    def open(self, directory):
        if ( directory != '' ):
          self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.summary_writer = tf.summary.FileWriter(
            self.directory, graph=tf.get_default_graph())

    def close(self):
        self.summary_writer.close()

    def add(self, summary, global_step):
        '''
        Add a new summary to the logs.
        '''
        self.summary_writer.add_summary(summary, global_step)
