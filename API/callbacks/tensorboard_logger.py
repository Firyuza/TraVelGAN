import numpy as np
import tensorflow as tf
import os

class TensorBoardLogger:
    def __init__(self, log_dir, type):
        self.log_dir = os.path.join(log_dir, type)
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

        return

    def log_scalar(self, name, value, step=None):
        with self.summary_writer.as_default():
            tf.summary.scalar(name, value, step=step)

        return

    def log_scalars(self, values_dict, step):
        with self.summary_writer.as_default():
            for name, value in values_dict.items():
                tf.summary.scalar(name, value, step=step)

        return