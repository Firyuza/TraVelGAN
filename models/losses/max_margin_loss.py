import tensorflow as tf
import numpy as np

from ..registry import LOSSES

@LOSSES.register_module
class MaxMarginLoss(tf.keras.losses.Loss):
    def __init__(self, delta, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='max_margin'):
        super(MaxMarginLoss, self).__init__(reduction=reduction,
                                        name=name)
        self.delta = delta

    def call(self, embeddings, dump_data=None):
        pairwise_distances_squared = tf.add(
            tf.reduce_sum(tf.square(embeddings), axis=[1], keepdims=True),
            tf.reduce_sum(tf.square(tf.transpose(embeddings)), axis=[0], keepdims=True))\
                                     - 2.0 * tf.matmul(embeddings, tf.transpose(embeddings))

        num_data = tf.shape(embeddings)[0]
        pairwise_distances = tf.sqrt(pairwise_distances_squared)
        mask_off_diagonals = tf.ones_like(pairwise_distances) - tf.linalg.diag(
            tf.ones([num_data]))
        pairwise_distances = tf.multiply(pairwise_distances, mask_off_diagonals)
        pairwise_distances = pairwise_distances + tf.linalg.diag([self.delta] * len(embeddings))

        max_elements = tf.maximum(0., self.delta - pairwise_distances)
        ones = np.ones([el for el in max_elements.shape])
        upper_non_zero = np.triu(ones, k=0)
        max_elements = tf.multiply(max_elements, upper_non_zero)

        result = tf.reduce_sum(max_elements)

        return result