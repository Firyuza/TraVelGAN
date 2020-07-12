import tensorflow as tf
import tensorflow_addons as tfa

from ..registry import LOSSES

@LOSSES.register_module
class DistanceLoss(tf.keras.losses.Loss):
    def __init__(self, distance_type=None, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='distance_loss'):
        super(DistanceLoss, self).__init__(reduction=reduction,
                                        name=name)

        self.distance_type = distance_type

    def call(self, real_embeddings, fake_embeddings):
        embd_repeat_column = tf.stack([real_embeddings] * len(real_embeddings), axis=1)
        embd_repeat_row = tf.stack([real_embeddings] * len(real_embeddings), axis=0)

        real_embd_diff = tf.subtract(embd_repeat_column, embd_repeat_row)

        embd_repeat_column = tf.stack([fake_embeddings] * len(fake_embeddings), axis=1)
        embd_repeat_row = tf.stack([fake_embeddings] * len(fake_embeddings), axis=0)

        fake_embd_diff = tf.subtract(embd_repeat_column, embd_repeat_row)

        dist_square = tf.math.squared_difference(real_embd_diff, fake_embd_diff)
        dist = tf.reduce_sum(dist_square, axis=2)
        dist = tf.reduce_sum(dist)

        # nrof_pairs = len(real_embeddings) * (len(real_embeddings) - 1)

        dist = tf.divide(dist, 2.)#tf.divide(dist, nrof_pairs)

        return dist