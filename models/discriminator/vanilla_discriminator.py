import tensorflow as tf

from GroupNorm import GroupNorm
from ..registry import DISCRIMINATOR
from .. import builder

@DISCRIMINATOR.register_module
class Discriminator(tf.keras.models.Model):
    def __init__(self, norm_cfg, loss_type=None, output_size=1):
        super(Discriminator, self).__init__()

        self.loss_type = loss_type

        self.leaky_relu = tf.keras.layers.LeakyReLU()
        self.conv14x14 = tf.keras.layers.Conv2D(32, (14, 14), strides=2)
        self.gn_1 = GroupNorm(32)
        self.conv7x7 = tf.keras.layers.Conv2D(64, (7, 7), strides=2)
        self.gn_2 = GroupNorm(64)
        self.conv4x4_1 = tf.keras.layers.Conv2D(128, (4, 4), strides=2)
        self.gn_3 = GroupNorm(128)
        self.conv4x4_2 = tf.keras.layers.Conv2D(256, (4, 4), strides=2)
        self.gn_4 = GroupNorm(256)
        self.conv4x4_3 = tf.keras.layers.Conv2D(512, (4, 4), strides=2)
        self.gn_5 = GroupNorm(512)
        self.dense = tf.keras.layers.Dense(output_size)

        if loss_type is not None:
            self.loss_type = builder.build_loss(loss_type)

        return

    def call_loss(self, real, fake):
        assert self.loss_type is not None

        losses = dict()

        real_predictions = tf.sigmoid(real)
        fake_predictions = tf.sigmoid(fake)

        losses['D_real'] = self.loss_type(real_predictions, tf.ones_like(real_predictions))
        losses['D_fake'] = self.loss_type(fake_predictions, tf.zeros_like(fake_predictions))
        losses['D_loss'] = losses['D_real'] + losses['D_fake']

        return losses


    def call(self, inputs, is_training=True):
        output = self.conv14x14(inputs)
        output = self.gn_1(output, is_training)
        output = self.leaky_relu(output)
        output = self.conv7x7(output)
        output = self.gn_2(output, is_training)
        output = self.leaky_relu(output)
        output = self.conv4x4_1(output)
        output = self.gn_3(output, is_training)
        output = self.leaky_relu(output)
        output = self.conv4x4_2(output)
        output = self.gn_4(output, is_training)
        output = self.leaky_relu(output)
        output = self.conv4x4_3(output)
        output = self.gn_5(output)
        output = self.leaky_relu(output)
        output = tf.reshape(output, [-1, output.shape[1] * output.shape[2] * 512])
        output = self.dense(output)

        return output
