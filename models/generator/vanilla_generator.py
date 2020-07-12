import tensorflow as tf

from ..registry import GENERATOR
from ..builder import build_norm_layer, build_loss

@GENERATOR.register_module
class Generator(tf.keras.models.Model):
    def __init__(self, dense_in_shape, dense_out_shape, reshape_shape,
                 conv_shapes, activation, norm_cfg, loss_type=None):
        super(Generator, self).__init__()

        self.dense_in_shape = dense_in_shape
        self.dense_out_shape = dense_out_shape
        self.reshape_shape = reshape_shape
        self.conv_shapes = conv_shapes

        self.activation = getattr(tf.nn, activation)

        self.loss_type = loss_type

        self.leaky_relu = tf.keras.layers.LeakyReLU()

        self.dense = tf.keras.layers.Dense(dense_out_shape, use_bias=False, input_shape=(dense_in_shape,))
        self.reshape_dense = tf.keras.layers.Reshape(reshape_shape)
        self.dense_gn = build_norm_layer(norm_cfg, reshape_shape[-1])

        self.deconv_layers = []
        for i in range(len(self.conv_shapes) - 1):
            conv_shape = self.conv_shapes[i]
            self.deconv_layers.append([tf.keras.layers.Conv2DTranspose(conv_shape[0], (conv_shape[1], conv_shape[2]),
                                                                     strides=(conv_shape[3], conv_shape[4]),
                                                                     padding='same', use_bias=False),
                                     build_norm_layer(norm_cfg, conv_shape[0])])

        conv_shape = self.conv_shapes[-1]
        self.deconv = tf.keras.layers.Conv2DTranspose(conv_shape[0], (conv_shape[1], conv_shape[2]),
                                                       strides=(conv_shape[3], conv_shape[4]),
                                                       padding='same', use_bias=False, activation=None)

        if loss_type is not None:
            self.loss_type = build_loss(loss_type)


    def call_loss(self, logits):
        assert self.loss_type is not None
        losses = dict()

        losses['G_adv'] = self.loss_type(self.activation(logits), tf.ones_like(logits))

        return losses

    def call(self, input_tensor, training=True):
        output = self.dense(input_tensor)
        output = self.reshape_dense(output)
        output = self.dense_gn(output, training)

        for deconv_layer in self.deconv_layers:
            output = deconv_layer[0](output)
            output = deconv_layer[1](output, training)

        output = self.deconv(output)

        return output