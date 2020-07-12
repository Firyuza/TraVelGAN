import tensorflow as tf

from ..registry import GENERATOR
from ..builder import build_norm_layer, build_loss

class UNetDownBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, max_pool_size, norm_cfg):
        super(UNetDownBlock, self).__init__()

        self.leaky_relu = tf.keras.layers.LeakyReLU()
        self.conv3x3_a = tf.keras.layers.Conv2D(filters, (kernel_size, kernel_size), padding='same')
        self.conv3x3_a_gn = tf.keras.layers.BatchNormalization() # build_norm_layer(norm_cfg, filters)
        self.conv3x3_b = tf.keras.layers.Conv2D(filters, (kernel_size, kernel_size), padding='same')
        self.conv3x3_b_gn = tf.keras.layers.BatchNormalization() # build_norm_layer(norm_cfg, filters)
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(max_pool_size, max_pool_size))

    def call(self, input_tensor, training=True):
        output = self.conv3x3_a(input_tensor)
        output = self.conv3x3_a_gn(output, training)
        output = self.leaky_relu(output)
        output_1 = self.conv3x3_b(output)
        output_1 = self.conv3x3_b_gn(output_1, training)
        output_1 = self.leaky_relu(output_1)
        output = self.max_pool(output_1)

        return output_1, output

class UNetUpBlock(tf.keras.layers.Layer):
    def __init__(self, filters, filters_concat, kernel_size, up_kernel_size, up_conv_strides, norm_cfg):
        super(UNetUpBlock, self).__init__()

        self.leaky_relu = tf.keras.layers.LeakyReLU()
        self.conv3x3_a = tf.keras.layers.Conv2D(filters, (kernel_size, kernel_size), padding='same')
        self.conv3x3_a_gn = tf.keras.layers.BatchNormalization() # build_norm_layer(norm_cfg, filters)
        self.conv3x3_b = tf.keras.layers.Conv2D(filters, (kernel_size, kernel_size), padding='same')
        self.conv3x3_b_gn = tf.keras.layers.BatchNormalization() # build_norm_layer(norm_cfg, filters)
        self.upconv = tf.keras.layers.Conv2DTranspose(filters, (up_kernel_size, up_kernel_size), strides=up_conv_strides)
        self.upconv_gn = tf.keras.layers.BatchNormalization() # build_norm_layer(norm_cfg, filters)
        self.conv3x3_c = tf.keras.layers.Conv2D(filters_concat, (up_kernel_size, up_kernel_size), padding='same')
        self.conv3x3_c_gn = tf.keras.layers.BatchNormalization() # build_norm_layer(norm_cfg, filters_concat)


    def call(self, input_tensor, input_tensor_concat, training=True):
        output = self.conv3x3_a(input_tensor)
        output = self.conv3x3_a_gn(output, training)
        output = self.leaky_relu(output)
        output = self.conv3x3_b(output)
        output = self.conv3x3_b_gn(output, training)
        output = self.leaky_relu(output)
        output = self.upconv(output)
        output = self.upconv_gn(output, training)
        output = self.leaky_relu(output)
        output = self.conv3x3_c(output)
        output = self.conv3x3_c_gn(output, training)
        output = self.leaky_relu(output)
        output = tf.concat([input_tensor_concat, output], axis=3)

        return output

@GENERATOR.register_module
class UNet(tf.keras.models.Model):
    def __init__(self, activation, norm_cfg, loss_type=None):
        super(UNet, self).__init__()

        self.activation = getattr(tf.nn, activation)
        self.loss_type = loss_type

        self.leaky_relu = tf.keras.layers.LeakyReLU()
        self.conv3x3_1 = UNetDownBlock(64, 3, 2, norm_cfg)
        self.conv3x3_2 = UNetDownBlock(128, 3, 2, norm_cfg)
        self.conv3x3_3 = UNetDownBlock(256, 3, 2, norm_cfg)
        self.conv3x3_4 = UNetDownBlock(512, 3, 2, norm_cfg)

        self.conv3x3_5 = UNetUpBlock(1024, 512, 3, 2, 2, norm_cfg)
        self.conv3x3_6 = UNetUpBlock(512, 256, 3, 2, 2, norm_cfg)
        self.conv3x3_7 = UNetUpBlock(256, 128, 3, 2, 2, norm_cfg)
        self.conv3x3_8 = UNetUpBlock(128, 64, 3, 2, 2, norm_cfg)

        self.conv3x3_9a = tf.keras.layers.Conv2D(64, (3, 3), padding='same')
        self.conv3x3_9a_gn = tf.keras.layers.BatchNormalization() # build_norm_layer(norm_cfg, 64)
        self.conv3x3_9b = tf.keras.layers.Conv2D(64, (3, 3), padding='same')
        self.conv3x3_9b_gn = tf.keras.layers.BatchNormalization() # build_norm_layer(norm_cfg, 64)
        self.conv1x1_9 = tf.keras.layers.Conv2D(3, (1, 1), padding='same')

        if loss_type is not None:
            self.loss_type = build_loss(loss_type)


    def call_loss(self, logits):
        assert self.loss_type is not None
        losses = dict()

        losses['G_adv'] = self.loss_type(tf.nn.sigmoid(logits), tf.ones_like(logits))

        return losses

    def call(self, input_tensor, training=True):
        output_1, output = self.conv3x3_1(input_tensor, training)
        output_2, output = self.conv3x3_2(output, training)
        output_3, output = self.conv3x3_3(output, training)
        output_4, output = self.conv3x3_4(output, training)

        output = self.conv3x3_5(output, output_4, training)
        output = self.conv3x3_6(output, output_3, training)
        output = self.conv3x3_7(output, output_2, training)
        output = self.conv3x3_8(output, output_1, training)

        output = self.conv3x3_9a(output)
        output = self.conv3x3_9a_gn(output, training)
        output = self.leaky_relu(output)
        output = self.conv3x3_9b(output)
        output = self.conv3x3_9a_gn(output, training)
        output = self.leaky_relu(output)
        output = self.conv1x1_9(output)

        output = self.activation(output)

        return output