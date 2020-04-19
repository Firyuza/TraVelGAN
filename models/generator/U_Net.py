import tensorflow as tf

from GroupNorm import GroupNorm
from ..registry import GENERATOR

@GENERATOR.register_module
class UNetDownBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, max_pool_size):
        super(UNetDownBlock, self).__init__()
        self.conv3x3_a = tf.keras.layers.Conv2D(filters, (kernel_size, kernel_size), padding='same')
        self.conv3x3_a_gn = GroupNorm(filters)
        self.conv3x3_b = tf.keras.layers.Conv2D(filters, (kernel_size, kernel_size), padding='same')
        self.conv3x3_b_gn = GroupNorm(filters)
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(max_pool_size, max_pool_size))

    def call(self, input_tensor, is_training=True):
        output = self.conv3x3_a(input_tensor)
        output = self.conv3x3_a_gn(output)
        output_1 = self.conv3x3_b(output)
        output_1 = self.conv3x3_b_gn(output_1)
        output = self.max_pool(output_1)

        return output_1, output

class UNetUpBlock(tf.keras.layers.Layer):
    def __init__(self, filters, filters_concat, kernel_size, up_kernel_size, up_conv_strides):
        super(UNetUpBlock, self).__init__()

        self.conv3x3_a = tf.keras.layers.Conv2D(filters, (kernel_size, kernel_size), padding='same')
        self.conv3x3_a_gn = GroupNorm(filters)
        self.conv3x3_b = tf.keras.layers.Conv2D(filters, (kernel_size, kernel_size), padding='same')
        self.conv3x3_b_gn = GroupNorm(filters)
        self.upconv = tf.keras.layers.Conv2DTranspose(filters, (up_kernel_size, up_kernel_size), strides=up_conv_strides)
        self.upconv_gn = GroupNorm(filters)
        self.conv3x3_c = tf.keras.layers.Conv2D(filters_concat, (up_kernel_size, up_kernel_size), padding='same')
        self.conv3x3_c_gn = GroupNorm(filters_concat)


    def call(self, input_tensor, input_tensor_concat, is_training=True):
        output = self.conv3x3_a(input_tensor)
        output = self.conv3x3_a_gn(output)
        output = self.conv3x3_b(output)
        output = self.conv3x3_b_gn(output)
        output = self.upconv(output)
        output = self.upconv_gn(output)
        output = self.conv3x3_c(output)
        output = self.conv3x3_c_gn(output)
        output = tf.concat([input_tensor_concat, output], axis=3)

        return output

class UNet(tf.keras.models.Model):
    def __init__(self, loss_type=None):
        super(UNet, self).__init__()

        self.loss_type = loss_type

        self.conv3x3_1 = UNetDownBlock(64, 3, 2)
        self.conv3x3_2 = UNetDownBlock(128, 3, 2)
        self.conv3x3_3 = UNetDownBlock(256, 3, 2)
        self.conv3x3_4 = UNetDownBlock(512, 3, 2)

        self.conv3x3_5 = UNetUpBlock(1024, 512, 3, 2, 2)
        self.conv3x3_6 = UNetUpBlock(512, 256, 3, 2, 2)
        self.conv3x3_7 = UNetUpBlock(256, 128, 3, 2, 2)
        self.conv3x3_8 = UNetUpBlock(128, 64, 3, 2, 2)

        self.conv3x3_9a = tf.keras.layers.Conv2D(64, (3, 3), padding='same')
        self.conv3x3_9a_gn = GroupNorm(64)
        self.conv3x3_9b = tf.keras.layers.Conv2D(64, (3, 3), padding='same')
        self.conv3x3_9b_gn = GroupNorm(64)
        self.conv1x1_9 = tf.keras.layers.Conv2D(3, (1, 1), padding='same')

        if loss_type is not None:
            self.loss_type = loss_type


    def call_loss(self, predictions):
        assert self.loss_type is not None
        losses = dict()

        losses['G_adv'] = self.loss_type(predictions, tf.ones_like(predictions))

        return losses

    def call(self, input_tensor, is_training=True):
        output_1, output = self.conv3x3_1(input_tensor)
        output_2, output = self.conv3x3_2(output)
        output_3, output = self.conv3x3_3(output)
        output_4, output = self.conv3x3_4(output)

        output = self.conv3x3_5(output, output_4)
        output = self.conv3x3_6(output, output_3)
        output = self.conv3x3_7(output, output_2)
        output = self.conv3x3_8(output, output_1)

        output = self.conv3x3_9a(output)
        output = self.conv3x3_9a_gn(output)
        output = self.conv3x3_9b(output)
        output = self.conv3x3_9a_gn(output)
        output = self.conv1x1_9(output)

        return output