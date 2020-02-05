import tensorflow as tf

class GroupNorm(tf.keras.layers.Layer):
    def __init__(self, C, G=32, eps=1e-5):
        super(GroupNorm, self).__init__()

        self.G = G
        self.eps = eps

        self.gamma = self.add_weight(shape=(C,),
                                     name='GroupNorm_gamma',
                                     initializer=tf.constant_initializer(1.))
        self.beta = self.add_weight(shape=(C,),
                                    name='GroupNorm_beta',
                                    initializer=tf.constant_initializer(0.))
        self.built = True

    def call(self, input_tensor, is_training=False):
        N, H, W, C = input_tensor.get_shape().as_list()
        input_tensor_tr = tf.transpose(input_tensor, [0, 3, 1, 2])
        G = min(self.G, C)

        x = tf.reshape(input_tensor_tr, [-1, G, C // G, H, W])

        mean, var = tf.nn.moments(x, [2, 3, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.eps)
        # per channel gamma and beta
        gamma = tf.reshape(self.gamma, [1, C, 1, 1])
        beta = tf.reshape(self.beta, [1, C, 1, 1])

        output_tensor = tf.reshape(x, [-1, C, H, W]) * gamma + beta
        output_tensor = tf.transpose(output_tensor, [0, 2, 3, 1])

        return output_tensor