import tensorflow as tf

from .base_optimizer import BaseOptimizer
from ..registry import OPTIMIZERS

@OPTIMIZERS.register_module
class TraVelGANOptimizer(BaseOptimizer):
    def __init__(self, optimizer_cfg):
        super(TraVelGANOptimizer, self).__init__(optimizer_cfg)

        return

    def apply_discriminator_gradients(self, tape, loss, model):
        grads = tape.gradient(loss, model.get_discriminator_variables())

        grads = [tf.clip_by_value(grad, -1., 1.) for grad in grads]

        self.optimizer.apply_gradients(zip(grads, model.get_discriminator_variables()))

        return

    def apply_generator_gradients(self, tape, loss, model):
        grads = tape.gradient(loss, model.get_generator_variables())

        grads = [tf.clip_by_value(grad, -1., 1.) for grad in grads]

        self.optimizer.apply_gradients(zip(grads, model.get_generator_variables()))

        return