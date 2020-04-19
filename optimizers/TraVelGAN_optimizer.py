import tensorflow as tf

from .base_optimizer import BaseOptimizer

class TraVelGANOptimizer(BaseOptimizer):
    def __init__(self, optimizer_cfg):
        super(TraVelGANOptimizer, self).__init__(optimizer_cfg)

        return

    def apply_discriminator_gradients(self, tape, loss, model):
        grads = tape.gradient(loss, model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return

    def apply_generator_gradients(self, tape, loss, model):
        grads = tape.gradient(loss, model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return