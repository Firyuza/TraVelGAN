from abc import ABCMeta, abstractmethod

import tensorflow as tf

class BaseOptimizer(metaclass=ABCMeta):

    def __init__(self, optimizer_cfg):
        tf_lr_schedule = getattr(tf.keras.optimizers.schedules, optimizer_cfg.pop('lr_schedule_type'))
        lr_schedule = tf_lr_schedule(**optimizer_cfg.pop('lr_schedule'))

        tf_optimizer = getattr(tf.keras.optimizers, optimizer_cfg.pop('type'))

        params = optimizer_cfg.pop('params')
        if params is None:
            self.optimizer = tf_optimizer(learning_rate=lr_schedule)
        else:
            self.optimizer = tf_optimizer(**params, learning_rate=lr_schedule)

        return

    def get_current_lr(self, step):
        lr = self.optimizer.learning_rate.__call__(step).numpy()

        return lr

    def get_weights(self):

        return self.optimizer.get_weights()

    def set_weights(self, weights):
        self.optimizer.set_weights(weights)

        return

    def assign_step(self, step):
        self.optimizer.iterations.assign(step)

        return

    @abstractmethod
    def apply_discriminator_gradients(self, *args):
        pass

    @abstractmethod
    def apply_generator_gradients(self, *args):
        pass