from .tensorboard_logger import TensorBoardLogger

class Callback:
    def __init__(self, model, optimizer, tensorboard_logger):
        self.optimizer = optimizer
        self.model = model
        self.tensorboard_logger = tensorboard_logger

        return

    def before_epoch(self):
        return

    def after_epoch(self):
        return

    def before_train(self):
        return

    def after_train(self):
        return

    def before_step(self):
        return

    def after_discriminator_step(self, tape, loss, values_dict, step, mode):
        self.optimizer.apply_discriminator_gradients(tape, loss, self.model)

        self.tensorboard_logger.log_scalar(mode, 'learning_rate', self.optimizer.get_current_lr(step), step)
        self.tensorboard_logger.log_scalars(mode, values_dict, step)

        return

    def after_generator_step(self, tape, loss, values_dict, step, mode):
        self.optimizer.apply_generator_gradients(tape, loss, self.model)

        self.tensorboard_logger.log_scalar(mode, 'learning_rate', self.optimizer.get_current_lr(step), step)
        self.tensorboard_logger.log_scalars(mode, values_dict, step)

        return