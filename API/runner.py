import logging
import os.path as osp
import os
import time
import tensorflow as tf
import h5py

from .callbacks.tensorboard_logger import TensorBoardLogger
from .callbacks.callback import Callback


class Runner(object):
    """A training helper for TesnorFlow.

    Args:
        model (:obj:`torch.nn.Module`): The model to be run.
        batch_processor (callable): A callable method that process a data
            batch. The interface of this method should be
            `batch_processor(model, data, train_mode) -> dict`
        optimizer (dict or :obj:`torch.optim.Optimizer`): If it is a dict,
            runner will construct an optimizer according to it.
        work_dir (str, optional): The working directory to save checkpoints
            and logs.
        log_level (int): Logging level.
        logger (:obj:`logging.Logger`): Custom logger. If `None`, use the
            default logger.
    """

    def __init__(self,
                 model,
                 batch_processor,
                 optimizer,
                 work_dir,
                 logger):
        assert callable(batch_processor)
        self.model = model
        self.optimizer = optimizer
        self.batch_processor = batch_processor

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        # self.timestamp = get_time_str()

        self.tensorboard_logger = TensorBoardLogger(work_dir)
        self.callback = Callback(model, optimizer, self.tensorboard_logger)

        self.logger = logger
        self.work_dir = work_dir
        self.mode = None
        self._epoch = 0
        self.step = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self.step

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    def load_checkpoint(self, filename):
        print('Loading variables')

        file = h5py.File(filename, 'r')
        weight = []
        for i in range(len(file.keys())):
            weight.append(file['weight' + str(i)].value)
        self.model.set_weights(weight)

        # file = h5py.File(filename.replace('model-', 'optimizer-'), 'r')
        # weight = []
        # for i in range(len(file.keys())):
        #     weight.append(file['weight' + str(i)].value)
        # self.optimizer.set_weights(weight)

        step = int(filename.split('.h5')[0].split('-')[-1])
        self.optimizer.assign_step(step)

        self.step = step

        return

    def save_checkpoint(self, out_dir, filename_tmpl='model-{}.h5'):
        print('Saving variables')
        filename = filename_tmpl.format(self.iter)
        filepath = osp.join(out_dir, 'models')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        save_path = osp.join(filepath, filename)

        file = h5py.File(save_path, 'w')
        weight = self.model.get_weights()
        for i in range(len(weight)):
            file.create_dataset('weight' + str(i), data=weight[i])
        file.close()

        filename = filename_tmpl.format(self.iter).replace('model', 'optimizer')
        save_path = osp.join(filepath, filename)
        file = h5py.File(save_path, 'w')
        weight = self.optimizer.get_weights()
        for i in range(len(weight)):
            file.create_dataset('weight' + str(i), data=weight[i])
        file.close()

        return

    def train(self, data_loader, **kwargs):
        self.mode = 'train'
        self.data_loader = data_loader

        self.callback.before_epoch()
        for i, data_batch in enumerate(data_loader.data_loader):
            self._inner_iter = i
            self.callback.before_step()

            with tf.GradientTape() as tape:
                outputs = self.batch_processor(self.model, data_batch,
                                               train_mode=True, **kwargs)

            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')

            self.callback.after_discriminator_step(tape, outputs, outputs, self.step, self.mode)

            if i > 0 and i % 5 == 0:
                with tf.GradientTape() as tape:
                    outputs = self.batch_processor(self.model, data_batch,
                                                   train_mode=True, **kwargs)

                self.callback.after_generator_step(tape, outputs, outputs, self.step, self.mode)

            with tf.GradientTape() as tape:
                outputs = self.batch_processor(self.model, data_batch,
                                               train_mode=True, **kwargs)

            self.callback.after_discriminator_step(tape, outputs, outputs, self.step, self.mode)

            self.step += 1

            if self.step % 10000 == 0:
                self.save_checkpoint(self.work_dir)

        self.callback.after_epoch()
        self._epoch += 1
        self.save_checkpoint(self.work_dir)

        return

    def valid(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'valid'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')

        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=False, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        """Start running.
        """
        self._max_epochs = max_epochs
        self.logger.info('Start running, work_dir: %s' % self.work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow, max_epochs)
        while self.epoch < max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow

                if isinstance(mode, str):
                    if not hasattr(self, mode):
                        raise ValueError(
                            'runner has no method named "{}" to run an epoch'.
                            format(mode))
                    epoch_runner = getattr(self, mode)
                elif callable(mode):  # custom train()
                    epoch_runner = mode
                else:
                    raise TypeError('mode in workflow must be a str or '
                                    'callable function, not {}'.format(
                                        type(mode)))
                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= max_epochs:
                        return
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)

        return