from abc import ABCMeta, abstractmethod

import tensorflow as tf
import numpy as np
import pycocotools.mask as maskUtils


class BaseGAN(tf.keras.models.Model, metaclass=ABCMeta):
    """Base class for GAN"""

    def __init__(self):
        super(BaseGAN, self).__init__()
        self.fp16_enabled = False

    @property
    def with_siamese_network(self):
        return hasattr(self, 'siamese_network') and self.siamese_network is not None

    @abstractmethod
    def call_accuracy(self, *args):
        pass

    @abstractmethod
    def call_generator_loss(self, *args):
        pass

    @abstractmethod
    def call_discriminator_loss(self, *args):
        pass

    @abstractmethod
    def build_model(self, *args):
        pass

    def build(self, *args):
        self.build_model(*args)

        return

    def call_test(self, imgs, img_metas, **kwargs):
        pass

    @abstractmethod
    def call_train(self, imgs, img_metas, **kwargs):
        """
        Args:
            img (list[Tensor]): list of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

             **kwargs: specific to concrete implementation
        """
        pass

    def call(self, img, img_meta, training=True, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if training:
            return self.call_train(img, img_meta, **kwargs)
        else:
            return self.call_test(img, img_meta, **kwargs)