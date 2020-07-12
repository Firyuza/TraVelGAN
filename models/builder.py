import inspect
import tensorflow as tf

from utils.registry import build_from_cfg
from .registry import (GENERATOR, DISCRIMINATOR, LOSSES, OPTIMIZERS, GAN, NORMALIZATION)


def build(cfg, registry, default_args=None):
    return build_from_cfg(cfg, registry, default_args)

def build_discriminator(cfg, default_args=None):
    return build(cfg, DISCRIMINATOR, default_args)

def build_generator(cfg, default_args=None):
    return build(cfg, GENERATOR, default_args)

def build_optimizer(cfg, default_args=None):
    return build(cfg, OPTIMIZERS, default_args)

def build_loss(cfg, default_args=None):
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        tf_loss = LOSSES.get(obj_type)
        if tf_loss is None:
            try:
                return getattr(tf.keras.losses, obj_type)()
            except:
                raise KeyError('{} is not in the {} registry'.format(
                    obj_type, LOSSES.name))
        else:
            return build(cfg, LOSSES)
    elif not inspect.isclass(obj_type):
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    else:
        return build(cfg, LOSSES)

def build_norm_layer(cfg, *inputs):
    """ Build normalization layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    args = cfg.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_type = NORMALIZATION.get(obj_type)
        if obj_type is None:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, NORMALIZATION.name))
    elif not inspect.isclass(obj_type):
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))

    return obj_type(*inputs)


def build_GAN(cfg, train_cfg=None, test_cfg=None):
    return build(cfg, GAN, dict(train_cfg=train_cfg, test_cfg=test_cfg))
