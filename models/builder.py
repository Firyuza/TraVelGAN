from utils.registry import build_from_cfg
from .registry import (GENERATOR, DISCRIMINATOR, LOSSES, GAN)


def build(cfg, registry, default_args=None):
    return build_from_cfg(cfg, registry, default_args)


def build_discriminator(cfg):
    return build(cfg, DISCRIMINATOR)


def build_generator(cfg):
    return build(cfg, GENERATOR)


def build_loss(cfg):
    return build(cfg, LOSSES)


def build_GAN(cfg, train_cfg=None, test_cfg=None):
    return build(cfg, GAN, dict(train_cfg=train_cfg, test_cfg=test_cfg))
