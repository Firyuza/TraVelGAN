from utils.registry import Registry

GENERATOR = Registry('generator')
DISCRIMINATOR = Registry('discriminator')
LOSSES = Registry('loss')
OPTIMIZERS = Registry('optimizer')
GAN = Registry('gan')
NORMALIZATION = Registry('normalization')