import tensorflow as tf

from .base import BaseGAN
from ..registry import GAN
from .. import builder

@GAN.register_module
class TraVeLGAN(BaseGAN):
    def __init__(self, train_cfg, test_cfg, discriminator, generator,
                 BC_loss, TraVeL_loss=None, siamese_loss=None,
                 siamese_network=None):
        super(TraVeLGAN, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.discriminator = builder.build_discriminator(discriminator)
        self.generator = builder.build_generator(generator)

        if siamese_network is not None:
            self.siamese_network = builder.build_discriminator(siamese_network)

    def build(self, input_shape, **kwargs):
        self.discriminator.build((None, self.train_cfg.image_size, self.train_cfg.image_size, self.train_cfg.image_channels))
        self.generator.build(
            (None, self.train_cfg.image_size, self.train_cfg.image_size, self.train_cfg.image_channels))

        if self.with_siamese_network:
            self.siamese_network.build((None, self.train_cfg.image_size, self.train_cfg.image_size, self.train_cfg.image_channels))

        return

    def call_train(self, image, label=None, training=True):
        return