import tensorflow as tf

from .base import BaseGAN
from ..registry import GAN
from .. import builder

@GAN.register_module
class TraVeLGAN(BaseGAN):
    def __init__(self, train_cfg, test_cfg, discriminator, generator,
                 siamese_network=None,
                 distance_loss=None, siamese_loss=None):
        super(TraVeLGAN, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.discriminator = builder.build_discriminator(discriminator)
        self.generator = builder.build_generator(generator)

        if siamese_network is not None:
            self.siamese_network = builder.build_discriminator(siamese_network)

        if distance_loss is not None:
            self.distance_loss = builder.build_loss(distance_loss)
        if siamese_loss is not None:
            self.siameese_loss = builder.build_loss(siamese_loss)

    def call_accuarcy(self, predictions, labels):
        return

    def call_discriminator_loss(self, losses, real_predictions, fake_predictions):
        D_losses = self.discriminator.call_loss(real_predictions, fake_predictions)

        losses.update(D_losses)

        return losses

    def call_generator_loss(self, losses, fake_predictions, real_embeddings, fake_embeddings):
        G_losses = self.generator.call_loss(fake_predictions)
        losses.update(G_losses)

        losses['TraVeL_loss'] = self.distance_loss(real_embeddings, fake_embeddings)
        losses['siamese_loss'] = self.max_margin_loss(real_embeddings, None)

        losses['S_loss'] = losses['siamese_loss'] + losses['TraVeL_loss']

        losses['G_loss'] = G_losses['G_adv'] + losses['TraVeL_loss']
        losses['G_loss'] += losses['S_loss']

        return losses

    def build_model(self, input_shape):
        self.discriminator.build((None, self.train_cfg.image_size, self.train_cfg.image_size, self.train_cfg.image_channels))
        self.generator.build(
            (None, self.train_cfg.image_size, self.train_cfg.image_size, self.train_cfg.image_channels))

        if self.with_siamese_network:
            self.siamese_network.build((None, self.train_cfg.image_size, self.train_cfg.image_size, self.train_cfg.image_channels))

        return

    def call_train(self, image, label=None):
        losses = dict()

        generated_image = self.generator(image, True)

        real_predictions = tf.sigmoid(self.discriminator(image, True))
        fake_predictions = tf.sigmoid(self.discriminator(generated_image, True))

        real_embeddings = self.siamese(image, True)
        fake_embeddings = self.siamese(generated_image, True)

        losses = self.call_discriminator_loss(losses, real_predictions, fake_predictions)
        losses = self.call_generator_loss(losses, fake_predictions, real_embeddings, fake_embeddings)

        return losses