import tensorflow as tf

from .base import BaseGAN
from ..registry import GAN
from .. import builder

@GAN.register_module
class TraVeLGAN(BaseGAN):
    def __init__(self, discriminator, generator,
                 noise_size,
                 siamese_network=None,
                 distance_loss=None, siamese_loss=None,
                 train_cfg=None, test_cfg=None):
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
            self.siamese_loss = builder.build_loss(siamese_loss)

    def get_discriminator_variables(self):
        return self.discriminator.trainable_variables

    def get_generator_variables(self):
        return self.generator.trainable_variables + self.siamese_network.trainable_variables

    def call_accuracy(self, predictions, labels):
        return

    def call_discriminator_loss(self, losses, real_logits, fake_logits):
        D_losses = self.discriminator.call_loss(real_logits, fake_logits)

        losses.update(D_losses)

        return losses

    def call_generator_loss(self, losses, fake_predictions, real_embeddings, fake_embeddings):
        G_losses = self.generator.call_loss(fake_predictions)
        losses.update(G_losses)

        losses['TraVeL_loss'] = self.distance_loss(real_embeddings, fake_embeddings)
        losses['Siamese_loss'] = self.siamese_loss(real_embeddings, None)

        losses['G_reg_loss'] = 0.0005 * tf.add_n([tf.nn.l2_loss(w) for w in
                                                  self.generator.trainable_variables + self.siamese_network.trainable_variables])
        # losses['S_loss'] = losses['siamese_loss'] + losses['TraVeL_loss']

        losses['G_loss'] = G_losses['G_adv'] + losses['TraVeL_loss'] + losses['G_reg_loss']
        losses['G_loss'] += losses['Siamese_loss']

        return losses


    def build_model(self, input_shape):
        self.discriminator.build((None, self.train_cfg.image_size, self.train_cfg.image_size,
                                  self.train_cfg.image_channels))
        self.generator.build((None, self.train_cfg.image_size, self.train_cfg.image_size,
                                  self.train_cfg.image_channels))

        if self.with_siamese_network:
            self.siamese_network.build((None, self.train_cfg.image_size, self.train_cfg.image_size,
                                        self.train_cfg.image_channels))

        return

    def call_train(self, image, label=None):
        losses = dict()

        # noise = tf.random.uniform([len(image), 100], minval=-1., maxval=1.)
        generated_image = self.generator(image, True)

        real_logits = self.discriminator(image, True)
        fake_logits = self.discriminator(generated_image, True)

        real_embeddings = self.siamese_network(image, True)
        real_embeddings = tf.nn.l2_normalize(real_embeddings)

        fake_embeddings = self.siamese_network(generated_image, True)
        fake_embeddings = tf.nn.l2_normalize(fake_embeddings)

        losses = self.call_discriminator_loss(losses, real_logits, fake_logits)
        losses = self.call_generator_loss(losses, fake_logits, real_embeddings, fake_embeddings)

        return losses