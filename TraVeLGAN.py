import tensorflow as tf
import tensorflow_datasets as tfds
import os
import hdf5storage

from Discriminator import Discriminator
from DataAugmentation import DataAugmentation
from config import cfg
from Losses import MaxMarginLoss, DistanceLoss
from U_Net import UNet

class TraVeLGAN:
    def __init__(self):
        self.data_augmentation = DataAugmentation(cfg.train)
        self.__load_data()
        self.__create_graph()

        return

    def __load_data(self):
        # (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
        # self.dataset = tfds.load(name="celeb_a", split=tfds.Split.TRAIN)
        # with h5py.File('/home/firiuza/PycharmProjects/TraVeLGAN/shoes_128.hdf5', 'r') as f:
        #     dset = f

        x = hdf5storage.loadmat('/home/firiuza/MachineLearning/zap_dataset/ut-zap50k-data/image-path.mat')

        self.data_paths = [os.path.join(cfg.dataset.data_dir, el) for el in os.listdir(cfg.dataset.data_dir)[:10]]

        return

    def __create_graph(self):
        self.discriminator = Discriminator(cfg.train.discriminator_size)
        self.discriminator.build((None, cfg.train.image_size,  cfg.train.image_size, cfg.train.image_channels))

        self.siamese = Discriminator(cfg.train.siamese_size)
        self.siamese.build((None, cfg.train.image_size,  cfg.train.image_size, cfg.train.image_channels))

        self.unet = UNet()
        self.unet.build((None, cfg.train.image_size,  cfg.train.image_size, cfg.train.image_channels))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        self.BC_loss = tf.keras.losses.BinaryCrossentropy()
        self.max_margin_loss = MaxMarginLoss(cfg.train.delta)
        self.distance_loss = DistanceLoss()

        return

    def run_train_epoch(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.data_paths))
        dataset = dataset.shuffle(buffer_size=len(self.data_paths))
        dataset = dataset.map(map_func=self.data_augmentation.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size=cfg.train.batch_size).prefetch(buffer_size=cfg.train.prefetch_buffer_size)

        for image in dataset:
            with tf.GradientTape() as tape:
                image = tf.ones(shape=(cfg.train.batch_size, cfg.train.image_size,  cfg.train.image_size, cfg.train.image_channels))
                generated_image = self.unet(image, True)

                real_predictions = tf.sigmoid(self.discriminator(image, True))
                fake_predictions = tf.sigmoid(self.discriminator(generated_image, True))

                D_real = self.BC_loss(real_predictions, tf.ones_like(real_predictions))
                D_fake = self.BC_loss(fake_predictions, tf.zeros_like(fake_predictions))
                D_loss = D_real + D_fake

                G_adv = self.BC_loss(fake_predictions, tf.ones_like(fake_predictions))

                real_embeddings = self.siamese(image, True)
                fake_embeddings = self.siamese(generated_image, True)

                TraVeL_loss = self.distance_loss(real_embeddings, fake_embeddings)
                siamese_loss = self.max_margin_loss(real_embeddings, None)

                G_loss = G_adv + TraVeL_loss
                S_loss = siamese_loss + TraVeL_loss

            # grads = tape.gradient(loss_value, self.unet.trainable_variables)
            # self.optimizer.apply_gradients(zip(grads, self.unet.trainable_variables))


        return

gan = TraVeLGAN()
gan.run_train_epoch()

