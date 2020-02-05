from easydict import EasyDict

cfg = EasyDict()

cfg.dataset = EasyDict()

cfg.dataset.data_dir = '/home/firiuza/MachineLearning/celebA/img_align_celeba/'

cfg.train = EasyDict()

cfg.train.batch_size = 4
cfg.train.prefetch_buffer_size = cfg.train.batch_size
# cfg.train.shuffle_buffer_size
cfg.train.image_size = 128
cfg.train.image_channels = 3

cfg.train.discriminator_size = 1
cfg.train.siamese_size = 128

cfg.train.image_average = 127
cfg.train.image_std = 27

cfg.train.delta = 1.