from datetime import datetime

# model settings
model = dict(
    type='TraVeLGAN',
    noise_size=100,
    discriminator=dict(
        type='Discriminator',
        loss_type=dict(type='BinaryCrossentropy'),
        norm_cfg=dict(type='GroupNorm')),
    generator=dict(
        type='UNet',
        loss_type=dict(type='BinaryCrossentropy'),
        norm_cfg=dict(type='GroupNorm'),
        # type='Generator',
        # dense_in_shape=100,
        # dense_out_shape=8 * 8 * 256,
        # reshape_shape=(8, 8, 256),
        # conv_shapes=[(256, 3, 3, 2, 2),
        #              (128, 3, 3, 2, 2),
        #              (64, 3, 3, 2, 2),
        #              (3, 3, 3, 2, 2)],
        activation='tanh'),
    siamese_network=dict(
        type='Discriminator',
        output_size=1000,
        norm_cfg=dict(type='GroupNorm')),
    distance_loss=dict(
        type='DistanceLoss'),
    siamese_loss=dict(
        type='MaxMarginLoss',
        delta=0.7),
)
# model training and testing settings
train_cfg = dict(
    image_size=128,
    image_channels=3)
test_cfg = dict(
    image_size=128,
    image_channels=3)

# dataset settings
data_loader_type = 'TensorSlicesDataset'
data_loader_chain_rule = {
    # 'shuffle': {'buffer_size': 10000,
    #             'reshuffle_each_iteration': True},
    'map': {'num_parallel_calls': 1},
    'batch': {'batch_size': 4},
}

dataset_type = 'DeepFashion2Dataset'
data_root = '/home/firiuza/MachineLearning/DeepFashion2/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=False, with_mask=False, with_pair_id=True),
    dict(type='Resize', img_scale=(128, 128), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'gt_labels', 'gt_pair_id']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(128, 128), keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'splits_annotations/train_deepfashion2.json',
        img_prefix=data_root + 'train/image',
        pipeline=train_pipeline),
    valid=dict(
        type=dataset_type,
        ann_file=data_root + 'splits_annotations/validation_deepfashion2.json',
        img_prefix=data_root + 'validation/image',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'splits_annotations/train_deepfashion2.json',
        img_prefix=data_root + 'validation/image',
        pipeline=test_pipeline,
        test_mode=True)
    )
data_loader = dict(
        train=dict(
            type=data_loader_type,
            ops_chain=data_loader_chain_rule,
            map_func_name='prepare_train_img'
        )
)
# learning policy
lr_schedule = dict(
    initial_learning_rate=2e-6,
    decay_steps=10000,
    decay_rate=0.99,
    staircase=True)
# optimizer
optimizer = dict(
    type='TraVelGANOptimizer',
    optimizer_cfg=dict(
        type='Adam',
        params=None,
        lr_schedule_type='ExponentialDecay',
        lr_schedule=lr_schedule)
)

use_TensorBoard=True

# runtime settings
total_epochs = 100
log_level = 'INFO'

work_dir = '/home/firiuza/PycharmProjects/TraVeLGAN/run_models/run_%s' % (datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))

restore_model_path = '/home/firiuza/PycharmProjects/TraVeLGAN/run_models/run_20200711-152758/models/model-50000.h5'
workflow = [('train', 1)]
