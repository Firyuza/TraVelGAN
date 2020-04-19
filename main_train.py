import argparse
import os

import tensorflow as tf

from utils.config import Config
from utils.registry import build_from_cfg
from datasets.builder import build_dataset, build_data_loader

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', default='/home/firiuza/PycharmProjects/TraVeLGAN/configs/travel_gan_config.py',
                        help='train config file path')
    parser.add_argument('--work_dir', default='', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', default='', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        default='',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        default='',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        default='',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)

    dataset = build_dataset(cfg.data.train)
    data_loader = build_data_loader(cfg.data_loader.train, default_args={'dataset': dataset})
    for images, labels in data_loader.data_loader:
        print(labels)