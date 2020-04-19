from .coco import CocoDataset
from .base_dataset import BaseDataset
from .deep_fashion2 import DeepFashion2Dataset
from .registry import DATASETS, PIPELINES, DATA_LOADER
from .builder import build_dataset, build_data_loader
from.tf_loader import TensorSlicesDataset

__all__ = [
    'CocoDataset', 'BaseDataset', 'DeepFashion2Dataset',
    'DATASETS', 'PIPELINES', 'DATA_LOADER', 'build_dataset',
    'build_data_loader', 'TensorSlicesDataset'
]
