from .registry import GENERATOR, DISCRIMINATOR, LOSSES, \
    OPTIMIZERS, GAN, NORMALIZATION

from .discriminator import *
from .GAN import *
from .generator import *
from .losses import *
from .normalization import *
from .optimizers import *

__all__ = [
    'GENERATOR', 'DISCRIMINATOR', 'LOSSES', 'OPTIMIZERS', 'GAN', 'NORMALIZATION'
]