
# Lists all available 'deepcv.meta.data' sub-modules
__all__ = ['augmentation', 'datasets', 'preprocess', 'singan', 'training_metadata', 'viz']

# TODO: remove or put `compression` back into __all__

# 'deepcv.meta.data' modules imports
from . import training_metadata
from .datasets import *
from . import singan
from . import viz
# from . import augmentation # Augmentation needs 'deepcv.meta.hyperparams' which cant be imported before 'deepcv.meta.data'
# from . import preprocess # Preprocess needs 'deepcv.meta.hyperparams' which cant be imported before 'deepcv.meta.data'

# TODO: remove or put `import deepcv.meta.data.compression` back
