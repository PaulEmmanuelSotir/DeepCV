
# Lists all available 'deepcv' sub-modules
__version__ = '0.0.2'
__all__ = ['detection', 'classification', 'keypoints', 'meta', 'stabilize_WIP', 'stitching_WIP', 'synchronization_WIP',
           'pipeline', 'run', 'utils', 'hooks']

# 'deepcv' modules imports
from . import utils
from . import hooks
from . import pipeline
from . import run

# default 'deepcv' subpackages imports
from . import meta
