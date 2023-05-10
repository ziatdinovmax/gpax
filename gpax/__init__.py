from . import utils, kernels, acquisition
from .gp import ExactGP
from .vgp import vExactGP
from .dkl import DKL
from .vidkl import viDKL
from vigp import viGP
from .spm import sPM
from .hypo import sample_next

from .__version__ import version as __version__

__all__ = ["utils", "kernels", "acquisition", "ExactGP", "vExactGP", "DKL",
           "viDKL", "viGP", "sPM", "sample_next", "__version__"]
