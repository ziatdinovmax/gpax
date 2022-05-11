from . import utils, kernels, acquisition
from .gp import ExactGP
from .vgp import vExactGP
from .dkl import DKL
from .vidkl import viDKL
from .spm import sPM

from .__version__ import version as __version__

__all__ = ["utils", "kernels", "acquisition", "ExactGP", "vExactGP", "DKL", "viDKL", "sPM", "__version__"]
