from . import utils, kernels, acquisition
from .gp import ExactGP
from .vgp import vExactGP
from .vigp import viGP
from .spm import sPM
from .hypo import sample_next
from .bnn import DKL, viDKL, iBNN, vi_iBNN, viMTDKL
from .multitask import MultiTaskGP

from .__version__ import version as __version__

__all__ = ["utils", "kernels", "acquisition", "ExactGP", "vExactGP", "DKL",
           "viDKL", "iBNN", "vi_iBNN", "MultiTaskGP", "viMTDKL", "viGP", "sPM",
           "sample_next", "__version__"]
