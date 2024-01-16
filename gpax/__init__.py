from .__version__ import version as __version__
from . import utils
from . import kernels
from . import acquisition
from .hypo import sample_next
from .models import (DKL, CoregGP, ExactGP, MultiTaskGP, iBNN, vExactGP,
                     vi_iBNN, viDKL, viGP, viMTDKL, VarNoiseGP, UIGP, MeasuredNoiseGP)

__all__ = ["utils", "kernels", "mtkernels", "acquisition", "ExactGP", "vExactGP", "DKL",
           "viDKL", "iBNN", "vi_iBNN", "MultiTaskGP", "viMTDKL", "viGP", "sPM", "VarNoiseGP",
           "UIGP", "MeasuredNoiseGP", "CoregGP", "sample_next", "__version__"]
