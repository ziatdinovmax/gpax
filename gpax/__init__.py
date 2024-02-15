from . import acquisition, kernels, utils
from .hypo import sample_next
from .models import (DKL, UIGP, CoregGP, ExactGP, MeasuredNoiseGP, MultiTaskGP,
                     VarNoiseGP, iBNN, sPM, vExactGP, vi_iBNN, viDKL, viGP,
                     viMTDKL, viSparseGP)

__all__ = [
    "utils",
    "kernels",
    "mtkernels",
    "acquisition",
    "ExactGP",
    "vExactGP",
    "DKL",
    "viDKL",
    "iBNN",
    "vi_iBNN",
    "MultiTaskGP",
    "viMTDKL",
    "viGP",
    "sPM",
    "VarNoiseGP",
    "UIGP",
    "MeasuredNoiseGP",
    "viSparseGP",
    "CoregGP",
    "sample_next",
    "__version__",
]

# DO NOT CHANGE BELOW ---------------------------------------------------------
# This is replaced at build time automatically during deployment and
# installation. Replacing anything will mess that up and crash the entire
# build.
__version__ = ...  # semantic-version-placeholder
# DO NOT CHANGE ABOVE ---------------------------------------------------------
