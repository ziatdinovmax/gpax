from gpax._version import __version__

from . import acquisition, kernels, utils
from .hypo import sample_next
from .models import (
    DKL,
    UIGP,
    CoregGP,
    ExactGP,
    MeasuredNoiseGP,
    MultiTaskGP,
    VarNoiseGP,
    iBNN,
    sPM,
    vExactGP,
    vi_iBNN,
    viDKL,
    viGP,
    viMTDKL,
    viSparseGP,
)

# WARNING: I don't think __all__ is really needed, and I'm not sure it's
# actually accurate
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
]
