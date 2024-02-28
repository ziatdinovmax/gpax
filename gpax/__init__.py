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


import dunamai as _dunamai

__version__ = _dunamai.get_version("gpax", third_choice=_dunamai.Version.from_any_vcs).serialize()
del _dunamai
