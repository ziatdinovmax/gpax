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
]

# DO NOT CHANGE BELOW ---------------------------------------------------------
# ALSO DO NOT TYPE dunder version ANYWHERE ABOVE THIS
# This is replaced at build time automatically during deployment and
# installation. Replacing anything will mess that up and crash the entire
# build.
__version__ = ...  # semantic-version-placeholder
# DO NOT CHANGE ABOVE ---------------------------------------------------------

# Silly hack. Useful for local development
if __version__ == ...:
    try:
        from dunamai import Version
        version = Version.from_any_vcs()
        __version__ = version.serialize()
    except ImportError:
        print("You are running a local copy of gpax (not installed via pip)")
        print("__version__ = ...; pip install dunamai to track local version")
        pass
    
