from .gp import ExactGP
from .vgp import vExactGP
from .vigp import viGP
from .hskgp import VarNoiseGP
from .spm import sPM
from .ibnn import iBNN
from .vi_ibnn import vi_iBNN
from .dkl import DKL
from .vidkl import viDKL
from .vi_mtdkl import viMTDKL
from .mtgp import MultiTaskGP
from .corgp import CoregGP
from .uigp import UIGP
from .mngp import MeasuredNoiseGP
from .linreg import LinReg

__all__ = [
    "ExactGP",
    "vExactGP",
    "viGP",
    "VarNoiseGP",
    "sPM",
    "iBNN",
    "vi_iBNN",
    "DKL",
    "viDKL",
    "viMTDKL",
    "MultiTaskGP",
    "CoregGP",
    "UIGP",
    "LinReg",
    "MeasuredNoiseGP"
]
