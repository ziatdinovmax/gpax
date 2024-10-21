from .gp import ExactGP
from .vigp import viGP
from .hskgp import VarNoiseGP
from .spm import sPM
from .ibnn import iBNN
from .vi_ibnn import vi_iBNN
from .dkl import DKL
from .vidkl import viDKL
from .mtdkl import MultiTaskDKL
from .vi_mtdkl import viMultiTaskDKL
from .mtgp import MultiTaskGP
from .vi_mtgp import viMultiTaskGP
from .uigp import UIGP
from .mngp import MeasuredNoiseGP
from .vi_mngp import viMeasuredNoiseGP
from .linreg import LinReg
from .sparse_gp import viSparseGP
from .bnn import BNN

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
    "MultiTaskGP",
    "CoregGP",
    "UIGP",
    "LinReg",
    "MeasuredNoiseGP",
    "viMeasuredNoiseGP",
    "viSparseGP",
    "BNN",
    "viMultiTaskGP",
    "MultiTaskDKL",
    "viMultiTaskDKL",
]
