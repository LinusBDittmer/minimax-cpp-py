from importlib.metadata import PackageNotFoundError, version

from ._minimax_core import (
    laplace_minimax,
    laplace_lp,
    DenominatorDensity,
    biased_laplace,
)

try:
    __version__ = version("minimax-cpppy")
except PackageNotFoundError:  # not installed (e.g. run from build tree)
    __version__ = "0.0.0+unknown"

__all__ = ["laplace_minimax", "laplace_lp", "DenominatorDensity",
           "biased_laplace", "__version__"]
