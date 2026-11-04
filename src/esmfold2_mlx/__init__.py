"""ESMFold2 on Apple silicon (unofficial): MLX/Metal trunk and sampler, 8-bit ESMC-6B.

    from esmfold2_mlx import Folder
    f = Folder("6B")                 # load + warm-up (once)
    res = f.fold("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG", "ubiquitin")
    res["pdb"], res["plddt"], res["ptm"]

Importing the package is cheap; torch and MLX load when you create a Folder.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("esmfold2-mlx")
except PackageNotFoundError:  # running from a source tree
    __version__ = "0+unknown"

__all__ = ["Folder", "__version__"]


def __getattr__(name):
    if name == "Folder":
        from .serve import Folder

        return Folder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
