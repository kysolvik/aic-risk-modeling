"""aic-risk-modeling: subpackages preprocess, train, eval, predict.

Subpackages load lazily (PEP 562) so the bare import doesn't pull in torch/TF.
"""

__all__ = ["train", "eval", "preprocess", "predict"]

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import train, eval, preprocess, predict  # type: ignore

def __getattr__(name: str):
    if name in __all__:
        module =importlib.import_module("." + name, __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
