"""Top-level package for aic-risk-modeling.

Subpackages:
- `train` (training utilities and data loaders)

Uses __getattr__ to do lazy submodule imports (according to PEP 562) to
reduce imports of heavy optional dependencies (e.g., `tensorflow-data-validation`).
"""

__all__ = ["train"]

# Lazily import subpackages on attribute access (PEP 562)
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Help type checkers see these names without importing heavy deps at runtime
    from . import train  # type: ignore
    from . import preprocess  # type: ignore


def __getattr__(name: str):
    """Lazily import subpackages when accessed as attributes.

    Example: `aic_risk_modeling.train` will import `aic_risk_modeling.train` on
    first access but `import aic_risk_modeling` will not import it.
    """
    if name in __all__:
        module =importlib.import_module("." + name, __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Expose lazy subpackages in help()/dir()."""
    return sorted(list(globals().keys()) + __all__)
