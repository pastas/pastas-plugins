# ruff: noqa: F401
import logging as _logging

from pastas_plugins.modflow.modflow import (
    ModflowDrn,
    ModflowGhb,
    ModflowRch,
    ModflowStoConfined,
    ModflowStoPhreatic,
    ModflowUzf,
)
from pastas_plugins.modflow.stressmodels import ModflowModel, ModflowModelApi
from pastas_plugins.modflow.version import __version__


def set_log_level(level: int | str) -> None:
    """Set the logging verbosity for all pastas_plugins.modflow modules.

    Parameters
    ----------
    level : int or str
        A standard Python logging level.  Common choices:

        * ``logging.DEBUG``   (10) – per-period parameter values, geometry sync
        * ``logging.INFO``    (20) – one summary line per simulation run
        * ``logging.WARNING`` (30) – silent during calibration (default)

    Examples
    --------
    >>> import logging
    >>> from pastas_plugins.modflow import set_log_level
    >>> set_log_level(logging.DEBUG)   # verbose debugging
    >>> set_log_level(logging.WARNING) # quiet during calibration
    """
    _logging.getLogger("pastas_plugins.modflow").setLevel(level)
