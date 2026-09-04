# ruff: noqa: F401
import pastas as ps

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

ps.stressmodels.ModflowModel = ModflowModel
