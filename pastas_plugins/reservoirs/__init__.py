# ruff: noqa: F401
import pastas as ps

from pastas_plugins.reservoirs.reservoir import Reservoir1, Reservoir2
from pastas_plugins.reservoirs.stressmodels import ReservoirModel
from pastas_plugins.reservoirs.version import __version__

ps.stressmodel.ReservoirModel = ReservoirModel
