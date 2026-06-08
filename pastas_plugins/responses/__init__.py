# ruff : noqa: F401
# ruff : noqa: F401
import pastas as ps

from pastas_plugins.responses.rfunc import Edelman, Theis
from pastas_plugins.responses.version import __version__

ps.rfunc.Edelman = Edelman
ps.rfunc.Theis = Theis
