# ruff : noqa: F401
import pastas as ps

from pastas_plugins.pest.solver import (
    PestGlmSolver,
    PestHpSolver,
    PestIesSolver,
    PestSenSolver,
    RandomizedMaximumLikelihoodSolver,
)

ps.solver.PestGlmSolver = PestGlmSolver
ps.solver.PestHpSolver = PestHpSolver
ps.solver.PestIesSolver = PestIesSolver
ps.solver.PestSenSolver = PestSenSolver
ps.solver.RandomizedMaximumLikelihoodSolver = RandomizedMaximumLikelihoodSolver
