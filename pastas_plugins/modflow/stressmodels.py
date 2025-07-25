import functools
from logging import getLogger
from pathlib import Path
from typing import Any, Optional, Tuple

import flopy
import numpy as np
from pandas import DataFrame, Series
from pastas.model import Model
from pastas.stressmodels import StressModelBase
from pastas.typing import ArrayLike, TimestampType

from .modflow import ModflowPackage

logger = getLogger(__name__)


class ModflowModel(StressModelBase):
    _name = "ModflowModel"

    def __init__(
        self,
        ml: Model,
        exe_name: str | Path,
        sim_ws: str | Path,
        silent: bool = True,
        raise_on_modflow_error: bool = False,
        solver_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.ml = ml
        self.constant_d_from_modflow = True
        if "constant_d" in ml.parameters.index:
            ml.del_constant()
            logger.info(
                "Make sure to delete the model parameter constant_d "
                "(`ml.del_constant('constant_d')`). Base elevation is now controlled by "
                "parameter `_d`."
            )
            self.constant_d_from_modflow = True
        self.exe_name = exe_name
        self.sim_ws = sim_ws
        self.raise_on_modflow_error = raise_on_modflow_error
        self.solver_kwargs = (
            dict(
                complexity="SIMPLE",
                outer_dvclose=1e-2,
                inner_dvclose=1e-2,
                rcloserecord=1e-1,
                linear_acceleration="BICGSTAB",
            )
            if solver_kwargs is None
            else solver_kwargs
        )
        self.silent = silent
        self._packages: dict[str, ModflowPackage] | None = None
        self._simulation: flopy.mf6.MFSimulation | None = None
        self._gwf: flopy.mf6.ModflowGwf | None = None
        self._nper = None
        StressModelBase.__init__(
            self,
            name=ml.name,
            tmin=self.ml.settings["tmin"],
            tmax=self.ml.settings["tmax"],
        )

        self.stress = [self.prec, self.evap]

        self.freq = self.prec.settings["freq"]
        self.set_init_parameters()

    def add_modflow_package(
        self, package: ModflowPackage | list[ModflowPackage]
    ) -> None:
        """Add a Modflow package to the model."""
        if self._packages is None:
            self._packages = {}
        if isinstance(package, ModflowPackage):
            package = [package]

        for p in package:
            if p._name in self._packages:
                logger.warning(f"Package {p._name} already exists. Overwriting it.")
            self._packages[p._name] = p

    def set_init_parameters(self) -> None:
        """Set the initial parameters back to their default values."""
        self.parameters = self.get_init_parameters(self.name)

    def to_dict(self, series: bool = True) -> dict:
        raise NotImplementedError()

    def get_stress(
        self,
        p: Optional[ArrayLike] = None,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        freq: Optional[str] = None,
        istress: int = 0,
        **kwargs,
    ) -> Tuple[Series, Series]:
        raise NotImplementedError()

    def simulate(
        self,
        p: ArrayLike,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
    ):
        h = self.get_head(p=p)
        return Series(
            data=h,
            index=self.ml.oseries.index,
            name=self.name,
        )

    def _get_block(self, p, dt, tmin, tmax):
        """Internal method to get the block-response function.
        Cannot be used (yet?) since there is no block response
        """
        # prec = np.zeros(len())
        # evap = np.zeros()
        # return modflow.simulate(np.mean(prec))
        raise NotImplementedError(
            "Block response function is not implemented for ModflowModel."
        )

    @functools.lru_cache(maxsize=5)
    def get_head(self, p):
        self.update_model(p=p)
        success, _ = self._simulation.run_simulation(silent=self.silent)
        if success:
            return self._gwf.output.head().get_ts((0, 0, 0))[:, 1]
        else:
            logger.error("ModflowError: model run failed with parameters: %s" % str(p))
            if self.raise_on_modflow_error:
                raise Exception(
                    "Modflow run failed. Check the LIST file for more information."
                )
            else:
                return np.zeros(self._nper)

    def setup_modflow_simulation(self) -> None:
        sim = flopy.mf6.MFSimulation(
            sim_name=self._name,
            version="mf6",
            exe_name=self.exe_name,
            sim_ws=self.sim_ws,
            lazy_io=True,
        )

        _ = flopy.mf6.ModflowTdis(
            sim,
            time_units="DAYS",
            nper=self._nper,
            perioddata=[(1, 1, 1) for _ in range(self._nper)],
        )

        gwf = flopy.mf6.ModflowGwf(
            sim,
            modelname=self._name,
            newtonoptions=["NEWTON"],
        )

        _ = flopy.mf6.ModflowIms(
            sim,
            **self.solver_kwargs,
        )

        _ = flopy.mf6.ModflowGwfnpf(
            self._gwf, save_flows=False, icelltype=0, pname="npf"
        )

        _ = flopy.mf6.ModflowGwfoc(
            self._gwf,
            head_filerecord=f"{self._gwf.name}.hds",
            saverecord=[("HEAD", "ALL")],
        )

        sim.write_simulation(silent=self.silent)

        if not self.constant_d_from_modflow:
            self.update_dis(d=0.0, height=1.0)
            self.update_ic(d=0.0)

        return sim, gwf

    def _remove_changing_package(self, package_name: str):
        if package_name in self._gwf.get_package_list():
            self._gwf.remove_package(package_name)

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"]
        )
        if self.constant_d_from_modflow:
            parameters.loc[name + "_d"] = (
                float(np.mean(self._head)),
                min(self._head),
                max(self._head),
                True,
                name,
                "uniform",
            )
        parameters.loc[name + "_S"] = (0.05, 0.001, 0.5, True, name, "uniform")
        return parameters

    def update_dis(self, d: float, height: float = 1.0):
        self._remove_changing_package("DIS")
        dis = flopy.mf6.ModflowGwfdis(
            self._gwf,
            length_units="METERS",
            nlay=1,
            nrow=1,
            ncol=1,
            delr=1,
            delc=1,
            top=d + height,
            botm=d,
            idomain=1,
            pname="dis",
        )
        dis.write()

    def update_ic(self, d: float):
        self._remove_changing_package("IC")
        ic = flopy.mf6.ModflowGwfic(self._gwf, strt=d, pname="ic")
        ic.write()

    def update_sto(self, s: float):
        self._remove_changing_package("STO")
        haq = (self._gwf.dis.top.array - self._gwf.dis.botm.array)[0]
        sto = flopy.mf6.ModflowGwfsto(
            self._gwf,
            save_flows=False,
            iconvert=0,
            ss=s / haq,
            sy=0.0,  # just to show the specific yield is not needed
            transient=True,
            pname="sto",
        )
        sto.write()


solver_kwargs_uzf = dict(
    print_option="summary",
    outer_dvclose=3e-2,
    outer_maximum=300,
    under_relaxation="dbd",
    linear_acceleration="BICGSTAB",
    under_relaxation_theta=0.7,
    under_relaxation_kappa=0.08,
    under_relaxation_gamma=0.05,
    under_relaxation_momentum=0.0,
    inner_dvclose=3e-2,
    rcloserecord="1000.0 strict",
    inner_maximum=500,
    relaxation_factor=0.97,
    number_orthogonalizations=2,
    preconditioner_levels=8,
    preconditioner_drop_tolerance=0.001,
)
