from functools import lru_cache as lru_cache
from inspect import signature
from logging import getLogger
from pathlib import Path
from typing import Any

import flopy
import numpy as np
from pandas import Series, Timestamp, concat, date_range
from pastas.model import Model
from pastas.stressmodels import StressModelBase
from pastas.timeseries import TimeSeries
from pastas.typing import ArrayLike

from .modflow import ModflowDis, ModflowIc, ModflowPackage, ModflowSto

logger = getLogger(__name__)


class ModflowModel(StressModelBase):
    _name = "ModflowModel"

    def __init__(
        self,
        ml: Model,
        exe_name: str | Path,
        sim_ws: str | Path,
        tmin: Timestamp | None = None,
        tmax: Timestamp | None = None,
        silent: bool = True,
        raise_on_modflow_error: bool = False,
        solver_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if tmin is None:
            if ml.settings["tmax"] is None:
                tmin = ml.oseries.settings["tmin"] - ml.settings["warmup"]
            else:
                tmin = ml.settings["tmin"] - ml.settings["warmup"]
        if tmax is None:
            tmax = (
                ml.settings["tmax"]
                if ml.settings["tmax"] is not None
                else ml.oseries.settings["tmax"]
            )
        StressModelBase.__init__(
            self,
            name="mfsm",
            tmin=tmin,
            tmax=tmax,
            rfunc=None,
        )
        self.ml = ml
        if "constant_d" in ml.parameters.index:
            ml.del_constant()
            logger.info(
                "Make sure to delete the model parameter constant_d "
                "(`ml.del_constant('constant_d')`). Base elevation is now controlled by "
                "parameter `_d`."
            )
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
        self._packages: dict[str, ModflowPackage] = {
            "DIS": ModflowDis(),
            "IC": ModflowIc(),
            "STO": ModflowSto(),
        }
        self._simulation, self._gwf = self.setup_modflow_simulation()

    @property
    def nper(self) -> int:
        """Number of stress periods."""
        return len(date_range(self.tmin, self.tmax, freq=self.ml.settings["freq"]))

    @property
    def nparam(self) -> int:
        """Number of parameters."""
        return len(self.parameters)

    def add_modflow_package(
        self, package: ModflowPackage | list[ModflowPackage]
    ) -> None:
        """Add a Modflow package to the model."""
        if isinstance(package, ModflowPackage):
            package = [package]

        for pack in package:
            if pack._name in self._packages:
                logger.warning(f"Package {pack._name} already exists. Overwriting it.")
            self._packages[pack._name] = pack
            pack_stress = pack.stress()
            if pack_stress is not None:
                # make sure the stresses are in the right time range
                for stress_name, stress_series in pack_stress.items():
                    ts = TimeSeries(stress_series, settings=stress_name)
                    ts.update_series(
                        tmin=self.tmin, tmax=self.tmax, freq=self.ml.settings["freq"]
                    )
                    setattr(pack, stress_name, ts.series)

        self.set_init_parameters()

    @property
    def package_parameter_names(self) -> dict[str, list[str]]:
        """Get the parameter names of the packages."""
        sigdict = {
            name: [
                x
                for x in signature(package.update_package).parameters
                if x != "modflow_gwf"
            ]
            for name, package in self._packages.items()
        }
        return sigdict

    def set_init_parameters(self) -> None:
        """Set the initial parameters back to their default values."""
        pdf = concat(
            [p.get_init_parameters(self.name) for p in self._packages.values()], axis=0
        )
        if f"{self.name}_d" in pdf.index:
            pdf.loc[f"{self.name}_d", ["initial", "pmin", "pmax"]] = (
                self.ml.oseries.series.mean(),
                self.ml.oseries.series.min() - self.ml.oseries.series.std(),
                self.ml.oseries.series.max() + self.ml.oseries.series.std(),
            )
        if pdf.index.duplicated(keep="first").any():
            pdf = pdf[~pdf.index.duplicated(keep="first")]
        self.parameters = pdf

    def to_dict(self) -> dict:
        raise NotImplementedError()

    def simulate(self, p: ArrayLike, *args: Any) -> Series:
        """Run the MODFLOW simulation and return the head time series."""
        h = self.get_head(p=p)
        return Series(
            data=h,
            index=date_range(
                start=self.tmin,
                end=self.tmax,
                freq=self.ml.settings["freq"],
            ),
        )

    # @lru_cache(maxsize=None)
    def get_head(self, p: ArrayLike) -> np.ndarray:
        """Run the MODFLOW simulation and return the head values."""
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
                return np.zeros(self.nper)

    def update_model(self, p: ArrayLike) -> None:
        """Update the model with the given parameters."""
        p_series = Series(p, index=self.parameters.index)
        for name, package in self._packages.items():
            self._remove_changing_package(package_name=name)
            pnames = [f"{self.name}_{x}" for x in self.package_parameter_names[name]]
            p_dict = {k.rsplit("_", 1)[-1]: v for k, v in p_series.loc[pnames].items()}
            package.update_package(modflow_gwf=self._gwf, **p_dict)
        self._gwf.name_file.write()

    def setup_modflow_simulation(
        self,
    ) -> tuple[flopy.mf6.MFSimulation, flopy.mf6.ModflowGwf]:
        """Set up the MODFLOW simulation."""
        sim = flopy.mf6.MFSimulation(
            sim_name=self.name,
            version="mf6",
            exe_name=self.exe_name,
            sim_ws=self.sim_ws,
            lazy_io=True,
        )

        _ = flopy.mf6.ModflowTdis(
            sim,
            time_units="DAYS",
            nper=self.nper,
            perioddata=[(1, 1, 1) for _ in range(self.nper)],
        )

        gwf = flopy.mf6.ModflowGwf(
            sim,
            modelname=self.name,
            newtonoptions=["NEWTON"],
        )

        _ = flopy.mf6.ModflowIms(
            sim,
            **self.solver_kwargs,
        )

        _ = flopy.mf6.ModflowGwfnpf(gwf, save_flows=False, icelltype=0, pname="npf")

        _ = flopy.mf6.ModflowGwfoc(
            gwf,
            head_filerecord=f"{gwf.name}.hds",
            saverecord=[("HEAD", "ALL")],
        )

        sim.write_simulation(silent=self.silent)
        return sim, gwf

    def _remove_changing_package(self, package_name: str):
        """Remove a package from the model if it exists."""
        if package_name in self._gwf.get_package_list():
            self._gwf.remove_package(package_name)


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
