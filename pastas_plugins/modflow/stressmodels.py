from functools import lru_cache as lru_cache
from inspect import signature
from logging import getLogger
from pathlib import Path
from typing import Any

import flopy
import modflowapi
import numpy as np
import pandas as pd
from pandas import Series, Timestamp, concat, date_range
from pastas.decorators import conditional_cachedmethod
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
        model: Model,
        exe_name: str | Path,
        sim_ws: str | Path,
        tmin: Timestamp | None = None,
        tmax: Timestamp | None = None,
        silent: bool = True,
        raise_on_modflow_error: bool = False,
        solver_kwargs: dict[str, Any] | None = None,
        name: str = "mfsm",
        add_to_model: bool = True,
    ) -> None:
        if tmin is None:
            if model.settings["tmax"] is None:
                tmin = model.oseries.settings["tmin"] - model.settings["warmup"]
            else:
                tmin = model.settings["tmin"] - model.settings["warmup"]
        if tmax is None:
            tmax = (
                model.settings["tmax"]
                if model.settings["tmax"] is not None
                else model.oseries.settings["tmax"]
            )
        StressModelBase.__init__(
            self,
            name=name,
            tmin=tmin,
            tmax=tmax,
            rfunc=None,
        )
        self.model = model
        self.add_to_model = add_to_model  # add stressmodel to pastas model
        if "constant_d" in model.parameters.index:
            model.del_constant()
            logger.info(
                "Make sure to delete the model parameter constant_d "
                "(`model.del_constant('constant_d')`). Base elevation is now controlled by "
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
        if self.add_to_model:
            self.model.add_stressmodel(self)  # add as stressmodel to pastas model

    @property
    def nper(self) -> int:
        """Number of stress periods."""
        return len(date_range(self.tmin, self.tmax, freq=self.model.settings["freq"]))

    @property
    def nparam(self) -> int:
        """Number of parameters."""
        return len(self.parameters)

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
            [p.get_init_parameters(self.name) for nam, p in self._packages.items()],
            axis=0,
        )
        # drop constant_d duplicates
        if pdf.index.duplicated(keep="first").any():
            pdf = pdf[~pdf.index.duplicated(keep="first")]
        if "constant_d" in pdf.index:
            pdf.loc["constant_d", ["initial", "pmin", "pmax"]] = (
                self.model.oseries.series.mean(),
                self.model.oseries.series.min() - self.model.oseries.series.std(),
                self.model.oseries.series.max() + self.model.oseries.series.std(),
            )
        self.parameters = pdf

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

    def add_modflow_package(
        self, package: ModflowPackage | list[ModflowPackage]
    ) -> None:
        """Add a Modflow package to the model."""
        if isinstance(package, ModflowPackage):
            package = [package]

        for ipkg in package:
            if ipkg._name in self._packages:
                logger.warning(f"Package {ipkg._name} already exists. Overwriting it.")
            self._packages[ipkg._name] = ipkg
            ipkg_stress = ipkg.stress()
            if ipkg_stress is not None:
                # make sure the stresses are in the right time range
                for stress_name, stress_series in ipkg_stress.items():
                    ts = TimeSeries(stress_series, settings=stress_name)
                    ts.update_series(
                        tmin=self.tmin, tmax=self.tmax, freq=self.model.settings["freq"]
                    )
                    setattr(ipkg, stress_name, ts.series)

        self.set_init_parameters()
        # TODO: remove once initialization of pastas models is fixed
        if self.add_to_model:
            self.model.add_stressmodel(
                self, replace=True
            )  # add as stressmodel to pastas model

    def _remove_changing_package(self, package_name: str):
        """Remove a package from the model if it exists."""
        if package_name in self._gwf.get_package_list():
            self._gwf.remove_package(package_name)

    @conditional_cachedmethod(lambda self: self._cache)
    def get_sim_index(self) -> pd.DatetimeIndex:
        return date_range(
            start=self.tmin,
            end=self.tmax,
            freq=self.model.settings["freq"],
        )

    def simulate(
        self, p: ArrayLike, *args: Any, tmin=None, tmax=None, **kwargs
    ) -> Series:
        """Run the MODFLOW simulation and return the head time series."""
        s = Series(
            data=self.get_head(p=tuple(p)),
            index=self.get_sim_index(),
        )
        if tmin is not None:
            s = s.loc[tmin:]
        if tmax is not None:
            s = s.loc[:tmax]
        return s

    @conditional_cachedmethod(lambda self: self._cache)
    def get_head(self, p: tuple) -> np.ndarray:
        """Run the MODFLOW simulation and return the head values."""
        success, _ = self._run_simulation(p)
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

    def _run_simulation(self, p: tuple) -> None:
        self.update_model(p=p)
        return self._simulation.run_simulation(silent=self.silent)

    def update_model(self, p: ArrayLike) -> None:
        """Update the model with the given parameters."""
        p_series = Series(p, index=self.parameters.index)
        for name, package in self._packages.items():
            self._remove_changing_package(package_name=name)
            pnames = [
                f"{name}_{x}" if x != "d" else "constant_d"
                for x in self.package_parameter_names[name]
            ]
            p_dict = {k.rsplit("_", 1)[-1]: v for k, v in p_series.loc[pnames].items()}
            logger.debug(f"Updating package {name} with parameters {p_dict}")
            package.update_package(modflow_gwf=self._gwf, **p_dict)
        self._gwf.name_file.write()

    def to_dict(self) -> dict:
        raise NotImplementedError()


class ModflowModelApi(ModflowModel):
    _name = "ModflowModelApi"

    def __init__(
        self,
        model: Model,
        dll: str | Path,
        sim_ws: str | Path,
        tmin: Timestamp | None = None,
        tmax: Timestamp | None = None,
        silent: bool = True,
        raise_on_modflow_error: bool = False,
        solver_kwargs: dict[str, Any] | None = None,
        add_to_model: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            exe_name=dll,
            sim_ws=sim_ws,
            tmin=tmin,
            tmax=tmax,
            silent=silent,
            raise_on_modflow_error=raise_on_modflow_error,
            solver_kwargs=solver_kwargs,
            name="mfapi",
            add_to_model=add_to_model,
        )
        self.dll = dll

    def set_init_parameters(self) -> None:
        """Set the initial parameters back to their default values."""
        super().set_init_parameters()
        # build the modflow model with initial parameters,
        # this is fine since all relevant parameters will
        # be modified in the API loop
        self.initialize_model(p=self.parameters["initial"].values)

    def initialize_model(self, p: ArrayLike) -> None:
        """Create the model with the given parameters."""
        p_series = Series(p, index=self.parameters.index)
        for name, package in self._packages.items():
            pnames = [
                f"{name}_{x}" if x != "d" else "constant_d"
                for x in self.package_parameter_names[name]
            ]
            p_dict = {k.rsplit("_", 1)[-1]: v for k, v in p_series.loc[pnames].items()}
            package.update_package(modflow_gwf=self._gwf, **p_dict)
        # write nam file after initialization
        self._gwf.name_file.write()

    def update_static_parameters(self, mf6, p: np.ndarray):
        """Update static parameters in packages."""
        # modify params in packages
        p_series = Series(p, index=self.parameters.index)
        for name, ipkg in self._packages.items():
            if hasattr(ipkg, "update_parameters"):
                pnames = [
                    f"{name}_{x}" if x != "d" else "constant_d"
                    for x in self.package_parameter_names[name]
                ]
                p_tuple = tuple(p_series.loc[pnames].tolist())
                ipkg.update_parameters(mf6, p_tuple)

    def update_timeseries(self, mf6, kper):
        """Update time series in packages."""
        for ipkg in self._packages.values():
            if hasattr(ipkg, "update_timeseries"):
                ipkg.update_timeseries(mf6, kper)

    def _run_simulation(self, p: tuple) -> bool:
        store_head = False  # about 1s slower when storing head
        if store_head:
            head = np.zeros(self.nper, dtype=float)
        # start the API fun
        mf6 = modflowapi.ModflowApi(self.dll, working_directory=self.sim_ws)
        mf6.initialize()

        success = False

        # time loop
        current_time = mf6.get_current_time()
        end_time = mf6.get_end_time()

        # maximum outer iterations
        max_iter = mf6.get_value(mf6.get_var_address("MXITER", "SLN_1"))

        # pre-compute recharge, this is faster than computing it every time step
        # TODO: maybe find a more robust way to find parameter RCH_f
        try:
            ipar = self.parameters.index.tolist().index("RCH_f")  # position in list
            rch = self._packages["RCH"]
            rch.compute_recharge(f=p[ipar])
        except ValueError:
            # no rch
            pass

        # model time loop
        kper = 0
        while current_time < end_time:
            # get dt and prepare for non-linear iterations
            dt = mf6.get_time_step()
            mf6.prepare_time_step(dt)

            # update static packages only at start of simulation
            if kper == 0:
                self.update_static_parameters(mf6=mf6, p=p)

            # update time series packages
            # NOTE: it would presumably be faster to rewrite the time series
            # file once the bug concerning the modflow API and time series is resolved
            # (see https://github.com/MODFLOW-ORG/modflowapi/issues/85)
            self.update_timeseries(mf6, kper)

            # convergence loop
            kiter = 0
            mf6.prepare_solve()
            while kiter < max_iter:
                # solve
                has_converged = mf6.solve()
                kiter += 1

                if has_converged:
                    break

            # finalize solve
            mf6.finalize_solve()

            # finalize time step and update time
            mf6.finalize_time_step()
            current_time = mf6.get_current_time()

            # terminate if model did not converge
            if not has_converged:
                break

            if store_head:
                head[kper] = mf6.get_value_ptr(f"{self.name.upper()}/X").item()

            # increment counter
            kper += 1

        # cleanup
        try:
            mf6.finalize()
            success = True
        except Exception as e:
            return success, e
        if store_head:
            return success, head
        else:
            return success, None


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
