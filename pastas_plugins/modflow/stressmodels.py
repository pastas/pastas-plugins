import logging
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

from .modflow import (
    ModflowDis,
    ModflowIc,
    ModflowPackage,
    ModflowStoConfined,
    ModflowStoPhreatic,
)

logger = getLogger(__name__)


class ModflowModel(StressModelBase):
    """MODFLOW-based stress model for pastas.

    Wraps a single-cell MODFLOW 6 groundwater model and exposes it as a
    pastas ``StressModelBase``, enabling MODFLOW-calibrated heads to be
    used directly in pastas time-series analysis.

    Parameters
    ----------
    model : pastas.Model
        The pastas model instance.
    exe_name : str or Path
        Path to the MODFLOW 6 executable.
    sim_ws : str or Path
        Working directory for the MODFLOW simulation files.
    tmin : Timestamp, optional
        Start of the simulation period. Defaults to model ``tmin``
        minus warmup.
    tmax : Timestamp, optional
        End of the simulation period. Defaults to model ``tmax``.
    silent : bool, optional
        Suppress MODFLOW console output. Default is ``True``.
    raise_on_modflow_error : bool, optional
        Raise an exception if the MODFLOW run fails. Default is
        ``False``.
    solver_kwargs : dict, optional
        Keyword arguments passed to ``flopy.mf6.ModflowIms``. If
        ``None``, a default SIMPLE solver configuration is used.
    name : str, optional
        Stress-model name. Default is ``"mfsm"``.
    save_heads : bool, optional
        Write head output to a ``.hds`` file. Default is ``True``.
    newtonoptions : str or None, optional
        Newton-Raphson linearization options passed to
        ``ModflowGwf``.
    """

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
        save_heads: bool = True,
        newtonoptions=None,
        iconvert: int = 0,
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
        self.save_heads = save_heads
        if "constant_d" in model.parameters.index:
            model.del_constant()
            logger.info(
                "Deleted the model constant. Base elevation "
                "is now controlled by a new MODFLOW parameter `constant_d`."
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
                linear_acceleration="CG",
            )
            if solver_kwargs is None
            else solver_kwargs
        )
        self.silent = silent
        self._packages: dict[str, ModflowPackage] = {
            "DIS": ModflowDis(),
            "IC": ModflowIc(),
            "STO": ModflowStoConfined() if iconvert == 0 else ModflowStoPhreatic(),
        }
        self._simulation, self._gwf = self.setup_modflow_simulation(
            newtonoptions=newtonoptions
        )

    @property
    def nper(self) -> int:
        """Number of stress periods.

        Returns
        -------
        int
            Number of periods in the simulation date range.
        """
        return len(date_range(self.tmin, self.tmax, freq=self.model.settings["freq"]))

    @property
    def nparam(self) -> int:
        """Number of parameters.

        Returns
        -------
        int
            Total number of calibration parameters.
        """
        return len(self.parameters)

    @property
    def package_parameter_names(self) -> dict[str, list[str]]:
        """Return calibration parameter names for each package.

        Returns
        -------
        dict of str to list of str
            Mapping from package name to list of parameter names,
            excluding ``"modflow_gwf"``.
        """
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
        if pdf.index.duplicated().any():
            pdf = pdf[~pdf.index.duplicated(keep="last")]
        if "constant_d" in pdf.index:
            pdf.loc["constant_d", ["initial", "pmin", "pmax"]] = (
                self.model.oseries.series.mean(),
                self.model.oseries.series.min() - self.model.oseries.series.std(),
                self.model.oseries.series.max() + self.model.oseries.series.std(),
            )
        self.parameters = pdf

    def setup_modflow_simulation(
        self,
        newtonoptions=None,
    ) -> tuple[flopy.mf6.MFSimulation, flopy.mf6.ModflowGwf]:
        """Set up the MODFLOW 6 simulation and write input files.

        Parameters
        ----------
        newtonoptions : str or None, optional
            Newton-Raphson linearization options passed to the GWF
            model. Default is ``None``.

        Returns
        -------
        sim : flopy.mf6.MFSimulation
            The configured simulation object.
        gwf : flopy.mf6.ModflowGwf
            The groundwater-flow model object.
        """
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
            newtonoptions=newtonoptions,
        )

        _ = flopy.mf6.ModflowIms(
            sim,
            **self.solver_kwargs,
        )

        _ = flopy.mf6.ModflowGwfnpf(gwf, save_flows=False, icelltype=0, pname="npf")

        if self.save_heads:
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
        """Add a MODFLOW package to the model.

        Parameters
        ----------
        package : ModflowPackage or list of ModflowPackage
            Package instance or list of instances to add. Existing
            packages with the same name are overwritten.
        """
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

    def _remove_changing_package(self, package_name: str):
        """Remove a package from the GWF model if it exists.

        Parameters
        ----------
        package_name : str
            Name of the package to remove (e.g. ``"DIS"``).
        """
        if package_name in self._gwf.get_package_list():
            self._gwf.remove_package(package_name)

    @conditional_cachedmethod(lambda self: self._cache)
    def get_sim_index(self) -> pd.DatetimeIndex:
        """Return the simulation date-time index.

        Returns
        -------
        pd.DatetimeIndex
            Date range from ``tmin`` to ``tmax`` at model frequency.
        """
        return date_range(
            start=self.tmin,
            end=self.tmax,
            freq=self.model.settings["freq"],
        )

    def simulate(
        self, p: ArrayLike, *args: Any, tmin=None, tmax=None, **kwargs
    ) -> Series:
        """Run the MODFLOW simulation and return the head time series.

        Parameters
        ----------
        p : array-like
            Parameter values in the same order as
            ``self.parameters``.
        tmin : Timestamp, optional
            Clip the result to start from this date.
        tmax : Timestamp, optional
            Clip the result to end at this date.

        Returns
        -------
        Series
            Simulated head time series indexed by simulation dates.
        """
        s = Series(
            data=self.get_head(p=tuple(p)),
            index=self.get_sim_index(),
        )
        s.name = self.name
        if tmin is not None:
            s = s.loc[tmin:]
        if tmax is not None:
            s = s.loc[:tmax]
        return s

    @conditional_cachedmethod(lambda self: self._cache)
    def get_head(self, p: tuple) -> np.ndarray:
        """Run the MODFLOW simulation and return the head values.

        Parameters
        ----------
        p : tuple of float
            Parameter values in the same order as
            ``self.parameters``.

        Returns
        -------
        np.ndarray
            1-D array of simulated heads, one per stress period.
        """
        success, head = self._run_simulation(p)
        if success and head is not None:
            return head
        elif success and head is None:
            return self._gwf.output.head().get_ts((0, 0, 0))[:, 1]
        else:
            logger.error("ModflowError: model run failed with parameters: %s" % str(p))
            if self.raise_on_modflow_error:
                raise Exception(
                    "Modflow run failed. Check the LIST file for more information."
                )
            else:
                return np.zeros(self.nper)

    def _run_simulation(self, p: tuple) -> tuple[bool, None]:
        """Update the model and run the MODFLOW simulation.

        Parameters
        ----------
        p : tuple of float
            Parameter values in the same order as
            ``self.parameters``.

        Returns
        -------
        tuple of (bool, None)
            ``(success, None)`` where ``success`` is ``True`` if the
            simulation converged.
        """
        p_series = Series(p, index=self.parameters.index)
        logger.info(
            "run_simulation parameters: %s",
            ", ".join(f"{k}={v:.6g}" for k, v in p_series.items()),
        )
        self.update_model(p=p)
        success, _ = self._simulation.run_simulation(silent=self.silent)
        return success, None

    def update_model(self, p: ArrayLike) -> None:
        """Update the model with the given parameters.

        Parameters
        ----------
        p : array-like
            Parameter values in the same order as
            ``self.parameters``.
        """
        p_series = Series(p, index=self.parameters.index)
        for name, package in self._packages.items():
            self._remove_changing_package(package_name=name)
            pnames = package.parameter_names
            p_dict = {k.rsplit("_", 1)[-1]: v for k, v in p_series.loc[pnames].items()}
            package.update_package(modflow_gwf=self._gwf, **p_dict)
        self._gwf.name_file.write()

    def to_dict(self) -> dict:
        """Serialize the model to a dictionary (not yet implemented)."""
        raise NotImplementedError()

    def get_recharge(self, p: tuple) -> np.ndarray:
        """Compute recharge array from parameters and return it.

        Parameters
        ----------
        p : tuple of float
            Parameter values in the same order as
            ``self.parameters``.

        Returns
        -------
        np.ndarray
            1-D array of recharge values, one per stress period.
        """
        p_series = Series(p, index=self.parameters.index)
        if "RCH" in self._packages:
            rch = self._packages["RCH"]
            return rch.compute_recharge(f=p_series.loc[rch.parameter_names[0]])
        elif "UZF" in self._packages:
            uzf = self._packages["UZF"]
            pnames = uzf.parameter_names
            p_dict = {k.rsplit("_", 1)[-1]: v for k, v in p_series.loc[pnames].items()}
            uzf.update_package(
                modflow_gwf=self._gwf,
                **p_dict,
                save_flows=True,
                budget_filerecord=f"{self._gwf.name.lower()}.uzf.bud",
            )
            _ = flopy.mf6.ModflowGwfoc(
                self._gwf,
                budget_filerecord=f"{self._gwf.name}.bud",
                saverecord=[("BUDGET", "ALL")],
            )
            self._gwf.name_file.write()
            _sim = self.simulate(p)
            self._gwf.remove_package("OC")  # remove the budget output package again
            # after simulation:
            uzobj = flopy.utils.CellBudgetFile(
                f"{self.sim_ws}/{self._gwf.name}.uzf.bud", precision="double"
            )
            rch_data = uzobj.get_ts((0, 0, 0), text="GWF")  # recharge to GW from UZF
            return pd.Series(index=self.get_sim_index(), data=rch_data[:, -1])
        else:
            raise ValueError("No RCH or UZF package found. Cannot compute recharge.")


class ModflowModelApi(ModflowModel):
    """MODFLOW 6 API-based stress model for pastas.

    Extends :class:`ModflowModel` to drive the simulation via the
    MODFLOW shared-library API (XMI interface) rather than subprocess
    execution. Parameters are updated in-memory each calibration step,
    avoiding file I/O overhead.

    Parameters
    ----------
    model : pastas.Model
        The pastas model instance.
    dll : str or Path
        Path to the MODFLOW 6 shared library
        (``libmf6.so`` / ``mf6.dll``).
    sim_ws : str or Path
        Working directory for the MODFLOW simulation files.
    tmin : Timestamp, optional
        Start of the simulation period. Defaults to model ``tmin``
        minus warmup.
    tmax : Timestamp, optional
        End of the simulation period. Defaults to model ``tmax``.
    silent : bool, optional
        Suppress MODFLOW console output. Default is ``True``.
    raise_on_modflow_error : bool, optional
        Raise an exception if the MODFLOW run fails. Default is
        ``False``.
    solver_kwargs : dict, optional
        Keyword arguments passed to ``flopy.mf6.ModflowIms``.
    newtonoptions : str or None, optional
        Newton-Raphson linearization options passed to the GWF model.
    param_precision : int or None, optional
        Number of decimal places to round parameter values to before
        passing them to the MODFLOW 6 API. MODFLOW 6 writes its text
        output with ``G15.7`` format (7 significant digits); rounding
        to a comparable precision avoids spurious cache misses caused
        by optimizer steps smaller than MODFLOW's own numerical
        resolution. Set to ``None`` to disable rounding and pass
        parameters at full float64 precision. Default is ``None``.
    """

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
        newtonoptions=None,
        param_precision: int | None = None,
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
            save_heads=False,
            newtonoptions=newtonoptions,
        )
        self.dll = dll
        self.param_precision = param_precision

    def simulate(
        self, p: ArrayLike, *args: Any, tmin=None, tmax=None, **kwargs
    ) -> Series:
        """Round parameters then delegate to the parent simulate.

        If :attr:`param_precision` is set, ``p`` is rounded to that
        many decimal places before being used as the cache key and
        before being written to MODFLOW memory. This prevents
        optimizer micro-steps (smaller than MODFLOW's own ``G15.7``
        output resolution) from generating spurious cache misses and
        unnecessary MODFLOW runs.

        Parameters
        ----------
        p : array-like
            Parameter values in the same order as
            ``self.parameters``.
        tmin : Timestamp, optional
            Clip the result to start from this date.
        tmax : Timestamp, optional
            Clip the result to end at this date.

        Returns
        -------
        Series
            Simulated head time series indexed by simulation dates.
        """
        if self.param_precision is not None:
            p = np.round(p, self.param_precision)
        return super().simulate(p, *args, tmin=tmin, tmax=tmax, **kwargs)

    def set_init_parameters(self) -> None:
        """Set the initial parameters back to their default values."""
        super().set_init_parameters()
        # build the modflow model with initial parameters,
        # this is fine since all relevant parameters will
        # be modified in the API loop
        self.initialize_model(p=self.parameters["initial"].values)

    def initialize_model(self, p: ArrayLike) -> None:
        """Write all packages using the given parameters.

        Parameters
        ----------
        p : array-like
            Parameter values in the same order as
            ``self.parameters``.
        """
        p_series = Series(p, index=self.parameters.index)
        for name, package in self._packages.items():
            pnames = package.parameter_names
            p_dict = {k.rsplit("_", 1)[-1]: v for k, v in p_series.loc[pnames].items()}
            package.update_package(modflow_gwf=self._gwf, **p_dict)
        # write nam file after initialization
        self._gwf.name_file.write()

    def _dispatch_to_packages(self, method_name: str, mf6, p) -> None:
        """Call a named method on every package that implements it.

        Builds the per-package parameter tuple from *p* and dispatches
        to ``pkg.<method_name>(mf6, p_tuple)`` for each package that
        has the method. Used internally by
        ``update_static_parameters`` and
        ``update_period_parameters`` to avoid duplicated iteration
        logic.

        Parameters
        ----------
        method_name : str
            Name of the method to call on each package.
        mf6 : modflowapi.ModflowApi
            Active MODFLOW 6 API instance.
        p : array-like
            Parameter values in the same order as
            ``self.parameters``.
        """
        p_series = Series(p, index=self.parameters.index)
        for pkg in self._packages.values():
            method = getattr(pkg, method_name, None)
            if method is not None:
                p_tuple = tuple(p_series.loc[pkg.parameter_names].tolist())
                method(mf6, p_tuple)

    def update_static_parameters(self, mf6, p: np.ndarray) -> None:
        """Update static (non-period) parameters in all packages.

        Should be called after ``mf6.initialize()``.

        Parameters
        ----------
        mf6 : modflowapi.ModflowApi
            Active MODFLOW 6 API instance.
        p : np.ndarray
            Parameter values in the same order as
            ``self.parameters``.
        """
        self._dispatch_to_packages("update_parameters", mf6, p)

    def update_period_parameters(self, mf6, p: np.ndarray) -> None:
        """Write API-calibrated period-data values once at kper=0.

        The input file contains only one PERIOD block for each affected
        package (GHB, UZF). MODFLOW re-reads that block at kper=0 via
        ``*_rp()``; for kper>0 it short-circuits because there is no
        new PERIOD block, leaving the in-memory arrays untouched. A
        single pointer write at kper=0 therefore persists for the
        entire simulation. Packages implement this via
        ``set_value_ptr`` (no address-string lookup).

        Parameters
        ----------
        mf6 : modflowapi.ModflowApi
            Active MODFLOW 6 API instance.
        p : np.ndarray
            Parameter values in the same order as
            ``self.parameters``.
        """
        self._dispatch_to_packages("update_period_parameters", mf6, p)

    def update_package_timeseries(self, p: tuple) -> None:
        """Dynamically find and update time series files for any package.

        Parameters
        ----------
        p : tuple of float
            Parameter values in the same order as
            ``self.parameters``.
        """
        p_series = Series(p, index=self.parameters.index)

        for name, package in self._packages.items():
            # Check if the package defines a time series writer
            if hasattr(package, "update_ts"):
                # Get the parameter names belonging to this specific package
                pnames = package.parameter_names

                # Extract values into a dictionary (e.g., {'f': 1.25})
                p_dict = {
                    k.rsplit("_", 1)[-1]: v for k, v in p_series.loc[pnames].items()
                }

                # Filter the dictionary to ONLY include arguments the package's
                # update_ts expects.
                sig = signature(package.update_ts)
                valid_kwargs = {k: v for k, v in p_dict.items() if k in sig.parameters}

                # Write the time series!
                package.write_ts(self._gwf, **valid_kwargs)

    def log_parameters(self, mf6, kper, level=logging.DEBUG):
        """Forward log_parameters to each package that implements it.

        Parameters
        ----------
        mf6 : modflowapi.ModflowApi
            Active MODFLOW 6 API instance.
        kper : int
            Current stress-period index passed to each package.
        level : int, optional
            Python logging level. Default is ``logging.DEBUG``.
        """
        for _, package in self._packages.items():
            if hasattr(package, "log_parameters"):
                package.log_parameters(mf6, kper, level=level)

    def _run_simulation(self, p: tuple) -> bool:
        """Run the MODFLOW simulation via the shared-library API.

        Parameters
        ----------
        p : tuple of float
            Parameter values in the same order as
            ``self.parameters``.

        Returns
        -------
        tuple of (bool, np.ndarray)
            ``(success, head)`` where ``success`` is ``True`` if all
            time steps converged and ``head`` is a 1-D array of
            simulated heads per stress period.
        """
        p_series = Series(p, index=self.parameters.index)
        logger.info(
            "run_simulation parameters: %s",
            ", ".join(f"{k}={v:.6g}" for k, v in p_series.items()),
        )

        # check if time series have to be updated
        self.update_package_timeseries(p)

        # start the API fun
        mf6 = modflowapi.ModflowApi(self.dll, working_directory=self.sim_ws)
        mf6.initialize()

        # Update static package arrays after initialize().
        self.update_static_parameters(mf6=mf6, p=p)

        success = False
        all_timesteps_converged = True

        # time loop
        current_time = mf6.get_current_time()
        end_time = mf6.get_end_time()

        # maximum outer iterations
        max_iter = mf6.get_value(mf6.get_var_address("MXITER", "SLN_1"))

        # Outside the loop (after initialization)
        head = np.zeros(self.nper, dtype=float)
        head_ptr = mf6.get_value_ptr(f"{self.name.upper()}/X")

        # model time loop
        kper = 0
        while current_time < end_time:
            dt = mf6.get_time_step()

            # prepare for non-linear iterations
            mf6.prepare_time_step(dt)

            # In first period update period parameters. These will be applied
            # to subsequent periods.
            if kper == 0:
                self.update_period_parameters(mf6=mf6, p=p)

            # # Keep period recharge in sync with the expected time-series value.
            # if rch_ptr is not None and rch_values is not None and kper < len(rch_values):
            #     rch_ptr[:] = rch_values[kper]

            mf6.prepare_solve()

            # Log at INFO for the first period so initial API values are
            # always visible; subsequent periods are logged at DEBUG.
            log_level = logging.INFO if kper == 0 else logging.DEBUG
            self.log_parameters(mf6, kper, level=log_level)

            # convergence loop
            kiter = 0
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
                all_timesteps_converged = False
                break

            head[kper] = head_ptr.item()

            # increment counter
            kper += 1

        # cleanup
        try:
            mf6.finalize()
            success = all_timesteps_converged
        except Exception:
            logger.warning("Exception occurred.", exc_info=True)
            return success, head
        return success, head


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
