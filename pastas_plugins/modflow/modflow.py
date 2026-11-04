# %%
import logging
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import flopy
import numpy as np
from pandas import DataFrame, Series

logger = logging.getLogger(__name__)


class ModflowApi:
    """Manage MODFLOW 6 API parameters for a single package.

    Acts as a dictionary mapping parameter names to full MODFLOW address
    strings. Provides both a slow path (``set_value``) and a fast cached
    pointer path (``set_value_ptr``) for writing values into MODFLOW memory.

    Parameters
    ----------
    pkg : str
        MODFLOW 6 package name, e.g. ``"STO"`` or ``"DIS"``.

    Notes
    -----
    Call ``set_model_name`` before using ``__getitem__``, ``set_value``,
    or ``get_value``.

    Examples
    --------
    .. code-block:: python

        api = ModflowApi(pkg="STO")
        api.set_model_name("my_model")
        api.add_parameters("SS", "SY")
        api.set_value(mf6, "SS", np.array([1e-3]))
    """

    def __init__(self, pkg: str):
        self.pkg = pkg
        self.model_name = None
        self.parameters = {}
        self._ptrs: dict[str, np.ndarray] = {}

    def __repr__(self):
        """Return string representation with package name and parameters."""
        s = f"ModflowApi(pkg_name={self.pkg}, model_name={self.model_name})"
        for pname in self.parameters:
            s += f"\n  - {pname}: {self[pname]}"
        return s

    def __getitem__(self, parameter_name: str):
        """Return the full MODFLOW address string for *parameter_name*.

        Parameters
        ----------
        parameter_name : str
            Name of a previously registered parameter.

        Returns
        -------
        str
            Address string, e.g. ``"MY_MODEL/STO/SS"``.
        """
        return self.parameters[parameter_name].format(
            model_name=self.model_name, pkg_name=self.pkg
        )

    def set_model_name(self, model_name: str) -> None:
        """Set the GWF model name used when building address strings.

        Parameters
        ----------
        model_name : str
            Name of the GWF model. Stored as upper-case.
        """
        self.model_name = model_name.upper()

    def add_parameters(self, *args) -> None:
        """Register one or more parameter names for this package.

        Parameters
        ----------
        *args : str
            Parameter names to register (must be strings).
        """
        for p in args:
            assert isinstance(p, str), "Parameter names must be strings."
            self.parameters[p] = "{model_name}/{pkg_name}/" + f"{p.upper()}"

    def set_value(self, mf6, parameter_name: str, value: np.ndarray) -> None:
        """Write *value* to MODFLOW memory via address-string lookup.

        Parameters
        ----------
        mf6 : modflowapi.ModflowApi
            Active MODFLOW 6 API instance.
        parameter_name : str
            Name of a registered parameter.
        value : np.ndarray
            Array to write into MODFLOW memory.
        """
        mf6.set_value(self[parameter_name], value)

    def get_value(self, mf6, parameter_name: str) -> np.ndarray:
        """Read the current value of a MODFLOW parameter.

        Parameters
        ----------
        mf6 : modflowapi.ModflowApi
            Active MODFLOW 6 API instance.
        parameter_name : str
            Name of a registered parameter.

        Returns
        -------
        np.ndarray
            Copy of the current MODFLOW memory array.
        """
        return mf6.get_value(self[parameter_name])

    def get_ptr(self, mf6, parameter_name: str) -> np.ndarray:
        """Return a shared-memory pointer to a MODFLOW variable.

        Parameters
        ----------
        mf6 : modflowapi.ModflowApi
            Active MODFLOW 6 API instance.
        parameter_name : str
            Name of a registered parameter.

        Returns
        -------
        np.ndarray
            Array sharing memory with the MODFLOW internal variable.
        """
        return mf6.get_value_ptr(self[parameter_name])

    def init_ptrs(self, mf6) -> None:
        """Resolve and cache shared-memory pointers for all registered parameters.

        Call at the end of ``update_parameters`` after all ``set_value`` writes.
        Subsequent ``set_value_ptr`` calls write directly into MODFLOW memory
        without any address-string lookup.

        Parameters
        ----------
        mf6 : modflowapi.ModflowApi
            Active MODFLOW 6 API instance.
        """
        self._ptrs = {p: mf6.get_value_ptr(self[p]) for p in self.parameters}

    def set_value_ptr(self, parameter_name: str, value) -> None:
        """Write *value* via cached pointer (fast path).

        Parameters
        ----------
        parameter_name : str
            Name of a registered parameter.
        value : array-like
            Value to write in-place into the cached MODFLOW memory array.

        Notes
        -----
        Requires ``init_ptrs`` to have been called first.
        """
        self._ptrs[parameter_name][:] = value


# %%


@runtime_checkable
class ModflowPackage(Protocol):
    """Define the interface that every MODFLOW package wrapper must implement."""

    _name: str
    api: ModflowApi

    @property
    def parameter_names(self) -> list: ...

    def get_init_parameters(self, name: str) -> DataFrame: ...

    def update_package(
        self, modflow_gwf: flopy.mf6.ModflowGwf, **kwargs: Any
    ) -> None: ...

    def update_parameters(self, mf6, params: tuple, **kwargs: Any) -> None: ...

    def stress(self) -> dict[str, Series] | None: ...


class ModflowPkgLoggerMixin:
    def log_parameters(self, mf6, kper, level=logging.DEBUG):
        """Log current in-memory pkg API parameter values.

        Parameters
        ----------
        mf6 : modflowapi.ModflowApi
            Active MODFLOW 6 API instance.
        kper : int
            Current stress-period index, shown in the log message.
        level : int, optional
            Python logging level. Default is ``logging.DEBUG``.
        """
        if not logger.isEnabledFor(level):
            return
        pdict = {
            ipar: self.api.get_value(mf6, ipar) for ipar in self.api.parameters.keys()
        }
        msg = f"kper={kper}, {self._name}: " + ", ".join(
            f"{k}={v}" for k, v in pdict.items()
        )
        logger.log(level, msg)


class ModflowDis(ModflowPkgLoggerMixin):
    """DIS package wrapper for single-layer, single-cell groundwater models.

    Geometry is defined from calibrated water level ``d`` (TOP = d + H,
    BOT = d - 100).
    """

    def __init__(self):
        self._name = "DIS"
        self.api = ModflowApi(self._name)

    @property
    def parameter_names(self):
        """Return calibration parameter names for this package.

        Returns
        -------
        list of str
            ``["constant_d", "DIS_H"]`` — water level offset and aquifer
            thickness.
        """
        return ["constant_d", self._name + "_H"]

    def get_init_parameters(self, name: str) -> DataFrame:
        """Return initial parameter bounds and metadata for this package.

        Parameters
        ----------
        name : str
            Stress-model instance name stored in the ``name`` column.

        Returns
        -------
        DataFrame
            Rows indexed by parameter name with columns ``initial``,
            ``pmin``, ``pmax``, ``vary``, ``name``, ``package``, ``dist``.
        """
        parameters = DataFrame(
            {
                "initial": [0.0, 0.0],
                "pmin": [np.nan, 0.0],
                "pmax": [np.nan, 10.0],
                "vary": [True, False],
                "name": [name, name],
                "package": [self._name, self._name],
                "dist": ["uniform", "uniform"],
            },
            index=self.parameter_names,
        )
        return parameters

    def update_package(
        self, modflow_gwf: flopy.mf6.ModflowGwf, d: float, H: float = 0.0
    ) -> None:
        """Write the DIS package with geometry derived from ``d`` and ``H``.

        Parameters
        ----------
        modflow_gwf : flopy.mf6.ModflowGwf
            Active flopy GWF model object.
        d : float
            Water level offset; TOP = d + H, BOT = d - 100.
        H : float, optional
            Aquifer thickness [L]. Default is ``1.0``.
        """
        botm = d - 100.0
        top = d + H
        dis = flopy.mf6.ModflowGwfdis(
            modflow_gwf,
            length_units="METERS",
            nlay=1,
            nrow=1,
            ncol=1,
            delr=1,
            delc=1,
            top=top,
            botm=botm,
            # idomain=1,
            pname=self._name,
        )
        dis.write()
        # API stuff
        self.api.set_model_name(modflow_gwf.name)
        self.api.add_parameters("TOP", "BOT")

    def update_parameters(self, mf6, params: tuple) -> None:
        """Write calibrated TOP and BOT to MODFLOW via cached pointers.

        Parameters
        ----------
        mf6 : modflowapi.ModflowApi
            Active MODFLOW 6 API instance.
        params : tuple of float
            ``(d, H)`` — water level offset and aquifer thickness.
        """
        d, H = params
        self.api.init_ptrs(mf6)
        self.api.set_value_ptr("TOP", np.array([d + H]))
        self.api.set_value_ptr("BOT", np.array([d - 100.0]))

    def stress(self) -> None:
        """Return None; this package has no stress time series.

        Returns
        -------
        None
        """
        return None


class ModflowIc(ModflowPkgLoggerMixin):
    """IC package wrapper. Sets STRT and working head X to calibrated ``d``."""

    def __init__(self):
        self._name = "IC"
        self.api = ModflowApi(self._name)

    @property
    def parameter_names(self):
        """Return calibration parameter names for this package.

        Returns
        -------
        list of str
            ``["constant_d"]`` — initial head / water level offset.
        """
        return ["constant_d"]

    def get_init_parameters(self, name: str) -> DataFrame:
        """Return initial parameter bounds and metadata for this package.

        Parameters
        ----------
        name : str
            Stress-model instance name stored in the ``name`` column.

        Returns
        -------
        DataFrame
            Rows indexed by parameter name with columns ``initial``,
            ``pmin``, ``pmax``, ``vary``, ``name``, ``package``, ``dist``.
        """
        parameters = DataFrame(
            {
                "initial": [0.0],
                "pmin": [np.nan],
                "pmax": [np.nan],
                "vary": [True],
                "name": [name],
                "package": [self._name],
                "dist": ["uniform"],
            },
            index=self.parameter_names,
        )
        return parameters

    def update_package(self, modflow_gwf: flopy.mf6.ModflowGwf, d: float) -> None:
        """Write the IC package with starting head equal to ``d``.

        Parameters
        ----------
        modflow_gwf : flopy.mf6.ModflowGwf
            Active flopy GWF model object.
        d : float
            Starting head [L].
        """
        ic = flopy.mf6.ModflowGwfic(modflow_gwf, strt=d, pname=self._name)
        ic.write()
        # API stuff
        self.api.set_model_name(modflow_gwf.name)
        self.api.add_parameters("STRT")

    def update_parameters(self, mf6, params) -> None:
        """Write calibrated STRT and reset working head array X.

        Parameters
        ----------
        mf6 : modflowapi.ModflowApi
            Active MODFLOW 6 API instance.
        params : tuple of float
            ``(d,)`` — water level offset.
        """
        (d,) = params
        self.api.init_ptrs(mf6)
        self.api.set_value_ptr("STRT", np.array([d]))
        # SPECIAL: IC/STRT is the file-read initial condition. mf6.initialize()
        # copies STRT → X before this method runs, so setting STRT alone does
        # NOT reset the working head array X. X is a model-level variable
        # (not package-scoped) and cannot be registered in the package API;
        # it must be reset directly via a raw get_value_ptr call.
        x_ptr = mf6.get_value_ptr(f"{self.api.model_name}/X")
        x_ptr[:] = d

    def stress(self) -> None:
        """Return None; this package has no stress time series.

        Returns
        -------
        None
        """
        return None


class ModflowStoConfined(ModflowPkgLoggerMixin):
    """STO package wrapper.

    Calibrates specific storage, normalised by aquifer thickness.
    """

    def __init__(self):
        self._name = "STO"
        self.api = ModflowApi(self._name)

    @property
    def parameter_names(self):
        """Return calibration parameter names for this package.

        Returns
        -------
        list of str
            ``["STO_S"]`` — specific storage.
        """
        return [self._name + "_S"]

    def get_init_parameters(self, name: str) -> DataFrame:
        """Return initial parameter bounds and metadata for this package.

        Parameters
        ----------
        name : str
            Stress-model instance name stored in the ``name`` column.

        Returns
        -------
        DataFrame
            Rows indexed by parameter name with columns ``initial``,
            ``pmin``, ``pmax``, ``vary``, ``name``, ``package``, ``dist``.
        """
        parameters = DataFrame(
            {
                "initial": [0.1],
                "pmin": [0.001],
                "pmax": [0.5],
                "vary": [True],
                "name": [name],
                "package": [self._name],
                "dist": ["uniform"],
            },
            index=self.parameter_names,
        )
        return parameters

    def update_package(self, modflow_gwf: flopy.mf6.ModflowGwf, S: float) -> None:
        """Write the STO package with specific storage normalised by aquifer thickness.

        Parameters
        ----------
        modflow_gwf : flopy.mf6.ModflowGwf
            Active flopy GWF model object.
        S : float
            Specific storage value [1/L] (normalised internally by aquifer thickness).
        """
        haq = modflow_gwf.dis.top.array[0, 0] - modflow_gwf.dis.botm.array[0, 0, 0]
        sto = flopy.mf6.ModflowGwfsto(
            modflow_gwf,
            save_flows=False,
            # iconvert=1,
            ss=S / haq,
            # sy=S,
            transient=True,
            # ss_confined_only=True,
            pname=self._name,
        )
        sto.write()
        # API stuff
        self.api.set_model_name(modflow_gwf.name)
        self.api.add_parameters("SS")

    def update_parameters(self, mf6, params) -> None:
        """Write calibrated SS normalised by aquifer thickness.

        Parameters
        ----------
        mf6 : modflowapi.ModflowApi
            Active MODFLOW 6 API instance.
        params : tuple of float
            ``(ss,)`` — specific storage value.
        """
        (ss,) = params
        # CROSS-PACKAGE READ: SS is normalized by aquifer thickness from DIS.
        # ORDERING DEPENDENCY: requires ModflowDis.update_parameters() to have
        # run first so that DIS/TOP and DIS/BOT already hold calibrated values.
        haq = mf6.get_value(f"{self.api.model_name}/DIS/TOP") - mf6.get_value(
            f"{self.api.model_name}/DIS/BOT"
        )
        self.api.init_ptrs(mf6)
        self.api.set_value_ptr("SS", ss / haq)

    def stress(self) -> None:
        """Return None; this package has no stress time series.

        Returns
        -------
        None
        """
        return None


class ModflowStoPhreatic(ModflowStoConfined):
    def __init__(self):
        self._name = "STO"
        self.api = ModflowApi(self._name)

    @property
    def parameter_names(self):
        """Return calibration parameter names for this package.

        Returns
        -------
        list of str
            ``["STO_Ss","STO_Sy"]`` — specific storage.
        """
        return [self._name + "_Ss", self._name + "_Sy"]

    def get_init_parameters(self, name: str) -> DataFrame:
        """Return initial parameter bounds and metadata for this package.

        Parameters
        ----------
        name : str
            Stress-model instance name stored in the ``name`` column.

        Returns
        -------
        DataFrame
            Rows indexed by parameter name with columns ``initial``,
            ``pmin``, ``pmax``, ``vary``, ``name``, ``package``, ``dist``.
        """
        parameters = DataFrame(
            {
                "initial": [1e-2, 0.2],
                "pmin": [1e-5, 1e-2],
                "pmax": [0.5, 0.5],
                "vary": [True, True],
                "name": [name, name],
                "package": [self._name, self._name],
                "dist": ["uniform", "uniform"],
            },
            index=self.parameter_names,
        )
        return parameters

    def update_package(
        self, modflow_gwf: flopy.mf6.ModflowGwf, Ss: float, Sy: float
    ) -> None:
        """Write the STO package with specific yield and storage normalised by aquifer thickness.

        Parameters
        ----------
        modflow_gwf : flopy.mf6.ModflowGwf
            Active flopy GWF model object.
        Ss : float
            Specific storage value [1/L] (normalised internally by aquifer thickness).
        Sy : float
            Specific yield.
        """
        haq = modflow_gwf.dis.top.array[0, 0] - modflow_gwf.dis.botm.array[0, 0, 0]
        sto = flopy.mf6.ModflowGwfsto(
            modflow_gwf,
            save_flows=False,
            iconvert=1,
            ss=Ss / haq,
            sy=Sy,
            transient=True,
            ss_confined_only=True,
            pname=self._name,
        )
        sto.write()
        # API stuff
        self.api.set_model_name(modflow_gwf.name)
        self.api.add_parameters("SS", "SY")

    def update_parameters(self, mf6, params) -> None:
        """Write calibrated SS normalised by aquifer thickness.

        Parameters
        ----------
        mf6 : modflowapi.ModflowApi
            Active MODFLOW 6 API instance.
        params : tuple of float
            ``(ss,)`` — specific storage value.
        """
        (ss, sy) = params
        # CROSS-PACKAGE READ: SS is normalized by aquifer thickness from DIS.
        # ORDERING DEPENDENCY: requires ModflowDis.update_parameters() to have
        # run first so that DIS/TOP and DIS/BOT already hold calibrated values.
        haq = mf6.get_value(f"{self.api.model_name}/DIS/TOP") - mf6.get_value(
            f"{self.api.model_name}/DIS/BOT"
        )
        self.api.init_ptrs(mf6)
        self.api.set_value_ptr("SS", ss / haq)
        self.api.set_value_ptr("SY", sy)


class ModflowGhb(ModflowPkgLoggerMixin):
    """GHB package wrapper. Calibrates boundary head and conductance."""

    def __init__(self):
        self._name = "GHB"
        self.api = ModflowApi(self._name)

    @property
    def parameter_names(self):
        """Return calibration parameter names for this package.

        Returns
        -------
        list of str
            ``["constant_d", "GHB_C"]`` — boundary head and conductance.
        """
        return ["constant_d", self._name + "_C"]

    def get_init_parameters(self, name: str) -> DataFrame:
        """Return initial parameter bounds and metadata for this package.

        Parameters
        ----------
        name : str
            Stress-model instance name stored in the ``name`` column.

        Returns
        -------
        DataFrame
            Rows indexed by parameter name with columns ``initial``,
            ``pmin``, ``pmax``, ``vary``, ``name``, ``package``, ``dist``.
        """
        parameters = DataFrame(
            {
                "initial": [0.0, 1e-3],
                "pmin": [np.nan, 1e-5],
                "pmax": [np.nan, 1e-1],
                "vary": [True, True],
                "name": [name, name],
                "package": [self._name, self._name],
                "dist": ["uniform", "uniform"],
            },
            index=self.parameter_names,
        )
        return parameters

    def update_package(
        self, modflow_gwf: flopy.mf6.ModflowGwf, d: float, C: float
    ) -> None:
        """Write GHB package with boundary head and conductance.

        Parameters
        ----------
        modflow_gwf : flopy.mf6.ModflowGwf
            Active flopy GWF model object.
        d : float
            Boundary head.
        C : float
            Conductance [L²/T].
        """
        ghb = flopy.mf6.ModflowGwfghb(
            modflow_gwf,
            maxbound=1,
            stress_period_data={0: [[(0, 0, 0), d, C]]},
            pname=self._name,
        )
        ghb.write()
        # API stuff
        self.api.set_model_name(modflow_gwf.name)
        self.api.add_parameters("BHEAD", "COND")

    def update_parameters(self, mf6, params) -> None:
        """Write calibrated BHEAD and COND and resolve pointer cache.

        Parameters
        ----------
        mf6 : modflowapi.ModflowApi
            Active MODFLOW 6 API instance.
        params : tuple of float
            ``(d, C)`` — boundary head and conductance.
        """
        d, C = params
        self.api.init_ptrs(mf6)
        self.api.set_value_ptr("BHEAD", np.array([d]))
        self.api.set_value_ptr("COND", np.array([C]))

    def update_period_parameters(self, mf6, params) -> None:
        """Override BHEAD and COND once at kper=0 via cached pointers.

        The input file defines a single PERIOD block. In-memory arrays are
        not re-read after kper=0, so values written here persist for the
        whole simulation.

        Parameters
        ----------
        mf6 : modflowapi.ModflowApi
            Active MODFLOW 6 API instance.
        params : tuple of float
            ``(d, C)`` — boundary head and conductance.
        """
        d, C = params
        self.api.set_value_ptr("BHEAD", np.array([d]))
        self.api.set_value_ptr("COND", np.array([C]))

    def stress(self) -> None:
        """Return None; this package has no stress time series.

        Returns
        -------
        None
        """
        return None


class ModflowRch(ModflowPkgLoggerMixin):
    """RCH package wrapper. Recharge = ``prec + f * evap`` via a time-series file."""

    def __init__(
        self,
        prec: Series,
        evap: Series,
    ):
        """Initialize with precipitation and evapotranspiration time series.

        Parameters
        ----------
        prec : Series
            Precipitation time series [L/T].
        evap : Series
            Evapotranspiration time series [L/T].
        """
        self._name = "RCH"
        self.prec = prec
        self.evap = evap
        self.recharge = None  # is recomputed

        self.api = ModflowApi(self._name)

    @property
    def parameter_names(self):
        """Return calibration parameter names for this package.

        Returns
        -------
        list of str
            ``["RCH_f"]`` — evaporation scaling factor.
        """
        return [self._name + "_f"]

    def get_init_parameters(self, name: str) -> DataFrame:
        """Return initial parameter bounds and metadata for this package.

        Parameters
        ----------
        name : str
            Stress-model instance name stored in the ``name`` column.

        Returns
        -------
        DataFrame
            Rows indexed by parameter name with columns ``initial``,
            ``pmin``, ``pmax``, ``vary``, ``name``, ``package``, ``dist``.
        """
        parameters = DataFrame(
            {
                "initial": [-1.0],
                "pmin": [-2.0],
                "pmax": [0.0],
                "vary": [True],
                "name": [name],
                "package": [self._name],
                "dist": ["uniform"],
            },
            index=self.parameter_names,
        )
        return parameters

    def compute_recharge(self, f: float) -> None:
        """Compute recharge as ``prec + f * evap`` and store in-place.

        The result is stored as a numpy array in ``self.recharge``.

        Parameters
        ----------
        f : float
            Evaporation scaling factor.
        """
        self.recharge = (self.prec + f * self.evap).to_numpy()

    def update_package(self, modflow_gwf: flopy.mf6.ModflowGwf, f: float) -> None:
        """Write RCH package and time-series file.

        Parameters
        ----------
        modflow_gwf : flopy.mf6.ModflowGwf
            Active flopy GWF model object.
        f : float
            Evaporation scaling factor used to compute recharge.
        """
        rch = flopy.mf6.ModflowGwfrch(
            modflow_gwf,
            maxbound=1,
            stress_period_data={0: [[(0, 0, 0), "recharge"]]},
            timeseries={
                "filename": f"{modflow_gwf.name.lower()}.rch_ts",
                "time_series_namerecord": ["recharge"],
            },
            pname=self._name,
        )
        rch.write()
        # write time series file
        self.write_ts(modflow_gwf, f)

        # API stuff
        self.api.set_model_name(modflow_gwf.name)
        self.api.add_parameters("RECHARGE")

    def write_ts(self, modflow_gwf: flopy.mf6.ModflowGwf, f: float) -> None:
        """Write the RCH time-series file for a given evaporation scaling factor.

        Parameters
        ----------
        modflow_gwf : flopy.mf6.ModflowGwf
            Active flopy GWF model object.
        f : float
            Evaporation scaling factor.
        """
        self.compute_recharge(f)

        # Prepare arrays
        time_array = np.arange(modflow_gwf.nper + 1, dtype=float)
        recharge_array = np.append(self.recharge, 0.0)
        filepath = Path(modflow_gwf.model_ws) / f"{modflow_gwf.name.lower()}.rch_ts"

        # Bypass FloPy and write directly
        fast_write_ts(
            filepath=filepath,
            names=["recharge"],
            methods=["stepwise"],
            time_array=time_array,
            data_arrays=[recharge_array],
        )

    def update_ts(self, modflow_gwf: flopy.mf6.ModflowGwf, f: float):
        """Re-write the RCH time-series file for a new calibration step.

        Parameters
        ----------
        modflow_gwf : flopy.mf6.ModflowGwf
            Active flopy GWF model object.
        f : float
            Evaporation scaling factor.
        """
        self.write_ts(modflow_gwf, f)

    def stress(self) -> dict[str, Series]:
        """Return precipitation and evaporation input series.

        Returns
        -------
        dict of str to Series
            Keys ``"prec"`` and ``"evap"``.
        """
        return {"prec": self.prec, "evap": self.evap}


class ModflowUzf(ModflowPkgLoggerMixin):
    """UZF package wrapper with a companion DRN for surface-water routing.

    Calibrates saturated hydraulic conductivity, water-content parameters,
    and ET extinction parameters. Automatically creates a DRN package to
    replace the deprecated ``simulate_gwseep`` option.
    """

    def __init__(
        self,
        prec: Series,
        evap: Series,
        simulate_et: bool = True,
        gwet_linear_or_square: None | Literal["linear", "square"] = "linear",
        ntrailwaves: int = 7,
        nwavesets: int = 40,
    ):
        """Initialize with precipitation/ET series and UZF solver settings.

        Parameters
        ----------
        prec : Series
            Precipitation time series used as infiltration rate [L/T].
        evap : Series
            Potential evapotranspiration time series [L/T].
        simulate_et : bool, optional
            Simulate ET in the unsaturated zone. Default is ``True``.
        gwet_linear_or_square : {"linear", "square"} or None, optional
            Groundwater ET formulation. Default is ``"linear"``.
        ntrailwaves : int, optional
            Number of trailing waves for the kinematic-wave solver.
        nwavesets : int, optional
            Number of wave sets for the kinematic-wave solver.
        """
        self._name = "UZF"
        self.api = ModflowApi(self._name)
        # api_drn manages the companion DRN package that UZF auto-creates for
        # surface-water routing (simulate_gwseep replacement). Its ELEV must
        # track DIS/TOP in API runs and is set via this separate API instance.
        self.api_drn = ModflowApi("DRN")
        self.prec = prec
        self.evap = evap
        self.simulate_et = simulate_et
        self.gwet_linear_or_square = gwet_linear_or_square
        self.ntrailwaves = ntrailwaves
        self.nwavesets = nwavesets

        # set some default parameters
        self._nlay = 1  # only one uzf cell / layer
        self._surfdep = 1e-5  # surface depression depth
        self._landflag = np.zeros(self._nlay, dtype=int)
        self._landflag[0] = 1

    @property
    def parameter_names(self):
        """Return calibration parameter names for this package.

        Returns
        -------
        list of str
            ``["DIS_H", "UZF_vks", "UZF_thtr", "UZF_thts",``
            ``"UZF_thextfrac", "UZF_eps", "UZF_extdpfrac"]``.
        """
        return [
            "DIS_H",  # needs to be shared with DIS pkg
            self._name + "_vks",
            self._name + "_thtr",
            self._name + "_thts",
            self._name + "_thextfrac",
            self._name + "_eps",
            self._name + "_extdpfrac",
        ]

    def get_init_parameters(self, name: str) -> DataFrame:
        """Return initial parameter bounds and metadata for this package.

        Parameters
        ----------
        name : str
            Stress-model instance name stored in the ``name`` column.

        Returns
        -------
        DataFrame
            Rows indexed by parameter name with columns ``initial``,
            ``pmin``, ``pmax``, ``vary``, ``name``, ``package``, ``dist``.
        """
        parameters = DataFrame(
            {
                "initial": [1.0, 1.0, 0.1, 0.3, 0.1, 5.0, 0.5],
                "pmin": [0.01, 0.0, 0.0, 0.2, 0.0, 3.5, 0.0],
                "pmax": [10.0, 10.0, 0.2, 0.4, 1.0, 10.0, 1.0],
                "vary": [True] * 7,
                "name": [name] * 7,
                "package": [f"DIS|{self._name}"] + [self._name] * 6,
                "dist": ["uniform"] * 7,
            },
            index=self.parameter_names,
        )
        return parameters

    def update_package(
        self,
        modflow_gwf: flopy.mf6.ModflowGwf,
        H: float,
        vks: float,
        thtr: float,
        thts: float,
        eps: float,
        thextfrac: float,
        extdpfrac: float,
        save_flows: bool = False,
        budget_filerecord: str | None = None,
        **kwargs,
    ) -> None:
        """Write UZF package, time-series file, and companion DRN.

        Parameters
        ----------
        modflow_gwf : flopy.mf6.ModflowGwf
            Active flopy GWF model object.
        H : float
            Aquifer thickness [L].
        vks : float
            Saturated vertical hydraulic conductivity [L/T].
        thtr : float
            Residual water content [-].
        thts : float
            Saturated water content [-].
        eps : float
            Brooks-Corey epsilon exponent [-].
        thextfrac : float
            ET extinction water content as fraction of available pore space.
        extdpfrac : float
            ET extinction depth as fraction of ``H`` [L/L].
        """
        extdp = extdpfrac * H
        thext = thtr + (thts - thtr) * thextfrac

        thti = (thts + thtr) / 2  # initial water content
        # Evapotranspiration in the unsaturated zone will be simulated as a
        # function of the specified potential evapotranspiration rate while
        # the water content (THETA) is greater than the ET extinction water
        # content (EXTWC).
        unsat_etwc = True
        # only if unsat_etae is True

        # Evapotranspiration in the unsaturated zone will be simulated
        # simulated using a capillary pressure based formulation. Capillary
        # pressure is calculated using the Brooks-Corey retention function.
        unsat_etae = False
        ha = 0.0  # air entry potential (head)
        hroot = 0.0  # the root potential (head)
        rootact = 0.0  # root length per soil volume [L^-2]

        uzf_pkdat = [
            [
                n,  # iuzno
                (0, 0, 0),  # gwf_cellid
                self._landflag[n],  # landflag
                n + 1 if (n + 1) != self._nlay else -1,  # ivertcon
                self._surfdep,  # surface depression depth
                vks,  # vertical saturated hydraulic conductivity
                thtr,  # residual water content
                thts,  # saturated water content
                thti,  # initial water content
                eps,  # brooks-corey epsilon exponent
                f"CELLID_UZF_{n:03d}",  # boundname
            ]
            for n in range(self._nlay)
        ]

        perioddata = {
            0: [
                [n, "finf", "pet", extdp, thext, ha, hroot, rootact]
                for n in range(self._nlay)
            ]
        }

        uzf = flopy.mf6.ModflowGwfuzf(
            modflow_gwf,
            print_input=False,
            print_flows=False,
            save_flows=save_flows,
            boundnames=True,
            # If this option is selected, evapotranspiration will be simulated
            # in the unsaturated zone but not in the saturated zone.
            simulate_et=True,
            # If this option is selected, evapotranspiration will be simulated
            # in both the unsaturated and saturated zones. The groundwater
            # evapotranspiration will be simulated using the original ET
            # formulation of MODFLOW-2005.
            linear_gwet=self.gwet_linear_or_square == "linear",
            # square_gwet: evapotranspiration will be simulated
            # in both the unsaturated and saturated zones. The groundwater
            # evapotranspiration will be simulated by assuming a constant
            # evapotranspiration rate for groundwater levels between land surface
            # (TOP) and land surface minus the evapotranspiration extinction
            # depth (TOP-EXTDP). Groundwater evapotranspiration is smoothly
            # reduced from the potential evapotranspiration rate to zero over a
            # nominal interval at TOP-EXTDP.
            square_gwet=self.gwet_linear_or_square == "square",
            unsat_etwc=unsat_etwc,
            unsat_etae=unsat_etae,
            simulate_gwseep=False,  # deprecated in favor of drn
            ntrailwaves=self.ntrailwaves,
            nwavesets=self.nwavesets,
            nuzfcells=self._nlay,
            packagedata=uzf_pkdat,
            perioddata=perioddata,
            filename=f"{modflow_gwf.name}.uzf",
            timeseries={
                "filename": f"{modflow_gwf.name.lower()}.uzf_ts",
                "time_series_namerecord": ["finf", "pet"],
            },
            budget_filerecord=budget_filerecord,
            pname=self._name,
            **kwargs,
        )
        logger.info(
            "UZF update_package: vks=%.6g, thtr=%.6g, thts=%.6g, "
            "thti=%.6g, eps=%.6g, extdp=%.6g, extwc(thext)=%.6g",
            vks,
            thtr,
            thts,
            thti,
            eps,
            extdp,
            thext,
        )
        uzf.write()
        self.write_ts(modflow_gwf)

        # simulate surface runoff, originally done by simulate_gwseep in uzf
        if "DRN" in modflow_gwf.get_package_list():
            modflow_gwf.remove_package("DRN")
        top = modflow_gwf.dis.top.array[0][0]
        elev = top - self._surfdep  # top - surfdep
        drn = flopy.mf6.ModflowGwfdrn(
            modflow_gwf,
            save_flows=False,
            maxbound=1,
            stress_period_data={0: [[(0, 0, 0), elev, 1e10]]},
            pname="DRN",
        )
        drn.write()

        # API stuff
        self.api.set_model_name(modflow_gwf.name)
        # Calibration parameters (packagedata scalars, read once at initialize):
        self.api.add_parameters("VKS", "THTR", "THTS", "THTI", "EPS")
        # Period-data variables (written once at kper=0; uzf_rp short-circuits
        # for kper>0 because the file has only one PERIOD block, so values
        # persist for the entire simulation):
        #   EXDP_PVAR  – uzf_rp reads EXTDP from file → this%extdp (EXDP_PVAR).
        #                uzf_ad copies EXDP_PVAR → EXTDPUZ every period, so
        #                calibrated extdp is applied at every solve.
        #   EXTWC_PVAR – same short-circuit logic applies.
        self.api.add_parameters("EXTWC_PVAR", "EXDP_PVAR")
        # Internal geometry arrays: uzf_ar populates these from the file-written
        # DIS/TOP, DIS/BOT, and X during initialize(). They are NOT auto-updated
        # when those values change via the API. Must be overridden explicitly in
        # update_parameters whenever the calibrated d differs from d_init.
        self.api.add_parameters("CELTOP", "CELBOT", "WATAB", "WATABOLD")
        # Companion DRN package (surface-water routing, replaces simulate_gwseep).
        # ELEV = DIS/TOP - SURFDEP; must be overridden once at kper=0.
        self.api_drn.set_model_name(modflow_gwf.name)
        self.api_drn.add_parameters("ELEV")

    def write_ts(self, modflow_gwf: flopy.mf6.ModflowGwf) -> None:
        """Write the UZF time-series file for infiltration and potential ET.

        Parameters
        ----------
        modflow_gwf : flopy.mf6.ModflowGwf
            Active flopy GWF model object.
        """
        # Prepare arrays
        time_array = np.arange(modflow_gwf.nper + 1, dtype=float)
        finf_array = np.append(self.prec, 0.0)
        pet_array = np.append(self.evap, 0.0)  # Ensure evap is positive!
        filepath = Path(modflow_gwf.model_ws) / f"{modflow_gwf.name.lower()}.uzf_ts"

        # Bypass FloPy and write directly
        fast_write_ts(
            filepath=filepath,
            names=["finf", "pet"],
            methods=["stepwise", "stepwise"],
            time_array=time_array,
            data_arrays=[finf_array, pet_array],
        )

    def update_parameters(self, mf6, p):
        """Write calibrated UZF parameters, sync geometry arrays, and init pointers.

        Parameters
        ----------
        mf6 : modflowapi.ModflowApi
            Active MODFLOW 6 API instance.
        p : tuple of float
            ``(H, vks, thtr, thts, thextfrac, eps, extdpfrac)``.
        """
        H, vks, thtr, thts, thextfrac, eps, extdpfrac = p
        thti = (thts + thtr) / 2

        # Resolve pointers for all registered UZF and companion DRN parameters
        # up front so every write below uses the fast ptr path.
        self.api.init_ptrs(mf6)
        self.api_drn.init_ptrs(mf6)

        # Package data: read once by MODFLOW at initialize(), safe to set at kper==0.
        self.api.set_value_ptr("VKS", np.array([vks]))
        self.api.set_value_ptr("THTR", np.array([thtr]))
        self.api.set_value_ptr("THTS", np.array([thts]))
        self.api.set_value_ptr("THTI", np.array([thti]))
        self.api.set_value_ptr("EPS", np.array([eps]))

        # Synchronize UzfCellGroupType geometry with the current (calibrated)
        # DIS and IC values.
        # DIS and IC update_parameters() have already run (dict insertion order:
        # DIS → IC → STO → GHB → UZF), so DIS/TOP, DIS/BOT and X already hold
        # the calibrated values when this method executes.
        top = mf6.get_value(f"{self.api.model_name}/DIS/TOP").ravel()
        bot = mf6.get_value(f"{self.api.model_name}/DIS/BOT").ravel()
        x = mf6.get_value(f"{self.api.model_name}/X").ravel()

        celtop = top - 0.5 * self._surfdep
        celbot = bot.copy().astype(float)
        watab = np.minimum(np.maximum(x, celbot), celtop).astype(float)
        # SPECIAL: uzf_ar sets these from file-read values during initialize();
        # they are NOT auto-updated when DIS/TOP, DIS/BOT, or X change via the
        # API. Must be overridden before prepare_time_step fires setwaves().
        self.api.set_value_ptr("CELTOP", celtop)
        self.api.set_value_ptr("CELBOT", celbot)
        self.api.set_value_ptr("WATAB", watab)
        self.api.set_value_ptr("WATABOLD", watab)
        logger.debug(
            "UZF geometry sync: celtop=%.6g, celbot=%.6g, watab=%.6g",
            celtop[0],
            celbot[0],
            watab[0],
        )

        # Apply period data updates for the first step.
        self.update_period_parameters(mf6, p)

    def update_period_parameters(self, mf6, p):
        """Override EXDP_PVAR, EXTWC_PVAR, and companion DRN ELEV at kper=0.

        The input file defines a single PERIOD block, so uzf_rp short-circuits
        after kper=0. Values written here persist for the whole simulation.
        Uses cached pointers set up by ``update_parameters``.

        Parameters
        ----------
        mf6 : modflowapi.ModflowApi
            Active MODFLOW 6 API instance.
        p : tuple of float
            ``(H, vks, thtr, thts, thextfrac, eps, extdpfrac)``.
        """
        H, vks, thtr, thts, thextfrac, eps, extdpfrac = p
        extdp = extdpfrac * H
        thext = thtr + (thts - thtr) * thextfrac
        # SPECIAL: period-data arrays reset by uzf_rp at kper=0 from file.
        self.api.set_value_ptr("EXDP_PVAR", np.array([extdp]))
        self.api.set_value_ptr("EXTWC_PVAR", np.array([thext]))

        # SPECIAL: companion DRN ELEV (surface-water routing).
        # UZF auto-creates this DRN with file-written ELEV = top - surfdep.
        # In API runs, DIS/TOP changes each optimization step; ELEV must follow.
        # surfdep is a fixed package constant (self._surfdep), not re-read.
        # CROSS-PACKAGE READ: top must come from the live DIS/TOP array so it
        # reflects the calibrated value already written by update_parameters.
        top = mf6.get_value(f"{self.api.model_name}/DIS/TOP").ravel()
        self.api_drn.set_value_ptr("ELEV", (top - self._surfdep).astype(float))

    def stress(self) -> dict[str, Series]:
        """Return precipitation and evaporation input series.

        Returns
        -------
        dict of str to Series
            Keys ``"prec"`` and ``"evap"``.
        """
        return {"prec": self.prec, "evap": self.evap}


class ModflowDrn(ModflowPkgLoggerMixin):
    """DRN package wrapper. Calibrates drain elevation fraction and conductance."""

    def __init__(self):
        self._name = "DRN"
        self.api = ModflowApi(self._name)

    @property
    def parameter_names(self):
        """Return calibration parameter names for this package.

        Returns
        -------
        list of str
            ``["DIS_H", "DRN_C"]`` — elevation fraction and
            conductance.
        """
        return [
            "DIS_H",  # parameter shared with DIS package
            self._name + "_C",
        ]

    def get_init_parameters(self, name: str) -> DataFrame:
        """Return initial parameter bounds and metadata for this package.

        Parameters
        ----------
        name : str
            Stress-model instance name stored in the ``name`` column.

        Returns
        -------
        DataFrame
            Rows indexed by parameter name with columns ``initial``,
            ``pmin``, ``pmax``, ``vary``, ``name``, ``package``, ``dist``.
        """
        parameters = DataFrame(
            {
                "initial": [1.0, 1e-3],
                "pmin": [0.0, 1e-5],
                "pmax": [10.0, 1e-1],
                "vary": [True, True],
                "name": [name, name],
                "package": ["DIS|" + self._name, self._name],
                "dist": ["uniform", "uniform"],
            },
            index=self.parameter_names,
        )
        return parameters

    def update_package(
        self, modflow_gwf: flopy.mf6.ModflowGwf, H: float, C: float
    ) -> None:
        """Write DRN package with drain elevation and conductance.

        Parameters
        ----------
        modflow_gwf : flopy.mf6.ModflowGwf
            Active flopy GWF model object.
        H : float
            Drain elevation.
        C : float
            Drain conductance [L²/T].
        """
        top = modflow_gwf.dis.top.array[0, 0]
        drn = flopy.mf6.ModflowGwfdrn(
            modflow_gwf,
            save_flows=False,
            boundnames=False,
            maxbound=1,
            stress_period_data={0: [[(0, 0, 0), top, C]]},
            pname=self._name,
        )
        drn.write()
        # API stuff
        self.api.set_model_name(modflow_gwf.name)
        self.api.add_parameters("ELEV", "COND")

    def update_parameters(self, mf6, params) -> None:
        """Write calibrated ELEV and COND and resolve pointer cache.

        Parameters
        ----------
        mf6 : modflowapi.ModflowApi
            Active MODFLOW 6 API instance.
        params : tuple of float
            ``(H, C)`` — elevation fraction and conductance.
        """
        H, C = params
        # H is used to compute top of aquifer, use that value as drain elevation.
        # CROSS-PACKAGE READ: requires ModflowDis.update_parameters() to have
        # run first so that DIS/TOP already holds calibrated value.
        top = mf6.get_value(f"{self.api.model_name}/DIS/TOP")
        self.api.init_ptrs(mf6)
        self.api.set_value_ptr("ELEV", np.array([top]))
        self.api.set_value_ptr("COND", np.array([C]))

    def stress(self) -> None:
        """Return None; this package has no stress time series.

        Returns
        -------
        None
        """
        return None


def fast_write_ts(
    filepath: Path | str,
    names: list[str],
    methods: list[str],
    time_array: np.ndarray,
    data_arrays: list[np.ndarray],
) -> None:
    """Write a MODFLOW 6 time-series file directly, bypassing flopy overhead.

    Parameters
    ----------
    filepath : Path or str
        Destination file path.
    names : list of str
        Time-series variable names, e.g. ``["recharge"]``.
    methods : list of str
        Interpolation methods per variable, e.g. ``["stepwise"]``.
    time_array : np.ndarray
        1-D array of simulation times [T] (one entry per stress period
        plus a trailing sentinel value).
    data_arrays : list of np.ndarray
        One 1-D array per variable, same length as *time_array*.
    """
    # 1. Build the MF6 required header
    header = "BEGIN ATTRIBUTES\n"
    header += f"  NAME {' '.join(names)}\n"
    header += f"  METHOD {' '.join(methods)}\n"
    header += "END ATTRIBUTES\n"
    header += "BEGIN TIMESERIES"

    # 2. Stack time and data into a single 2D array
    # MODFLOW requires time in the first column
    stacked_data = np.column_stack([time_array] + data_arrays)

    # 3. Write directly to disk
    with open(filepath, "w") as f:
        f.write(header + "\n")
        # %g drops trailing zeros — keeps file size small and parsing fast
        np.savetxt(f, stacked_data, fmt="%g", delimiter=" ")
        f.write("END TIMESERIES\n")


# %%
