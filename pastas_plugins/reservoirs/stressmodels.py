from collections import namedtuple
from typing import Any

from pandas import DataFrame, Series, Timestamp
from pastas.model import Model
from pastas.stressmodels import StressModelBase
from pastas.timeseries import TimeSeries
from pastas.typing import ArrayLike, StressSettingsDict

from pastas_plugins.reservoirs import reservoir as reservoir_module

# Define namedtuple at the module level to make it picklable
PrecEvapTuple = namedtuple("StressesTuple", ("prec", "evap"))


class ReservoirModel(StressModelBase):
    """Time series model consisting of a single reservoir with two stresses.

    The first stress causes the head to go up and the second stress causes
    the head to go down.

    Parameters
    ----------
    model: pastas.Model
        The Pastas Model instance to which the reservoir model is added.
    stress: list of pandas.Series or list of pastas.timeseries
        list of two pandas.Series or pastas.timeseries objects containing the
        stresses. Usually the first is the precipitation and the second the
        evaporation.
    reservoir: ReservoirBase class or str
        The reservoir class (e.g. Reservoir1) or the name of a reservoir class
        defined in pastas_plugins.reservoirs.reservoir (e.g. "Reservoir1").
    name: str
        Name of the stress
    meanhead: float
        Mean head of the reservoir, used to set the initial value and bounds of
        the reservoir's d parameter.
    settings: list of dicts or strs, optional
        The settings of the stresses. This can be a string referring to a
        predefined settings dict, or a dict with the settings to apply.
        Refer to the docstring of pastas.TimeSeries for further information.
        Default is ("prec", "evap").
    metadata: list of dicts, optional
        dictionary containing metadata about the stress. This is passed onto
        the TimeSeries object.
    meanstress: float, optional
        Mean of the stress difference, kept for backward compatibility. Not
        used since there is no response function.

    Notes
    -----
    The order in which the stresses are provided is the order the metadata
    and settings dictionaries or string are passed onto the TimeSeries
    objects. By default, the precipitation stress is the first and the
    evaporation stress the second stress.

    See Also
    --------
    pastas.timeseries
    """

    def __init__(
        self,
        model: Model,
        stress,
        reservoir,
        name: str,
        meanhead: float,
        settings=("prec", "evap"),
        metadata=(None, None),
        meanstress=None,
    ):
        _ = meanstress  # not used, kept for backward compatibility
        if isinstance(reservoir, str):
            reservoir = getattr(reservoir_module, reservoir)

        # Set reservoir object
        self.reservoir = reservoir(meanhead)

        # Set the precipitation and evaporation time series
        self.set_stress(prec=stress[0], settings=settings[0], metadata=metadata[0])
        self.set_stress(evap=stress[1], settings=settings[1], metadata=metadata[1])

        # Select indices from validated stress where both series are available.
        index = self.prec.series.index.intersection(self.evap.series.index)
        if index.empty:
            msg = (
                "The two stresses that were provided have no "
                "overlapping time indices. Please make sure the "
                "indices of the time series overlap."
            )
            raise ValueError(msg)

        # First check the series, then determine tmin and tmax
        self.prec.update_series(tmin=index.min(), tmax=index.max())
        self.evap.update_series(tmin=index.min(), tmax=index.max())

        super().__init__(
            model=model,
            name=name,
            tmin=index.min(),
            tmax=index.max(),
            rfunc=None,
        )
        self.set_init_parameters()
        if self.model is not None:
            self.model._add_stressmodel(self)

    @property
    def prec(self) -> TimeSeries:
        """Precipitation time series."""
        return self._prec

    @prec.setter
    def prec(self, series: Series) -> None:
        """Set the precipitation time series."""
        self.set_stress(prec=series)

    @property
    def evap(self) -> TimeSeries:
        """Evaporation time series."""
        return self._evap

    @evap.setter
    def evap(self, series: Series) -> None:
        """Set the evaporation time series."""
        self.set_stress(evap=series)

    @property
    def stresses(self) -> tuple[TimeSeries, ...]:
        """All the stress time series in the stressmodel as a tuple."""
        return PrecEvapTuple(prec=self.prec, evap=self.evap)

    @property
    def nsplit(self) -> int:
        """Number of time series the contribution can be split in."""
        return 1

    def set_stress(
        self,
        prec: Series | None = None,
        evap: Series | None = None,
        settings: str | StressSettingsDict | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Set the stress time series for the ReservoirModel.

        Parameters
        ----------
        prec: pandas.Series, optional
            pandas.Series with pandas.DatetimeIndex containing the precipitation
            series.
        evap: pandas.Series, optional
            pandas.Series with pandas.DatetimeIndex containing the evaporation
            series.
        settings: str or dict, optional
            The settings of the time series. This can be a string referring to a
            predefined settings dictionary (defined in ps.timeseries.settings), or
            a dictionary with the settings to apply.
        metadata: dict, optional
            Dictionary with metadata of the stress time series.
        """

        def _set_stress(
            attr_name: str,
            stress: Series,
            settings: str | StressSettingsDict | None,
            metadata: dict[str, Any] | None,
        ) -> None:
            if hasattr(self, attr_name):
                if getattr(self, attr_name).metadata is not None and metadata is None:
                    metadata = getattr(self, attr_name).metadata
                if getattr(self, attr_name).settings is not None and settings is None:
                    settings = getattr(self, attr_name).settings
            setattr(
                self,
                attr_name,
                TimeSeries(stress, settings=settings, metadata=metadata),
            )

        if sum([prec is not None, evap is not None]) > 1:
            msg = "Only one of prec or evap can be set at a time."
            raise ValueError(msg)

        if prec is not None:
            _set_stress("_prec", prec, settings, metadata)
        elif evap is not None:
            _set_stress("_evap", evap, settings, metadata)
        else:
            msg = "No stress time series provided to set. Please provide either prec or evap."
            raise ValueError(msg)

    def set_init_parameters(self) -> None:
        """Set the initial parameters back to their default values."""
        self.parameters = self.reservoir.get_init_parameters(self.name)

    def simulate(
        self,
        p: ArrayLike,
        tmin: Timestamp | str | None = None,
        tmax: Timestamp | str | None = None,
        freq: str | None = None,
        dt: float = 1.0,
        istress: int | None = None,
    ) -> Series:
        """Simulates the head contribution.

        Parameters
        ----------
        p: array_like
            array_like object with the values as floats representing the
            model parameters.
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        freq: str, optional
        dt: float, time step
        istress: int, not used

        Returns
        -------
        pandas.Series
            The simulated head contribution.
        """
        _ = dt, istress
        stresses = self.get_stress(tmin=tmin, tmax=tmax, freq=freq)
        prec = stresses["prec"]
        evap = stresses["evap"]
        h = Series(
            data=self.reservoir.simulate(prec, evap, p),
            index=prec.index,
            name=self.name,
        )
        return h

    def get_stress(
        self,
        p: ArrayLike | None = None,
        tmin: Timestamp | str | None = None,
        tmax: Timestamp | str | None = None,
        freq: str | None = None,
        istress: int | None = None,
        **kwargs,
    ) -> DataFrame:
        """Get the precipitation and evaporation stress series.

        Returns
        -------
        pandas.DataFrame
            DataFrame with the precipitation and evaporation series as columns.
        """
        _ = p, istress, kwargs
        tmin = self.tmin if tmin is None else tmin
        tmax = self.tmax if tmax is None else tmax

        self.update_stress(tmin=tmin, tmax=tmax, freq=freq)

        return DataFrame({"prec": self.prec.series, "evap": self.evap.series})

    def to_dict(self, series: bool = True) -> dict[str, Any]:
        """Export the ReservoirModel to a dictionary.

        Parameters
        ----------
        series: bool, optional
            Whether to include the time series data of the stresses in the
            dictionary. Default is True.

        Returns
        -------
        data: dict
            dictionary with all necessary information to reconstruct the
            ReservoirModel object.
        """
        data = super().to_dict() | {
            "stress": [ts.to_dict(series=series) for ts in self.stresses],
            "reservoir": self.reservoir._name,
            "meanhead": self.reservoir.initialhead,
        }
        return data
