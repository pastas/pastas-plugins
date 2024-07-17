from pandas import Series
from pastas.stressmodels import StressModelBase
from pastas.timeseries import TimeSeries


class ReservoirModel(StressModelBase):
    """Time series model consisting of a single reservoir with two stresses.

    The first stress causes the head to go up and the second stress causes
    the head to go down.

    Parameters
    ----------
    stress: list of pandas.Series or list of pastas.timeseries
        list of two pandas.Series or pastas.timeseries objects containing the
        stresses. Usually the first is the precipitation and the second the
        evaporation.
    name: str
        Name of the stress
    settings: list of dicts or strs, optional
        The settings of the stresses. This can be a string referring to a
        predefined settings dict, or a dict with the settings to apply.
        Refer to the docstring of pastas.Timeseries for further information.
        Default is ("prec", "evap").
    metadata: list of dicts, optional
        dictionary containing metadata about the stress. This is passed onto
        the TimeSeries object.

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

    _name = "ReservoirModel"

    def __init__(
        self,
        stress,
        reservoir,
        name,
        meanhead,
        settings=("prec", "evap"),
        metadata=(None, None),
        meanstress=None,
    ):
        # Set resevoir object
        self.reservoir = reservoir(meanhead)

        # Code below is copied from StressModel2 and may not be optimal
        # Check the series, then determine tmin and tmax
        stress0 = TimeSeries(stress[0], settings=settings[0], metadata=metadata[0])
        stress1 = TimeSeries(stress[1], settings=settings[1], metadata=metadata[1])

        # Select indices from validated stress where both series are available.
        index = stress0.series.index.intersection(stress1.series.index)
        if index.empty:
            msg = (
                "The two stresses that were provided have no "
                "overlapping time indices. Please make sure the "
                "indices of the time series overlap."
            )
            # logger.error(msg)
            raise Exception(msg)

        # First check the series, then determine tmin and tmax
        stress0.update_series(tmin=index.min(), tmax=index.max())
        stress1.update_series(tmin=index.min(), tmax=index.max())

        if meanstress is None:
            meanstress = (stress0.series - stress1.series).std()

        StressModelBase.__init__(
            self, name=name, tmin=index.min(), tmax=index.max(), rfunc=None
        )
        self.stress.append(stress0)
        self.stress.append(stress1)

        self.freq = stress0.settings["freq"]
        self.set_init_parameters()

    def set_init_parameters(self):
        """Set the initial parameters back to their default values."""
        self.parameters = self.reservoir.get_init_parameters(self.name)

    def simulate(self, p, tmin=None, tmax=None, freq=None, dt=1, istress=None):
        """Simulates the head contribution.

        Parameters
        ----------
        p: array_like
            array_like object with the values as floats representing the
            model parameters.
        tmin: str, optional
        tmax: str, optional
        freq: str, optional
        dt: float, time step
        istress: int, not used

        Returns
        -------
        pandas.Series
            The simulated head contribution.
        """

        stress = self.get_stress(tmin=tmin, tmax=tmax, freq=freq)
        h = Series(
            data=self.reservoir.simulate(stress[0], stress[1], p),
            index=stress[0].index,
            name=self.name,
            fastpath=True,
        )
        return h

    def get_stress(self, p=None, tmin=None, tmax=None, freq=None, istress=0, **kwargs):
        if tmin is None:
            tmin = self.tmin
        if tmax is None:
            tmax = self.tmax

        self.update_stress(tmin=tmin, tmax=tmax, freq=freq)

        return self.stress[0].series, self.stress[1].series

    def to_dict(self, series=True):
        """Method to export the StressModel object.

        Returns
        -------
        data: dict
            dictionary with all necessary information to reconstruct the
            StressModel object.
        """
        pass

    def _get_block(self, p, dt, tmin, tmax):
        """Internal method to get the block-response function.
        Cannot be used (yet?) since there is no block response"""
        pass
