from logging import getLogger
from typing import List, Optional, Tuple, Union

from pandas import Series
from pastas.stressmodels import StressModelBase
from pastas.timeseries import TimeSeries
from pastas.typing import ArrayLike, TimestampType

from .modflow import Modflow

logger = getLogger(__name__)


class ModflowModel(StressModelBase):
    _name = "ModflowModel"

    def __init__(
        self,
        stress: List[Series],
        modflow: Modflow,
        name: str,
        settings: Optional[Tuple[Union[str, dict], Union[str, dict]]] = (
            "prec",
            "evap",
        ),
        metadata: Optional[Tuple[dict, dict]] = (None, None),
        meanstress: Optional[float] = None,
    ) -> None:
        # Set resevoir object
        self.modflow = modflow

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
            logger.error(msg)
            raise Exception(msg)

        # First check the series, then determine tmin and tmax
        stress0.update_series(tmin=index.min(), tmax=index.max())
        stress1.update_series(tmin=index.min(), tmax=index.max())

        if meanstress is None:
            meanstress = (stress0.series - stress1.series).std()

        StressModelBase.__init__(self, name=name, tmin=index.min(), tmax=index.max())
        self.stress.append(stress0)
        self.stress.append(stress1)

        self.freq = stress0.settings["freq"]
        self.set_init_parameters()

    def set_init_parameters(self) -> None:
        """Set the initial parameters back to their default values."""
        self.parameters = self.modflow.get_init_parameters(self.name)

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
        if tmin is None:
            tmin = self.tmin
        if tmax is None:
            tmax = self.tmax

        self.update_stress(tmin=tmin, tmax=tmax, freq=freq)

        return self.stress[0].series, self.stress[1].series

    def simulate(
        self,
        p: ArrayLike,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        freq: Optional[str] = None,
        dt: float = 1.0,
    ):
        stress = self.get_stress(tmin=tmin, tmax=tmax, freq=freq)
        h = self.modflow.simulate(
            p=p,
            stress=stress,
        )
        return Series(
            data=h,
            index=stress[0].index,
            name=self.name,
        )

    def _get_block(self, p, dt, tmin, tmax):
        """Internal method to get the block-response function.
        Cannot be used (yet?) since there is no block response
        """
        # prec = np.zeros(len())
        # evap = np.zeros()
        # return modflow.simulate(np.mean(prec))
        pass
