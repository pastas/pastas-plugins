from scipy.special import k0
from scipy.signal import fftconvolve
from typing import Any
from logging import getLogger

from pandas import DataFrame, Series, Timestamp

from pastas.stressmodels import StressModelBase, WellModel
from pastas.rfunc import Hantush

import numpy as np

from pastas.typing import (
    ArrayLike,
    Model,
    RFunc,
    StressSettingsDict,
)

logger = getLogger(__name__)


class WellModel2(WellModel):
    """
    WellModel2 is a subclass of WellModel that uses the response function Hantush
    instead of HantushWellModel. The parameters of WellModel2 are the geohydrological
    parameters T, S and c instead of the parameters A, b and A of the
    HantushWellModel. The distance to the observation point is used to scale the
    response function for each stress, as in WellModel."""

    _name = "WellModel2"

    def __init__(
        self,
        stress: list[Series],
        name: str,
        distances: ArrayLike,
        rfunc: RFunc | None = None,
        up: bool = False,
        settings: str | StressSettingsDict = "well",
        sort_wells: bool = True,
        metadata: list[dict[str, Any]] = None,
        max_cache_size: int = None,
    ) -> None:
        # check response function
        if rfunc is None:
            rfunc = Hantush()
        elif not isinstance(rfunc, Hantush):
            raise NotImplementedError("WellModel2 only supports the rfunc Hantush!")

        # check if number of stresses and distances match
        if len(stress) != len(distances):
            msg = (
                "The number of stresses does not match the number of distances "
                "provided."
            )
            logger.error(msg)
            raise ValueError(msg)
        else:
            self.distances = Series(
                index=[s.squeeze().name for s in stress],
                data=distances,
                name="distances",
            )

        # parse settings input
        if settings is None or isinstance(settings, str) or isinstance(settings, dict):
            settings = len(stress) * [settings]

        # if metadata is passed as dict -> convert to list
        if metadata is not None and isinstance(metadata, dict):
            metadata = [metadata]

        # parse stresses input
        stress = self._handle_stress(stress, settings, metadata)

        # sort wells by distance
        self.sort_wells = sort_wells
        if self.sort_wells:
            stress = [
                s for _, s in sorted(zip(distances, stress), key=lambda pair: pair[0])
            ]
            self.distances.sort_values(inplace=True)

        # estimate gain_scale_factor w/ max of stresses stdev
        gain_scale_factor = np.max([s.series.std() for s in stress])

        tmin = np.min([s.series.index.min() for s in stress])
        tmax = np.max([s.series.index.max() for s in stress])

        StressModelBase.__init__(
            self,
            name=name,
            tmin=tmin,
            tmax=tmax,
            rfunc=rfunc,
            up=up,
            gain_scale_factor=gain_scale_factor,
            max_cache_size=max_cache_size,
        )

        self.stress = stress
        self.freq = self.stress[0].settings["freq"]
        self.set_init_parameters()

    def set_init_parameters(self) -> None:
        parameters = DataFrame(
            [
                (1e3, 1e1, 1e5, True, self.name, "uniform"),
                (1e-4, 1e-6, 1e-1, True, self.name, "uniform"),
                (1e3, 1e1, 1e5, True, self.name, "uniform"),
            ],
            index=[self.name + "_T", self.name + "_S", self.name + "_c"],
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"],
        )
        self.parameters = parameters

    def simulate(
        self,
        p: ArrayLike | None = None,
        tmin: Timestamp | str | None = None,
        tmax: Timestamp | str | None = None,
        freq: str | None = None,
        dt: float = 1.0,
        istress: int | None = None,
    ) -> Series:
        T, S, c = p
        lab = np.sqrt(T * c)
        a = c * S

        distances = self.get_distances(istress=istress)
        stress_df = self.get_stress(
            p=p, tmin=tmin, tmax=tmax, freq=freq, istress=istress, squeeze=False
        )
        h = Series(data=0, index=self.stress[0].series.index, name=self.name)
        for name, r in distances.items():
            stress = stress_df.loc[:, name]
            npoints = stress.index.size
            A = k0(r / lab) / (2 * np.pi * T)
            b = r**2 / (4 * T * c)
            block = self._get_block([A, a, b], dt, tmin, tmax)
            if not self.rfunc.up:
                block = -block
            contrib = fftconvolve(stress, block, "full")[:npoints]
            h = h.add(Series(contrib, index=stress.index), fill_value=0.0)
        if istress is not None:
            if isinstance(istress, list):
                h.name = self.name + "_" + "+".join(str(i) for i in istress)
            elif self.stress[istress].name is not None:
                h.name = self.stress[istress].name
            else:
                h.name = self.name + "_" + str(istress)
        else:
            h.name = self.name
        return h

    def variance_gain(
        self, model: Model, istress: int | None = None, r: ArrayLike | None = None
    ) -> float:
        # not implemented for WellModel2
        raise NotImplementedError("Variance of gain is not implemented for WellModel2!")
