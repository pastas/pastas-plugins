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

    def __init__(self, *args, rfunc: RFunc | None = None, **kwargs) -> None:
        # call super init to set all properties the same as WellModel, except for rfunc
        super().__init__(*args, **kwargs)
        # check response function
        if rfunc is None:
            rfunc = Hantush()
        elif not isinstance(rfunc, Hantush):
            raise NotImplementedError("WellModel2 only supports the rfunc Hantush!")
        up = self.rfunc.up
        gain_scale_factor = self.rfunc.gain_scale_factor
        rfunc.update_rfunc_settings(up=up, gain_scale_factor=gain_scale_factor)
        self.rfunc = rfunc
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
