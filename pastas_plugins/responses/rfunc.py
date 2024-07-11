from typing import Optional

import numpy as np
from pandas import DataFrame
from pastas.rfunc import RfuncBase
from pastas.typing import ArrayLike
from scipy.special import exp1


class Theis(RfuncBase):
    """Theis response function for pumping between two ditches.

    Parameters
    ----------
    cutoff: float, optional
        The cutoff value of the response function.
    nterms: int, optional
        The number of terms to use in the Theis response function.
    **kwargs
        Any other parameter that is passed to the RfuncBase class.
    """

    _name = "Theis"

    def __init__(self, cutoff: float = 0.999, nterms: int = 10, **kwargs) -> None:
        RfuncBase.__init__(self, cutoff=cutoff, **kwargs)
        self.nparam = 3
        self.nterms = nterms

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"]
        )
        if self.up:
            parameters.loc[name + "_A"] = (
                1 / self.gain_scale_factor,
                1e-5,
                100 / self.gain_scale_factor,
                True,
                name,
                "uniform",
            )
        elif self.up is False:
            parameters.loc[name + "_A"] = (
                -1 / self.gain_scale_factor,
                -100 / self.gain_scale_factor,
                -1e-5,
                True,
                name,
                "uniform",
            )
        else:
            parameters.loc[name + "_A"] = (
                1 / self.gain_scale_factor,
                np.nan,
                np.nan,
                True,
                name,
                "uniform",
            )

        parameters.loc[name + "_a"] = (1e2, 0.01, 1e5, True, name, "uniform")
        parameters.loc[name + "_b"] = (1e-3, 1e-3, 0.499999, True, name, "uniform")

        return parameters

    def get_tmax(self, p: ArrayLike, cutoff: Optional[float] = None) -> float:
        if cutoff is None:
            cutoff = self.cutoff
        return -p[1] * np.log(1 - cutoff)

    @staticmethod
    def gain(p: ArrayLike) -> float:
        return p[0]

    def step(
        self,
        p: ArrayLike,
        dt: float = 1.0,
        cutoff: Optional[float] = None,
        maxtmax: Optional[int] = None,
    ) -> ArrayLike:
        t = self.get_t(p=p, dt=dt, cutoff=cutoff, maxtmax=maxtmax)
        # A = Q / (4 * np.pi * T)
        # a = S * L ** 2 / (np.pi ** 2 * T)
        # b = (x - xw) / L with xw = 0
        A = p[0]
        a = p[1]
        b = p[2]

        def theis(A: float, a: float, b: float, t: ArrayLike) -> ArrayLike:
            # works only along the line y=0
            u = a * b**2 * np.pi**2 / (4 * t)
            return A * exp1(u)

        s = theis(A=A, a=a, b=b, t=t)
        for i in range(1, self.nterms + 1, 2):
            s -= theis(A=A, a=a, b=-i + b, t=t) + theis(A=A, a=a, b=i + b, t=t)
            s += theis(A=A, a=a, b=-(i + 1) + b, t=t) + theis(
                A=A, a=a, b=(i + 1) + b, t=t
            )

        return s

    def to_dict(self):
        """Method to export the response function to a dictionary.
        Returns
        -------
        data: dict
            dictionary with all necessary information to reconstruct the rfunc object.
        Notes
        -----
        The exported dictionary should exactly match the input arguments of __init__.
        """
        data = {
            "class": self._name,
            "up": self.up,
            "gain_scale_factor": self.gain_scale_factor,
            "cutoff": self.cutoff,
            "kind": self.kind,
            "t": self.t,
        }
        return data