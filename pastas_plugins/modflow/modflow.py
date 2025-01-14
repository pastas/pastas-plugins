import functools
import logging
from typing import List, Protocol

import flopy
import numpy as np
from pandas import DataFrame, Series
from pastas.typing import ArrayLike

logger = logging.getLogger(__name__)


class Modflow(Protocol):
    def __init__(self) -> None: ...

    def get_init_parameters(self) -> DataFrame: ...

    def create_model(self) -> None: ...

    def simulate(self) -> ArrayLike: ...


class ModflowRch:
    def __init__(
        self, exe_name: str, sim_ws: str, raise_on_modflow_error: bool = False
    ):
        self.exe_name = exe_name
        self.sim_ws = sim_ws
        self._name = "mf_rch"
        self._stress = None
        self._simulation = None
        self._gwf = None
        self._changing_packages = (
            "STO",
            "GHB",
            "RCH",
        )
        self.raise_on_modflow_error = raise_on_modflow_error

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(columns=["initial", "pmin", "pmax", "vary", "name"])
        parameters.loc[name + "_sy"] = (0.05, 0.001, 0.5, True, name)
        parameters.loc[name + "_c"] = (220, 1e1, 1e8, True, name)
        parameters.loc[name + "_f"] = (-1.0, -2.0, 0.0, True, name)
        return parameters

    def create_model(self) -> None:
        sim = flopy.mf6.MFSimulation(
            sim_name=self._name,
            version="mf6",
            exe_name=self.exe_name,
            sim_ws=self.sim_ws,
            lazy_io=True,
        )

        _ = flopy.mf6.ModflowTdis(
            sim,
            time_units="DAYS",
            nper=self._nper,
            perioddata=[(1, 1, 1) for _ in range(self._nper)],
        )

        gwf = flopy.mf6.ModflowGwf(
            sim,
            modelname=self._name,
        )

        _ = flopy.mf6.ModflowIms(
            sim,
            complexity="SIMPLE",
            outer_dvclose=1e-2,
            inner_dvclose=1e-2,
            rcloserecord=1e-1,
            linear_acceleration="BICGSTAB",
            pname=None,
        )
        # sim.register_ims_package(imsgwf, [self._name])

        _ = flopy.mf6.ModflowGwfdis(
            gwf,
            length_units="METERS",
            nlay=1,
            nrow=1,
            ncol=1,
            delr=1,
            delc=1,
            top=1.0,
            botm=0.0,
            idomain=1,
            pname=None,
        )

        _ = flopy.mf6.ModflowGwfnpf(
            gwf, save_flows=False, icelltype=0, k=1.0, pname="npf"
        )

        _ = flopy.mf6.ModflowGwfic(gwf, strt=0.0, pname="ic")

        _ = flopy.mf6.ModflowGwfoc(
            gwf,
            head_filerecord=f"{gwf.name}.hds",
            saverecord=[("HEAD", "ALL")],
            pname=None,
        )

        sim.write_simulation(silent=True)
        self._simulation = sim
        self._gwf = gwf

    def update_model(self, p: ArrayLike):
        sy, c, f = p[0:3]

        d = 0.0
        r = self._stress[0] + f * self._stress[1]

        # remove existing packages
        if all(
            [True for x in self._gwf.get_package_list() if x in self._changing_packages]
        ):
            [self._gwf.remove_package(x) for x in self._changing_packages]

        haq = (self._gwf.dis.top.array - self._gwf.dis.botm.array)[0]
        sto = flopy.mf6.ModflowGwfsto(
            self._gwf,
            save_flows=False,
            iconvert=0,
            ss=sy / haq,
            transient=True,
            pname="sto",
        )
        sto.write()

        # ghb
        ghb = flopy.mf6.ModflowGwfghb(
            self._gwf,
            maxbound=1,
            stress_period_data={0: [[(0, 0, 0), d, 1.0 / c]]},
            pname="ghb",
        )
        ghb.write()

        rts = [(i, x) for i, x in zip(range(self._nper + 1), np.append(r, 0.0))]

        ts_dict = {
            "filename": "recharge.ts",
            "timeseries": rts,
            "time_series_namerecord": ["recharge"],
            "interpolation_methodrecord": ["stepwise"],
        }

        rch = flopy.mf6.ModflowGwfrch(
            self._gwf,
            maxbound=1,
            pname="rch",
            stress_period_data={0: [[(0, 0, 0), "recharge"]]},
            timeseries=ts_dict,
        )
        rch.write()
        rch.ts.write()

        self._gwf.name_file.write()

    @functools.lru_cache(maxsize=5)
    def _get_head(self, p):
        self.update_model(p=p)
        success, _ = self._simulation.run_simulation(silent=True)
        if success:
            return self._gwf.output.head().get_ts((0, 0, 0))[:, 1]
        else:
            logger.error(
                "ModflowError: model run failed with parameters: "
                f"sy={p[0]}, c={p[1]}, f={p[2]}"
            )
            if self.raise_on_modflow_error:
                raise Exception(
                    "Modflow run failed. Check the LIST file for more information."
                )
            else:
                return np.zeros(self._nper)

    def simulate(self, p: ArrayLike, stress: List[Series]) -> ArrayLike:
        if self._simulation is None:
            self._stress = stress
            self._nper = len(self._stress[0])
            self.create_model()
        return self._get_head(tuple(p))
