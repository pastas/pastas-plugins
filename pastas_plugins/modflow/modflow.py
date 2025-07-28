import logging
from typing import Any, Literal, Protocol

import flopy
import numpy as np
from pandas import DataFrame, Series
from pastas.typing import ArrayLike

logger = logging.getLogger(__name__)


class ModflowPackage(Protocol):
    name: str

    def get_init_parameters(self, name: str) -> DataFrame: ...

    def update_package(
        self, modflow_gwf: flopy.mf6.ModflowGwf, p: ArrayLike
    ) -> Any: ...

    def stress() -> list[Series] | None: ...


class ModflowGhb:
    def __init__(self):
        self._name = "GHB"

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            {
                "initial": [1.0, 1e-3],
                "pmin": [0.0, 1e-5],
                "pmax": [10.0, 1e-1],
                "vary": [True, True],
                "name": [name, name],
                "dist": ["uniform", "uniform"],
            },
            index=[name + "_D", name + "_C"],
        )
        return parameters

    def update_package(
        self, modflow_gwf: flopy.mf6.ModflowGwf, d: float, C: float
    ) -> flopy.mf6.ModflowGwfghb:
        ghb = flopy.mf6.ModflowGwfghb(
            modflow_gwf,
            maxbound=1,
            stress_period_data={0: [[(0, 0, 0), d, C]]},
            pname=self._name,
        )
        ghb.write()
        return ghb

    def stress(self) -> None:
        return None


class ModflowRch:
    def __init__(
        self,
        prec: Series,
        evap: Series,
    ):
        self._name = "RCH"
        self.prec = prec
        self.evap = evap
        # index prec and evap on the correct times

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            {
                "initial": [-1.0],
                "pmin": [-2.0],
                "pmax": [0.0],
                "vary": [True],
                "name": [name],
                "dist": ["uniform"],
            },
            index=[name + "_f"],
        )
        return parameters

    def update_package(self, modflow_gwf: flopy.mf6.ModflowGwf, f: float):
        self._remove_changing_package("RCH")
        rech = self.prec + f * self.evap
        rts = [(i, x) for i, x in zip(range(self._nper + 1), np.append(rech, 0.0))]

        ts_dict = {
            "filename": f"{self._gwf.name}.rch_ts",
            "timeseries": rts,
            "time_series_namerecord": ["recharge"],
            "interpolation_methodrecord": ["stepwise"],
        }

        rch = flopy.mf6.ModflowGwfrch(
            modflow_gwf,
            maxbound=1,
            pname=self.name,
            stress_period_data={0: [[(0, 0, 0), "recharge"]]},
            timeseries=ts_dict,
        )
        rch.write()
        rch.ts.write()

    def stress(self) -> list[Series]:
        return [self.prec, self.evap]


class ModflowUzf:
    def __init__(
        self,
        prec: Series,
        evap: Series,
        simulate_et: bool = True,
        gwet_linear_or_square: None | Literal["linear", "square"] = "linear",
        ntrailwaves: int = 7,
        nwavesets: int = 40,
    ):
        self._name = "UZF"
        self.prec = prec
        self.evap = evap
        self.simulate_et = simulate_et
        self.gwet_linear_or_square = gwet_linear_or_square
        self.ntrailwaves = ntrailwaves
        self.nwavesets = nwavesets

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            {
                "initial": [1.0, 1.0, 0.1, 0.3, 0.1, 5.0, 0.5],
                "pmin": [0.01, 0.0, 0.0, 0.2, 0.0, 3.5, 0.0],
                "pmax": [10.0, 10.0, 0.2, 0.4, 1.0, 10.0, 1.0],
                "vary": [True] * 7,
                "name": [name] * 7,
                "dist": ["uniform"] * 7,
            },
            index=[
                name + "_height",
                name + "_vks",
                name + "_thtr",
                name + "_thts",
                name + "_thextfrac",
                name + "_eps",
                name + "_extdpfrac",
            ],
        )
        return parameters

    def update_package(
        self,
        modflow_gwf: flopy.mf6.ModflowGwf,
        vks: float,
        thts: float,
        thtr: float,
        thext: float,
        eps: float,
        extdp: float,
    ):
        # d, C, s, height, vks, thtr, thts, thextfrac, eps, extdpfrac = p[0:10]
        # extdp = extdpfrac * height
        # thext = thtr + (thts - thtr) * thextfrac

        self._remove_changing_package("UZF")
        finf = self.prec
        pet = self.evap  # make sure et is positive!

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
        rootact = 0.0  # the length of roots in a given volume of soil divided by that volume [L^-2]

        nlay = 1  # only one uzf cell / layer

        uzf_pkdat = [
            [
                n,  # iuzno
                (0, 0, 0),  # gwf_cellid
                1 if n == 0 else 0,  # landflag
                n + 1 if (n + 1) != nlay else -1,  # ivertcon
                1e-5,  # surface depression depth
                vks,  # vertical saturated hydraulic conductivity
                thtr,  # residual water content
                thts,  # saturated water content
                thti,  # initial water content
                eps,  # brooks-corey epsilon exponent
                f"CELLID_UZF_{n:03d}",  # boundname
            ]
            for n in range(nlay)
        ]

        uzfts = [
            (i, finfi, peti)
            for i, finfi, peti in zip(
                range(self._nper + 1), np.append(finf, 0.0), np.append(pet, 0.0)
            )
        ]
        ts_dict = {
            "filename": f"{self._gwf.name}.uzf_ts",
            "timeseries": uzfts,
            "time_series_namerecord": ["finf", "pet"],
            "interpolation_methodrecord": ["stepwise", "stepwise"],
        }
        perioddata = {
            0: [
                [n, "finf", "pet", extdp, thext, ha, hroot, rootact]
                for n in range(nlay)
            ]
        }

        uzf = flopy.mf6.ModflowGwfuzf(
            self._gwf,
            print_input=True,
            print_flows=True,
            save_flows=False,
            boundnames=True,
            # If this option is selected, evapotranspiration will be simulated
            # in the unsaturated zone but not in the saturated zone.
            simulate_et=True,
            # If this option is selected, evapotranspiration will be simulated
            # in both the unsaturated and saturated zones. The groundwater
            # evapotranspiration will be simulated using the original ET
            # formulation of MODFLOW-2005.
            linear_gwet=self.gwet_linear_or_square == "linear",
            # square_gwet: If this option is selected, evapotranspiration will be simulated
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
            nuzfcells=nlay,
            packagedata=uzf_pkdat,
            perioddata=perioddata,
            timeseries=ts_dict,
            pname="uzf",
            filename=f"{self._name}.uzf",
        )
        uzf.write()
        uzf.ts.write()

        self._remove_changing_package("DRN")
        top = self._gwf.dis.top.array[0][0]
        elev = top - 1e-5  # top - surfdep
        drn = flopy.mf6.ModflowGwfdrn(
            self._gwf,
            save_flows=False,
            maxbound=1,
            stress_period_data={0: [[(0, 0, 0), elev, 1e10]]},
            pname="drn",
        )
        drn.write()

    def stress(self) -> list[Series]:
        return [self.prec, self.evap]


class ModflowDrn:
    def __init__(self, **kwargs):
        self._name = "DRN"

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            {
                "initial": [1.0, 1e-3],
                "pmin": [0.0, 1e-5],
                "pmax": [10.0, 1e-1],
                "vary": [True, True],
                "name": [name, name],
                "dist": ["uniform", "uniform"],
            },
            index=[name + "_h_drn", name + "_C_drn"],
        )
        return parameters

    def update_drn(self, d: float, C: float) -> None:
        drn = flopy.mf6.ModflowGwfdrn(
            self._gwf,
            print_input=True,
            print_flows=True,
            save_flows=False,
            boundnames=True,
            maxbound=1,
            stress_period_data={0: [[(0, 0, 0), d, C]]},
            pname="drn",
        )
        drn.write()

    def stress(self) -> None:
        return None


class ModflowSto:
    def __init__(self):
        self._name = "STO"

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            {
                "initial": [1.0, 0.3],
                "pmin": [0.0, 0.001],
                "pmax": [10.0, 1.0],
                "vary": [True, True],
                "name": [name, name],
                "dist": ["uniform", "uniform"],
            },
            index=[name + "_h_drn", name + "_s_drn"],
        )
        return parameters

    def update_package(self, modflow_gwf: flopy.mf6.ModflowGwf, s: float, s_drn: float):
        self._remove_changing_package("STO")
        haq = (modflow_gwf.dis.top.array - modflow_gwf.dis.botm.array)[0]
        sto = flopy.mf6.ModflowGwfsto(
            modflow_gwf,
            save_flows=False,
            iconvert=1,
            ss=s_drn / haq,
            sy=s,
            transient=True,
            pname=self._name,
            ss_confined_only=True,
        )
        sto.write()

    def stress(self) -> None:
        return None


class ModflowDrnSto(ModflowDrn, ModflowSto):
    def __init__(self):
        self._name = "DRN_STO"

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = ModflowDrn.get_init_parameters(self, name)
        parameters.loc[name + "_s_drn"] = (0.3, 0.001, 1.0, True, name, "uniform")
        return parameters

    def update_model(self, p: ArrayLike):
        if self.constant_d_from_modflow:
            d = p[0]
            p = p[1:]
            self.update_ic(d=d)
        else:
            d = 0.0
        C, s, f, h_drn, c_drn, s_drn = p
        self.update_dis(d=0, height=d + h_drn)
        self.update_sto(s=s, s_drn=s_drn)
        self.update_ghb(d=d, C=C)
        self.update_rch(f=f)
        self.update_drn(d=d + h_drn, c=c_drn)
        self._gwf.name_file.write()

    def stress(self) -> None:
        return None
