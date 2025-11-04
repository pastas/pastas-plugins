import logging
from typing import Any, Literal, Protocol, runtime_checkable

import flopy
import numpy as np
from pandas import DataFrame, Series

logger = logging.getLogger(__name__)


@runtime_checkable
class ModflowPackage(Protocol):
    _name: str

    def get_init_parameters(self, name: str) -> DataFrame: ...

    def update_package(
        self, modflow_gwf: flopy.mf6.ModflowGwf, **kwargs: Any
    ) -> None: ...

    def update_parameters(self, mf6, params: tuple, **kwargs: Any) -> None: ...

    def stress(self) -> dict[str, Series] | None: ...


class ModflowDis:
    def __init__(self):
        self._name = "DIS"
        self.parameter_addreses = {}
        self.model_name = None

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            {
                "initial": [0.0, 1.0],
                "pmin": [np.nan, 0.0],
                "pmax": [np.nan, 10.0],
                "vary": [True, False],
                "name": [name, name],
                "dist": ["uniform", "uniform"],
            },
            index=["constant_d", self._name + "_H"],
        )
        return parameters

    def update_package(
        self, modflow_gwf: flopy.mf6.ModflowGwf, d: float, H: float = 1.0
    ) -> None:
        """Update the discretization package."""
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
        self.model_name = modflow_gwf.name.upper()
        self.parameter_addreses = {
            "top": f"{self.model_name}/{self._name}/TOP",
            "bot": f"{self.model_name}/{self._name}/BOT",
        }

    def update_parameters(self, mf6, params: tuple) -> None:
        d, H = params
        bot = np.array([d - 100.0])
        top = np.array([d + H])
        mf6.set_value(self.parameter_addreses["top"], top)
        mf6.set_value(self.parameter_addreses["bot"], bot)

    def stress(self) -> None:
        return None


class ModflowIc:
    def __init__(self):
        self._name = "IC"
        self.parameter_addreses = {}
        self.model_name = None

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            {
                "initial": [0.0],
                "pmin": [np.nan],
                "pmax": [np.nan],
                "vary": [True],
                "name": [name],
                "dist": ["uniform"],
            },
            index=["constant_d"],
        )
        return parameters

    def update_package(self, modflow_gwf: flopy.mf6.ModflowGwf, d: float) -> None:
        """Update the initial conditions package."""
        ic = flopy.mf6.ModflowGwfic(modflow_gwf, strt=d, pname=self._name)
        ic.write()
        self.model_name = modflow_gwf.name.upper()
        self.parameter_addreses = {"d": f"{self.model_name}/{self._name}/STRT"}

    def update_parameters(self, mf6, params) -> None:
        (d,) = params
        mf6.set_value(self.parameter_addreses["d"], np.array([d]))

    def stress(self) -> None:
        return None


class ModflowSto:
    def __init__(self):
        self._name = "STO"
        self.parameter_addreses = {}
        self.model_name = None

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            {
                "initial": [0.1],
                "pmin": [0.001],
                "pmax": [0.5],
                "vary": [True],
                "name": [name],
                "dist": ["uniform"],
            },
            index=[self._name + "_S"],
        )
        return parameters

    def update_package(self, modflow_gwf: flopy.mf6.ModflowGwf, S: float) -> None:
        """Update the storage package."""
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
        self.model_name = modflow_gwf.name.upper()
        self.parameter_addreses = {"S": f"{modflow_gwf.name.upper()}/{self._name}/SS"}

    def update_parameters(self, mf6, params) -> None:
        (S,) = params
        haq = mf6.get_value(f"{self.model_name}/DIS/TOP") - mf6.get_value(
            f"{self.model_name}/DIS/BOT"
        )
        mf6.set_value(self.parameter_addreses["S"], S / haq)

    def stress(self) -> None:
        return None


class ModflowGhb:
    def __init__(self):
        self._name = "GHB"
        self.parameter_adresses = {}

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            {
                "initial": [0.0, 1e-3],
                "pmin": [np.nan, 1e-5],
                "pmax": [np.nan, 1e-1],
                "vary": [True, True],
                "name": [name, name],
                "dist": ["uniform", "uniform"],
            },
            index=["constant_d", self._name + "_C"],
        )
        return parameters

    def update_package(
        self, modflow_gwf: flopy.mf6.ModflowGwf, d: float, C: float
    ) -> None:
        ghb = flopy.mf6.ModflowGwfghb(
            modflow_gwf,
            maxbound=1,
            stress_period_data={0: [[(0, 0, 0), d, C]]},
            pname=self._name,
        )
        ghb.write()
        self.parameter_adresses = {
            "d": f"{modflow_gwf.name.upper()}/{self._name}/BHEAD",
            "C": f"{modflow_gwf.name.upper()}/{self._name}/COND",
        }

    def update_parameters(self, mf6, params) -> None:
        d, C = params
        mf6.set_value(self.parameter_adresses["d"], np.array([d]))
        mf6.set_value(self.parameter_adresses["C"], np.array([C]))

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

        # computation variables
        self.recharge = None
        self.parameter_adresses = {}
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
            index=[self._name + "_f"],
        )
        return parameters

    def compute_recharge(self, f: float) -> Series:
        self.recharge = (self.prec + f * self.evap).to_numpy()

    def update_package(self, modflow_gwf: flopy.mf6.ModflowGwf, f: float) -> None:
        use_timeseries = True
        rech = self.prec + f * self.evap
        rts = list(zip(range(modflow_gwf.nper + 1), np.append(rech, 0.0)))
        ts_dict = {
            "filename": f"{modflow_gwf.name}.rch_ts",
            "timeseries": rts,
            "time_series_namerecord": ["recharge"],
            "interpolation_methodrecord": ["stepwise"],
        }
        if use_timeseries:
            spd = {0: [[(0, 0, 0), "recharge"]]}
            maxbound = 1
        else:
            # spd = {i: [[(0, 0, 0), v]] for i, v in enumerate(rech)}
            spd = {0: [[(0, 0, 0), 0.0]]}
            maxbound = 1
            ts_dict = None

        rch = flopy.mf6.ModflowGwfrch(
            modflow_gwf,
            maxbound=maxbound,
            stress_period_data=spd,
            timeseries=ts_dict,
            pname=self._name,
        )
        rch.write()
        if use_timeseries:
            rch.ts.write()

        self.model_name = modflow_gwf.name.upper()
        self.parameter_adresses["recharge"] = f"{self.model_name}/{self._name}/RECHARGE"

    def update_timeseries(self, mf6, kper) -> None:
        # rech = (
        #     self.prec.iloc[kper : kper + 1] + p[0] * self.evap.iloc[kper : kper + 1]
        # ).to_numpy()
        mf6.set_value(
            self.parameter_adresses["recharge"], self.recharge[kper : kper + 1]
        )
        # rts = list(zip(range(model.nper + 1), np.append(rech, 0.0)))
        # ts = flopy.mf6.ModflowUtlts(
        #     rch,
        #     time_series_namerecord=["recharge"],
        #     interpolation_methodrecord=["stepwise"],
        #     timeseries=rts,
        #     filename=f"{model.name.lower()}.rch_ts",
        # )
        # ts.write()

    def stress(self) -> dict[str, Series]:
        return {"prec": self.prec, "evap": self.evap}


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
                name + "_H",
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
        H: float,
        vks: float,
        thtr: float,
        thts: float,
        eps: float,
        thextfrac: float,
        extdpfrac: float,
    ) -> None:
        extdp = extdpfrac * H
        thext = thtr + (thts - thtr) * thextfrac

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
        surfdep = 1e-5  # surface depression depth
        uzf_pkdat = [
            [
                n,  # iuzno
                (0, 0, 0),  # gwf_cellid
                1 if n == 0 else 0,  # landflag
                n + 1 if (n + 1) != nlay else -1,  # ivertcon
                surfdep,  # surface depression depth
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
                range(modflow_gwf.nper + 1), np.append(finf, 0.0), np.append(pet, 0.0)
            )
        ]
        ts_dict = {
            "filename": f"{modflow_gwf.name}.uzf_ts",
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
            modflow_gwf,
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
            filename=f"{modflow_gwf.name}.uzf",
            pname=self._name,
        )
        uzf.write()
        uzf.ts.write()

        # simulate surface runoff, originally done by simulate_gwseep in uzf
        if "DRN" in modflow_gwf.get_package_list():
            modflow_gwf.remove_package("DRN")
        top = modflow_gwf.dis.top.array[0][0]
        elev = top - surfdep  # top - surfdep
        drn = flopy.mf6.ModflowGwfdrn(
            modflow_gwf,
            save_flows=False,
            maxbound=1,
            stress_period_data={0: [[(0, 0, 0), elev, 1e10]]},
            pname="DRN",
        )
        drn.write()

    def stress(self) -> dict[str, Series]:
        return {"prec": self.prec, "evap": self.evap}


class ModflowDrn:
    def __init__(self):
        self._name = "DRN"

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            {
                "initial": [0.5, 1e-3],
                "pmin": [0.0, 1e-5],
                "pmax": [1.0, 1e-1],
                "vary": [True, True],
                "name": [name, name],
                "dist": ["uniform", "uniform"],
            },
            index=[name + "_drnHfrac", name + "_drnC"],
        )
        return parameters

    def update_package(
        self, modflow_gwf: flopy.mf6.ModflowGwf, drnHfrac: float, drnC: float
    ) -> None:
        top = modflow_gwf.dis.top.array[0, 0]
        botm = modflow_gwf.dis.botm.array[0, 0, 0]
        drnH = botm + drnHfrac * (top - botm)
        drn = flopy.mf6.ModflowGwfdrn(
            modflow_gwf,
            print_input=True,
            print_flows=True,
            save_flows=False,
            boundnames=True,
            maxbound=1,
            stress_period_data={0: [[(0, 0, 0), drnH, drnC]]},
            pname=self._name,
        )
        drn.write()

    def stress(self) -> None:
        return None


class ModflowSto2:
    def __init__(self):
        self._name = "STO"

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            {
                "initial": [0.5, 0.3, 0.1],
                "pmin": [0.0, 0.01, 0.001],
                "pmax": [1.0, 1.0, 0.5],
                "vary": [True, True, True],
                "name": [name, name, True],
                "dist": ["uniform", "uniform", "uniform"],
            },
            index=[name + "_drnSfrac", name + "_drnS", name + "_S"],
        )
        return parameters

    def update_package(
        self, modflow_gwf: flopy.mf6.ModflowGwf, drnSfrac: float, drnS: float, S: float
    ) -> None:
        top = modflow_gwf.dis.top.array[0, 0]
        botm = modflow_gwf.dis.botm.array[0, 0, 0]
        drnH = botm + drnSfrac * (top - botm)
        haq = top - botm

        sto = flopy.mf6.ModflowGwfsto(
            modflow_gwf,
            save_flows=False,
            iconvert=1,
            ss=drnS / haq,
            sy=S,
            transient=True,
            ss_confined_only=True,
            pname=self._name,
        )
        sto.write()

        # TODO: This is probably not correct, double check with old method
        ModflowDis().update_package(modflow_gwf, d=botm, H=top + drnH)

    def stress(self) -> None:
        return None


class ModflowDrnSto:
    def __init__(self):
        self._name = "DRN"

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            {
                "initial": [0.0, 1e-3, 0.3, 0.1],
                "pmin": [0.0, 1e-5, 0.01, 0.001],
                "pmax": [1.0, 1e-1, 0.5, 0.5],
                "vary": [True, True, True, True],
                "name": [name, name, name, name],
                "dist": ["uniform", "uniform", "uniform", "uniform"],
            },
            index=[name + "_drnHfrac", name + "_drnC", name + "_drnS", name + "_S"],
        )
        return parameters

    def update_package(
        self,
        modflow_gwf: flopy.mf6.ModflowGwf,
        drnHfrac: float,
        drnC: float,
        drnS: float,
        S: float,
    ) -> None:
        ModflowDrn().update_package(modflow_gwf, drnHfrac=drnHfrac, drnC=drnC)

        if "STO" in modflow_gwf.get_package_list():
            modflow_gwf.remove_package("STO")
        top = modflow_gwf.dis.top.array[0, 0]
        botm = modflow_gwf.dis.botm.array[0, 0, 0]
        haq = top - botm
        sto = flopy.mf6.ModflowGwfsto(
            modflow_gwf,
            save_flows=False,
            iconvert=1,
            ss=drnS / haq,
            sy=S,
            transient=True,
            ss_confined_only=True,
            pname="STO",
        )
        sto.write()

    def stress(self) -> None:
        return None
