from abc import abstractmethod
import functools
import logging
from typing import List, Literal

import flopy
import numpy as np
from pandas import DataFrame, Series
from pastas.typing import ArrayLike

logger = logging.getLogger(__name__)


class Modflow:
    def __init__(
        self,
        exe_name: str,
        sim_ws: str,
        head: Series | None = None,
        raise_on_modflow_error: bool = False,
    ) -> None:
        self.exe_name = exe_name
        self.sim_ws = sim_ws
        self.raise_on_modflow_error = raise_on_modflow_error
        self._simulation = None
        self._gwf = None
        self._name = "mf_base"
        self._stress = None
        self._nper = None
        if head is not None:
            logger.info(
                "Make sure to delete the model parameter constant_d"
                "(`ml.del_constant()`). Base elevation is now controled by"
                "parameter `_d`."
            )
            self._head = head
            self.constant_d_from_modflow = True
            self._changing_packages = ("DIS", "IC", "STO", "GHB")
        else:
            self._head = None
            self.constant_d_from_modflow = False
            self._changing_packages = ("STO", "GHB")

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"]
        )
        if self.constant_d_from_modflow:
            parameters.loc[name + "_d"] = (
                float(np.mean(self._head)),
                min(self._head),
                max(self._head),
                True,
                name,
                "uniform",
            )
        parameters.loc[name + "_c"] = (220, 1e1, 1e8, True, name, "uniform")
        parameters.loc[name + "_s"] = (0.05, 0.001, 0.5, True, name, "uniform")
        return parameters

    def base_model(self) -> None:
        self._simulation = flopy.mf6.MFSimulation(
            sim_name=self._name,
            version="mf6",
            exe_name=self.exe_name,
            sim_ws=self.sim_ws,
            lazy_io=True,
        )

        _ = flopy.mf6.ModflowTdis(
            self._simulation,
            time_units="DAYS",
            nper=self._nper,
            perioddata=[(1, 1, 1) for _ in range(self._nper)],
        )

        self._gwf = flopy.mf6.ModflowGwf(
            self._simulation,
            modelname=self._name,
        )

        _ = flopy.mf6.ModflowIms(
            self._simulation,
            complexity="SIMPLE",
            outer_dvclose=1e-2,
            inner_dvclose=1e-2,
            rcloserecord=1e-1,
            linear_acceleration="BICGSTAB",
            pname=None,
        )
        # sim.register_ims_package(imsgwf, [self._name])

        _ = flopy.mf6.ModflowGwfnpf(
            self._gwf, save_flows=False, icelltype=0, k=1.0, pname="npf"
        )

        _ = flopy.mf6.ModflowGwfoc(
            self._gwf,
            head_filerecord=f"{self._gwf.name}.hds",
            saverecord=[("HEAD", "ALL")],
            pname=None,
        )

        self._simulation.write_simulation(silent=True)

        if not self.constant_d_from_modflow:
            self.update_dis(d=0.0, height=1.0)
            self.update_ic(d=0.0)

    @abstractmethod
    def update_model(self, p: ArrayLike) -> None:
        pass

    @functools.lru_cache(maxsize=5)
    def get_head(self, p):
        self.update_model(p=p)
        success, _ = self._simulation.run_simulation(silent=True)
        if success:
            return self._gwf.output.head().get_ts((0, 0, 0))[:, 1]
        else:
            logger.error("ModflowError: model run failed with parameters: %s" % str(p))
            if self.raise_on_modflow_error:
                raise Exception(
                    "Modflow run failed. Check the LIST file for more information."
                )
            else:
                return np.zeros(self._nper)

    def simulate(self, p: ArrayLike, stress: List[Series]) -> ArrayLike:
        if self._simulation is None:
            self._stress = stress
            self._nper = len(stress[0])
            self.base_model()
        return self.get_head(tuple(p))

    def remove_changing_packages(self):
        for cp in self._changing_packages:
            if cp in self._gwf.get_package_list():
                self._gwf.remove_package(cp)

    def update_dis(self, d: float, height: float = 1.0):
        dis = flopy.mf6.ModflowGwfdis(
            self._gwf,
            length_units="METERS",
            nlay=1,
            nrow=1,
            ncol=1,
            delr=1,
            delc=1,
            top=d + height,
            botm=d,
            idomain=1,
            pname="dis",
        )
        dis.write()

    def update_ic(self, d: float):
        ic = flopy.mf6.ModflowGwfic(self._gwf, strt=d, pname="ic")
        ic.write()

    def update_sto(self, s: float):
        haq = (self._gwf.dis.top.array - self._gwf.dis.botm.array)[0]
        sto = flopy.mf6.ModflowGwfsto(
            self._gwf,
            save_flows=False,
            iconvert=0,
            ss=s / haq,
            transient=True,
            pname="sto",
        )
        sto.write()

    def update_ghb(self, d: float, c: float):
        # ghb
        ghb = flopy.mf6.ModflowGwfghb(
            self._gwf,
            maxbound=1,
            stress_period_data={0: [[(0, 0, 0), d, 1.0 / c]]},
            pname="ghb",
        )
        ghb.write()


class ModflowRch(Modflow):
    def __init__(
        self,
        exe_name: str,
        sim_ws: str,
        head: Series | None = None,
        raise_on_modflow_error: bool = False,
    ):
        Modflow.__init__(
            self,
            exe_name=exe_name,
            sim_ws=sim_ws,
            head=head,
            raise_on_modflow_error=raise_on_modflow_error,
        )
        self._name = "mf_rch"
        self._changing_packages = (
            ("DIS", "IC", "STO", "GHB", "RCH")
            if self.constant_d_from_modflow
            else ("STO", "GHB", "RCH")
        )

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = Modflow.get_init_parameters(self, name)
        parameters.loc[name + "_f"] = (-1.0, -2.0, 0.0, True, name, "uniform")
        return parameters

    def update_rch(self, rech: Series):
        rts = [(i, x) for i, x in zip(range(self._nper + 1), np.append(rech, 0.0))]

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

    def update_model(self, p: ArrayLike):
        self.remove_changing_packages()

        if self.constant_d_from_modflow:
            d, c, s, f = p[0:4]
            self.update_dis(d=d, height=1.0)
            self.update_ic(d=d)
        else:
            c, s, f = p[0:3]
            d = 0.0

        self.update_sto(s=s)
        self.update_ghb(d=d, c=c)

        rech = self._stress[0] + f * self._stress[1]
        self.update_rch(rech=rech)

        self._gwf.name_file.write()


class ModflowUzf(Modflow):
    def __init__(
        self,
        exe_name: str,
        sim_ws: str,
        nlay: int,
        simulate_et: bool = True,
        gwet_linear_or_square: None | Literal["linear", "square"] = "linear",
        unsat_et_wc_or_ae: Literal["wc", "ae"] = "wc",
        ntrailwaves: int = 15,
        nwavesets: int = 75,
        raise_on_modflow_error: bool = False,
    ):
        self._name = "mf_uzf"
        self._stress = None
        self._changing_packages = ("STO", "GHB", "UZF")
        Modflow.__init__(
            self,
            exe_name=exe_name,
            sim_ws=sim_ws,
            raise_on_modflow_error=raise_on_modflow_error,
        )
        self.nlay = nlay
        self.simulate_et = simulate_et
        self.gwet_linear_or_square = gwet_linear_or_square
        self.unsat_et_wc_or_ae = unsat_et_wc_or_ae
        self.ntrailwaves = ntrailwaves
        self.nwavesets = nwavesets

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = Modflow.get_init_parameters(self, name)
        parameters.loc[name + "_height"] = (1.0, 5.0, 2.0, True, name, "uniform")
        parameters.loc[name + "_vks"] = (0.0, 10.0, 1.0, True, name, "uniform")
        parameters.loc[name + "_thtr"] = (0.0, 0.2, 0.1, True, name, "uniform")
        parameters.loc[name + "_thts"] = (0.2, 0.4, 0.3, True, name, "uniform")
        parameters.loc[name + "_eps"] = (3.5, 10.0, 4.0, True, name, "uniform")
        parameters.loc[name + "_extdp"] = (0.0, 1.0, 0.5, True, name, "uniform")
        return parameters

    def update_model(self, p: ArrayLike):
        if self.head is None:
            d, c, s, height, vks, thtr, thts, eps, extdp, ha, hroot, rootact = p[0:10]
        else:
            c, s, height, vks, thtr, thts, eps, extdp, ha, hroot, rootact = p[0:9]
            d = 0.0

        self.remove_changing_packages()
        self.update_dis(d=d, height=height)
        self.update_ic(d=d)
        self.update_sto(s=s)
        self.update_ghb(d=d, c=c)

        finf = self._stress[0]
        pet = self._stress[1]  # make sure et is positive!
        # note: for specifying uzf number, use fortran indexing!

        thti = (thts - thtr) / 2  # initial water content
        extwc = thtr  # extiction water content
        ha = 0.0  # air entry potential (head)
        hroot = 0.0  # the root potential (head)
        rootact = 0.0  # the length of roots in a given volume of soil divided by that volume [L^-2]

        uzf_pkdat = [
            [
                n,  # iuzno
                (n, 0, 0),  # uzf_cellid
                1 if n == 0 else 0,  # landflag
                n + 1 if (n + 1) != self.nlay else -1,  # ivertcon
                1e-5,  # surface depression depth
                vks,  # vertical saturated hydraulic conductivity
                thtr,  # residual water content
                thts,  # saturated water content
                thti,  # initial water content
                eps,  # brooks-corey epsilon exponent
                f"uzf_cell_{n:02d}",  # boundname
            ]
            for n in range(self.nlay)
        ]

        uzf_spd = {}
        for iper, (finf_i, pet_i) in enumerate(zip(finf, pet)):
            data = [
                [
                    n,  # node
                    finf_i,  # infiltration rate
                    pet_i,  # evapotranspiration rate
                    extdp,  # extinction depth, always specified, but is only used if SIMULATE ET is specified
                    extwc,  # always specified, but is only used if SIMULATE ET and UNSAT ETWC are specified
                    ha,  # always specified, but is only used if SIMULATE ET and UNSAT ETAE are specified
                    hroot,  # always specified, but is only used if SIMULATE ET and UNSAT ETAE are specified
                    rootact,  # always specified, but is only used if SIMULATE ET and UNSAT ETAE are specified
                ]
                for n in range(self.nlay)
            ]
            uzf_spd[iper] = data
        # uzf_spd = dict([(i, []) for i, (ir, er) in enumerate(zip(p, e))])

        _ = flopy.mf6.ModflowGwfuzf(
            self._gwf,
            print_input=True,  # list of UZF information will be written to the listing file immediately after it is read.
            print_flows=True,  # the list of UZF flow rates will be printed to the listing file for every flow rates are printed for the last time step of each stress period
            save_flows=False,
            boundnames=True,  #  boundary names may be provided with the list of UZF cells
            simulate_et=True,  # If this option is selected, evapotranspiration will be simulated in the unsaturated zone but not in the saturated zone.
            # If this option is selected, evapotranspiration will be simulated in both the unsaturated and saturated zones. The groundwater evapotranspiration will be simulated using the original ET formulation of MODFLOW-2005.
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
            unsat_etwc=self.unsat_et_wc_or_ae
            == "wc",  # Evapotranspiration in the unsaturated zone will be simulated as a function of the specified potential evapotranspiration rate while the water content (THETA) is greater than the ET extinction water content (EXTWC).
            unsat_etae=self.unsat_et_wc_or_ae
            == "ae",  # Evapotranspiration in the unsaturated zone will be simulated simulated using a capillary pressure based formulation. Capillary pressure is calculated using the Brooks-Corey retention function.
            simulate_gwseep=True,
            ntrailwaves=self.ntrailwaves,
            nwavesets=self.nwavesets,
            nuzfcells=self.nlay,
            packagedata=uzf_pkdat,
            perioddata=uzf_spd,
            # budget_filerecord=f"{self._name}.uzf.bud",
            wc_filerecord=f"{self._name}.uzf.bin",
            # observations=uzf_obs,
            pname="uzf",
            filename=f"{self._name}.uzf",
        )
        self._gwf.name_file.write()
