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
        self, exe_name: str, sim_ws: str, raise_on_modflow_error: bool = False
    ) -> None:
        self.exe_name = exe_name
        self.sim_ws = sim_ws
        self.raise_on_modflow_error = raise_on_modflow_error
        self._name = "mf_base"
        self._changing_packages = ("STO", "GHB")

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "name", "dist"]
        )
        parameters.loc[name + "_c"] = (220, 1e1, 1e8, True, name, "uniform")
        parameters.loc[name + "_s"] = (0.05, 0.001, 0.5, True, name, "uniform")
        return parameters

    def base_model(self) -> None:
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
            self._nper = len(self._stress[0])
            self.base_model()
        return self.get_head(tuple(p))

    def remove_changing_packages(self):
        for cp in self._changing_packages:
            if cp in self._gwf.get_package_list():
                self._gwf.remove_package(cp)

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
        self, exe_name: str, sim_ws: str, raise_on_modflow_error: bool = False
    ):
        self._name = "mf_rch"
        self._stress = None
        self._simulation = None
        self._gwf = None
        self._changing_packages = ("STO", "GHB", "RCH")
        Modflow.__init__(self, exe_name, sim_ws, raise_on_modflow_error)

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = Modflow.get_init_parameters(self, name)
        parameters.loc[name + "_f"] = (-1.0, -2.0, 0.0, True, name, "uniform")
        return parameters

    def update_model(self, p: ArrayLike):
        c, s, f = p[0:3]

        d = 0.0

        rech = self._stress[0] + f * self._stress[1]

        self.remove_changing_packages()
        self.update_sto(s=s)
        self.update_ghb(d=d, c=c)

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

        self._gwf.name_file.write()


class ModflowUzf(Modflow):
    def __init__(
        self,
        exe_name: str,
        sim_ws: str,
        nlay: int,
        simulate_et: bool = True,
        gwet_linear_or_square: Literal["linear", "square"] = "linear",
        unsat_et_wc_or_ae: Literal["wc", "ae"] = "ae",
        raise_on_modflow_error: bool = False,
    ):
        self._name = "mf_uzf"
        self._stress = None
        self._simulation = None
        self._gwf = None
        self._changing_packages = ("STO", "GHB", "UZF")
        Modflow.__init__(self, exe_name, sim_ws, raise_on_modflow_error)
        self.nlay = nlay
        self.simulate_et = simulate_et
        self.gwet_linear_or_square = gwet_linear_or_square
        self.unsat_et_wc_or_ae = unsat_et_wc_or_ae

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = Modflow.get_init_parameters(self, name)
        parameters.loc[name + "_k"] = (0.0, 10.0, 1.0, True, name, "uniform")
        parameters.loc[name + "_thtr"] = (0.0, 0.2, 0.1, True, name, "uniform")
        parameters.loc[name + "_thts"] = (0.2, 0.4, 0.3, True, name, "uniform")
        parameters.loc[name + "_eps"] = (3.5, 10.0, 4.0, True, name, "uniform")
        parameters.loc[name + "_extdp"] = (0.0, 1.0, 0.5, True, name, "uniform")
        parameters.loc[name + "_ha"] = (0.0, 0.1, 0.0, False, name, "uniform")
        parameters.loc[name + "_hroot"] = (0.0, 0.1, 0.0, False, name, "uniform")
        parameters.loc[name + "_rootact"] = (0.0, 0.1, 0.0, False, name, "uniform")
        return parameters

    def update_model(self, p: ArrayLike):
        c, s, k, thtr, thts, eps, extdp, ha, hroot, rootact = p[0:9]

        d = 0.0

        self.remove_changing_packages()
        self.update_sto(s=s)
        self.update_ghb(d=d, c=c)

        p = self._stress[0]
        e = self._stress[1]  # make sure et is positive
        # note: for specifying uzf number, use fortran indexing!

        surfdep = 1e-5
        thti = (thts - thtr) / 2  # initial water content
        extwc = thtr  # extiction water content

        # If IVERTCON is greater than zero then residual potential evapotranspiration not satisfied in the UZF cell is applied to the underlying UZF and GWF cells.
        ivertcon = [i + 1 for i in range(self.nlay)]
        ivertcon[-1] = -1

        uzf_pkdat = [
            [
                i,  # iuzno
                (i, 0, 0),  # cellid
                1 if i == 0 else 0,  # landflag
                ivertcon[i],  # ivertcon
                surfdep,  # surfdep
                k,  # vks
                thtr,  # thtr
                thts,  # thts
                thti,  # thti
                eps,  # eps
                f"uzf_{i:02d}",  # boundname
            ]
            for i in range(self.nlay)
        ]

        uzf_spd = {}
        for iper, (ir, er) in enumerate(zip(p, e)):
            data = [
                [i, ir, er, extdp, extwc, ha, hroot, rootact] for i in range(self.nlay)
            ]
            uzf_spd[iper] = data
        # uzf_spd = dict([(i, []) for i, (ir, er) in enumerate(zip(p, e))])

        linear_gwet = self.gwet_linear_or_square == "linear"
        unsat_etae = self.unsat_et_wc_or_ae == "ae"
        _ = flopy.mf6.ModflowGwfuzf(
            self._gwf,
            print_input=True,
            print_flows=True,
            save_flows=True,
            boundnames=True,
            simulate_et=True,  # If this option is selected, evapotranspiration will be simulated in the unsaturated zone but not in the saturated zone.
            linear_gwet=linear_gwet,  # If this option is selected, evapotranspiration will be simulated in both the unsaturated and saturated zones. The groundwater evapotranspiration will be simulated using the original ET formulation of MODFLOW-2005.
            square_gwet=not linear_gwet,  # If this option is selected, evapotranspiration will be simulated in both the unsaturated and saturated zones. The groundwater evapotranspiration will be simulated by assuming a constant evapotranspiration rate for groundwater levels between land surface (TOP) and land surface minus the evapotranspiration extinction depth (TOP-EXTDP). Groundwater evapotranspiration is smoothly reduced from the potential evapotranspiration rate to zero over a nominal interval at TOP-EXTDP.
            unsat_etwc=not unsat_etae,  # Evapotranspiration in the unsaturated zone will be simulated as a function of the specified potential evapotranspiration rate while the water content (THETA) is greater than the ET extinction water content (EXTWC).
            unsat_etae=unsat_etae,  # Evapotranspiration in the unsaturated zone will be simulated simulated using a capillary pressure based formulation. Capillary pressure is calculated using the Brooks-Corey retention function.
            simulate_gwseep=True,
            ntrailwaves=15,
            nwavesets=75,
            nuzfcells=len(uzf_pkdat),
            packagedata=uzf_pkdat,
            perioddata=uzf_spd,
            budget_filerecord=f"{self._name}.uzf.bud",
            wc_filerecord=f"{self._name}.uzf.bin",
            # observations=uzf_obs,
            pname="uzf",
            filename=f"{self._name}.uzf",
        )
        self._gwf.name_file.write()
