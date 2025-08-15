import functools
import logging
from abc import abstractmethod
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
        solver_kwargs: dict | None = None,
        raise_on_modflow_error: bool = False,
        silent=True,
    ) -> None:
        self.exe_name = exe_name
        self.sim_ws = sim_ws
        self.raise_on_modflow_error = raise_on_modflow_error
        self.silent = silent
        self._simulation = None
        self._gwf = None
        self._name = "mf_base"
        self._stress = None
        self._nper = None
        self.solver_kwargs = (
            dict(
                complexity="SIMPLE",
                outer_dvclose=1e-2,
                inner_dvclose=1e-2,
                rcloserecord=1e-1,
                linear_acceleration="BICGSTAB",
            )
            if solver_kwargs is None
            else solver_kwargs
        )
        if head is not None:
            logger.info(
                "Make sure to delete the model parameter constant_d"
                "(`ml.del_constant()`). Base elevation is now controled by"
                "parameter `_d`."
            )
            self._head = head
            self.constant_d_from_modflow = True
        else:
            self._head = None
            self.constant_d_from_modflow = False

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
            newtonoptions=["NEWTON"],
        )

        _ = flopy.mf6.ModflowIms(
            self._simulation,
            **self.solver_kwargs,
        )

        _ = flopy.mf6.ModflowGwfnpf(
            self._gwf, save_flows=False, icelltype=0, pname="npf"
        )

        _ = flopy.mf6.ModflowGwfoc(
            self._gwf,
            head_filerecord=f"{self._gwf.name}.hds",
            saverecord=[("HEAD", "ALL")],
        )

        self._simulation.write_simulation(silent=self.silent)

        if not self.constant_d_from_modflow:
            self.update_dis(d=0.0, height=1.0)
            self.update_ic(d=0.0)

    @abstractmethod
    def update_model(self, p: ArrayLike) -> None:
        pass

    @functools.lru_cache(maxsize=5)
    def get_head(self, p):
        self.update_model(p=p)
        success, _ = self._simulation.run_simulation(silent=self.silent)
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

    def _remove_changing_package(self, package_name: str):
        if package_name in self._gwf.get_package_list():
            self._gwf.remove_package(package_name)

    def update_dis(self, d: float, height: float = 1.0):
        self._remove_changing_package("DIS")
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
        self._remove_changing_package("IC")
        ic = flopy.mf6.ModflowGwfic(self._gwf, strt=d, pname="ic")
        ic.write()

    def update_sto(self, s: float):
        self._remove_changing_package("STO")
        haq = (self._gwf.dis.top.array - self._gwf.dis.botm.array)[0]
        sto = flopy.mf6.ModflowGwfsto(
            self._gwf,
            save_flows=False,
            iconvert=0,
            ss=s / haq,
            sy=0.0,  # just to show the specific yield is not needed
            transient=True,
            pname="sto",
        )
        sto.write()

    def update_ghb(self, d: float, c: float):
        self._remove_changing_package("GHB")
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
        solver_kwargs: dict | None = None,
        raise_on_modflow_error: bool = False,
        **kwargs,
    ):
        Modflow.__init__(
            self,
            exe_name=exe_name,
            sim_ws=sim_ws,
            head=head,
            solver_kwargs=solver_kwargs,
            raise_on_modflow_error=raise_on_modflow_error,
            **kwargs,
        )
        self._name = "mf_rch"

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = Modflow.get_init_parameters(self, name)
        parameters.loc[name + "_f"] = (-1.0, -2.0, 0.0, True, name, "uniform")
        return parameters

    def update_rch(self, f: float):
        self._remove_changing_package("RCH")
        rech = self._stress[0] + f * self._stress[1]
        rts = [(i, x) for i, x in zip(range(self._nper + 1), np.append(rech, 0.0))]

        ts_dict = {
            "filename": f"{self._gwf.name}.rch_ts",
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
        if self.constant_d_from_modflow:
            d, c, s, f = p[0:4]
            self.update_dis(d=d, height=1.0)
            self.update_ic(d=d)
        else:
            c, s, f = p[0:3]
            d = 0.0

        self.update_sto(s=s)
        self.update_ghb(d=d, c=c)
        self.update_rch(f=f)
        self._gwf.name_file.write()


class ModflowUzf(Modflow):
    def __init__(
        self,
        exe_name: str,
        sim_ws: str,
        head: Series | None = None,
        simulate_et: bool = True,
        gwet_linear_or_square: None | Literal["linear", "square"] = "linear",
        ntrailwaves: int = 7,
        nwavesets: int = 40,
        solver_kwargs: dict | None = None,
        raise_on_modflow_error: bool = False,
    ):
        if solver_kwargs is None:
            solver_kwargs = dict(
                print_option="summary",
                outer_dvclose=3e-2,
                outer_maximum=300,
                under_relaxation="dbd",
                linear_acceleration="BICGSTAB",
                under_relaxation_theta=0.7,
                under_relaxation_kappa=0.08,
                under_relaxation_gamma=0.05,
                under_relaxation_momentum=0.0,
                inner_dvclose=3e-2,
                rcloserecord="1000.0 strict",
                inner_maximum=500,
                relaxation_factor=0.97,
                number_orthogonalizations=2,
                preconditioner_levels=8,
                preconditioner_drop_tolerance=0.001,
            )
        Modflow.__init__(
            self,
            exe_name=exe_name,
            sim_ws=sim_ws,
            head=head,
            solver_kwargs=solver_kwargs,
            raise_on_modflow_error=raise_on_modflow_error,
        )
        self._name = "mf_uzf"
        self.simulate_et = simulate_et
        self.gwet_linear_or_square = gwet_linear_or_square
        self.ntrailwaves = ntrailwaves
        self.nwavesets = nwavesets

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = Modflow.get_init_parameters(self, name)
        parameters.loc[name + "_height"] = (1.0, 0.01, 10.0, True, name, "uniform")
        parameters.loc[name + "_vks"] = (1.0, 0.0, 10.0, True, name, "uniform")
        parameters.loc[name + "_thtr"] = (0.1, 0.0, 0.2, True, name, "uniform")
        parameters.loc[name + "_thts"] = (0.3, 0.2, 0.4, True, name, "uniform")
        parameters.loc[name + "_thextfrac"] = (0.1, 0.0, 1.0, True, name, "uniform")
        parameters.loc[name + "_eps"] = (5.0, 3.5, 10.0, True, name, "uniform")
        parameters.loc[name + "_extdpfrac"] = (0.5, 0.0, 1.0, True, name, "uniform")
        return parameters

    def update_uzf(
        self,
        vks: float,
        thts: float,
        thtr: float,
        thext: float,
        eps: float,
        extdp: float,
    ):
        self._remove_changing_package("UZF")
        finf = self._stress[0]
        pet = self._stress[1]  # make sure et is positive!

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

    def update_drn(self):
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

    def update_model(self, p: ArrayLike):
        if self.constant_d_from_modflow:
            d, c, s, height, vks, thtr, thts, thextfrac, eps, extdpfrac = p[0:10]
            self.update_ic(d=d)
        else:
            c, s, height, vks, thtr, thts, thextfrac, eps, extdpfrac = p[0:9]
            d = 0.0

        self.update_dis(d=d, height=height)
        self.update_sto(s=s)
        self.update_ghb(d=d, c=c)
        self.update_drn()
        extdp = extdpfrac * height
        thext = thtr + (thts - thtr) * thextfrac
        self.update_uzf(
            vks=vks, thts=thts, thtr=thtr, thext=thext, eps=eps, extdp=extdp
        )
        self._gwf.name_file.write()


class ModflowDrn(ModflowRch):
    def __init__(self, **kwargs):
        ModflowRch.__init__(self, **kwargs)
        self._name = "mf_drn"

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = ModflowRch.get_init_parameters(self, name)
        parameters.loc[name + "_h_drn"] = (1.0, 0.0, 10.0, True, name, "uniform")
        parameters.loc[name + "_c_drn"] = (220, 1e1, 1e8, True, name, "uniform")
        return parameters

    def update_drn(self, d: float, c: float):
        drn = flopy.mf6.ModflowGwfdrn(
            self._gwf,
            print_input=True,
            print_flows=True,
            save_flows=False,
            boundnames=True,
            maxbound=1,
            stress_period_data={0: [[(0, 0, 0), d, 1 / c]]},
            pname="drn",
        )
        drn.write()

    def update_model(self, p: ArrayLike):
        if self.constant_d_from_modflow:
            d = p[0]
            p = p[1:]
            self.update_ic(d=d)
        else:
            d = 0.0
        c, s, f, h_drn, c_drn = p
        self.update_dis(d=0, height=1.0)
        self.update_sto(s=s)
        self.update_ghb(d=d, c=c)
        self.update_rch(f=f)
        self.update_drn(d=d + h_drn, c=c_drn)
        self._gwf.name_file.write()


class ModflowSto(ModflowRch):
    def __init__(self, **kwargs):
        ModflowRch.__init__(self, **kwargs)
        self._name = "mf_sto"

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = ModflowRch.get_init_parameters(self, name)
        parameters.loc[name + "_h_drn"] = (1.0, 0.0, 10.0, True, name, "uniform")
        parameters.loc[name + "_s_drn"] = (0.3, 0.001, 1.0, True, name, "uniform")
        return parameters

    def update_sto(self, s: float, s_drn: float):
        self._remove_changing_package("STO")
        haq = (self._gwf.dis.top.array - self._gwf.dis.botm.array)[0]
        sto = flopy.mf6.ModflowGwfsto(
            self._gwf,
            save_flows=False,
            iconvert=1,
            ss=s_drn / haq,
            sy=s,
            transient=True,
            pname="sto",
            ss_confined_only=True,
        )
        sto.write()

    def update_model(self, p: ArrayLike):
        if self.constant_d_from_modflow:
            d = p[0]
            p = p[1:]
            self.update_ic(d=d)
        else:
            d = 0.0
        c, s, f, h_drn, s_drn = p
        self.update_dis(d=0, height=d + h_drn)
        self.update_sto(s=s, s_drn=s_drn)
        self.update_ghb(d=d, c=c)
        self.update_rch(f=f)
        self._gwf.name_file.write()


class ModflowDrnSto(ModflowDrn, ModflowSto):
    def __init__(self, **kwargs):
        ModflowDrn.__init__(self, **kwargs)
        self._name = "mf_drn_sto"

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
        c, s, f, h_drn, c_drn, s_drn = p
        self.update_dis(d=0, height=d + h_drn)
        self.update_sto(s=s, s_drn=s_drn)
        self.update_ghb(d=d, c=c)
        self.update_rch(f=f)
        self.update_drn(d=d + h_drn, c=c_drn)
        self._gwf.name_file.write()
