import logging
from pathlib import Path
from platform import node as get_computername
from shutil import copy as copy_file
from threading import Thread
from time import sleep
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyemu
from pandas import DataFrame
from pastas.solver import BaseSolver
from psutil import cpu_count

logger = logging.getLogger(__name__)

np.random.seed(pyemu.en.SEED)  # set seed


class PestSolver(BaseSolver):
    """PEST solver base class"""

    def __init__(
        self,
        exe_name: Union[str, Path] = "pestpp",
        model_ws: Union[str, Path] = Path("model"),
        temp_ws: Union[str, Path] = Path("temp"),
        noptmax: int = 0,
        control_data: Optional[dict] = None,
        pcov: Optional[DataFrame] = None,
        nfev: Optional[int] = None,
        long_names: bool = True,
        **kwargs,
    ) -> None:
        BaseSolver.__init__(self, pcov=pcov, nfev=nfev, **kwargs)
        # model workspace (for pastas files)
        self.model_ws = Path(model_ws).resolve()
        if not self.model_ws.exists():
            self.model_ws.mkdir(parents=True)
        # template workspace (for pest files)
        self.temp_ws = Path(temp_ws).resolve()
        self.exe_name = Path(exe_name)  # pest executable
        self.pf = pyemu.utils.PstFrom(
            original_d=self.model_ws,
            new_d=self.temp_ws,
            remove_existing=True,
            longnames=long_names,
        )
        copy_file(self.exe_name, self.temp_ws)  # copy pest executable
        self.noptmax = noptmax
        self.control_data = control_data
        self.run_function = """def run():
    # load packages
    from pathlib import Path

    from pandas import read_csv
    from pastas.io.base import load as load_model

    # base path
    fpath = Path(__file__).parent

    # load pastas model
    ml = load_model(fpath / "model.pas")

    # update model parameters
    parameters = read_csv(fpath / "parameters_sel.csv", index_col=0)
    for pname, val in parameters.loc[:, "optimal"].items():
        pname = pname.replace("_g", "_A") if pname.endswith("_g") else pname
        ml.set_parameter(pname, optimal=val)

    # simulate
    simulation = ml.simulate()
    simulation.loc[ml.observations().index].to_csv(fpath / "simulation.csv")"""

    def write_run_function(self):
        """Write the run function to a file"""
        with (self.model_ws / "_run_pastas_model.py").open("w") as f:
            f.write(self.run_function)

    def setup_model(self):
        """Setup and export Pastas model for optimization"""
        # setup parameters
        self.vary = self.ml.parameters.vary.values.astype(bool)
        parameters = self.ml.parameters[self.vary].copy()
        parameters.index = [p.replace("_A", "_g") if p.endswith("_A") else p for p in parameters.index]
        parameters.index.name = "parnames"
        parameters.loc[:, "optimal"] = parameters.loc[:, "initial"]
        if "constant_d" in parameters.index:
            self.ml.set_parameter(
                "constant_d",
                pmin=parameters.at["constant_d", "initial"] - 10.0,
                pmax=parameters.at["constant_d", "initial"] + 10.0,
            )
        par_sel = parameters.loc[:, ["optimal"]]
        par_sel.to_csv(self.model_ws / "parameters_sel.csv")
        copy_file(self.model_ws / "parameters_sel.csv", self.temp_ws)
        self.par_sel = par_sel

        # observations
        observations = self.ml.observations()
        observations.name = "Observations"
        observations.to_csv(self.model_ws / "simulation.csv")
        copy_file(self.model_ws / "simulation.csv", self.temp_ws)
        self.observations = observations

        # model
        self.ml.to_file(self.model_ws / "model.pas")
        copy_file(self.model_ws / "model.pas", self.temp_ws)

        # write run function
        self.write_run_function()

    def load_pst(self) -> pyemu.Pst:
        """Load PEST control file"""
        return pyemu.Pst(str(self.temp_ws / "pest.pst"))

    def write_pst(self, pst: pyemu.Pst, version: int = 2) -> None:
        pst.write(self.pf.new_d / "pest.pst", version=version)

    def setup_files(self, version: int = 2, obs_std: float = 0.00):
        """Setup PEST structure for optimization"""
        # parameters
        self.pf.add_parameters(
            self.model_ws / "parameters_sel.csv",
            index_cols=[self.par_sel.index.name],
            use_cols=self.par_sel.columns.to_list(),
            par_type="grid",
            par_style="direct",
            transform="none",
            # pargp=self.par_sel.columns.to_list(),
            # par_name_base=self.par_sel.columns.to_list(), #[x.split("_")[0] for x in self.par_sel.columns],
            # lower_bound=self.ml.parameters.loc[self.vary, "pmin"].values.tolist(),
            # upper_bound=self.ml.parameters.loc[self.vary, "pmax"].values.tolist(),
            # ult_lbound = self.ml.parameters.loc[self.vary, ["pmin"]].transpose().values.tolist(),
            # ult_ubound = self.ml.parameters.loc[self.vary, ["pmax"]].transpose().values.tolist(),
        )

        # observations and simulation
        self.pf.add_observations(
            "simulation.csv",
            index_cols=[self.observations.index.name],
            use_cols=[self.observations.name],
        )

        # python scripts to run
        self.pf.add_py_function(
            self.model_ws / "_run_pastas_model.py", "run()", is_pre_cmd=None
        )
        self.pf.mod_py_cmds.append("run()")

        # create control file
        pst = self.pf.build_pst(self.pf.new_d / "pest.pst", version=version)
        # parameter bounds
        pst.parameter_data.loc[:, ["parlbnd"]] = self.ml.parameters.loc[self.vary, "pmin"].values
        pst.parameter_data.loc[:, ["parubnd"]] = self.ml.parameters.loc[self.vary, "pmax"].values
        pst.parameter_data.loc[:, ["parchglim"]] = "relative"
        pst.parameter_data.loc[:, ["pargp"]] = self.par_sel.columns.to_list()
        if obs_std > 0.0:
            pst.observation_data.loc[:, "standard_deviation"] = obs_std
        pst.control_data.noptmax = self.noptmax  # optimization runs
        if self.control_data is not None:
            for key, value in self.control_data.items():
                if key == "control_data":
                    logger.warning(
                        "noptmax is set as an attribute and can't be set using the `control_data` dictionary"
                    )
                else:
                    setattr(pst.control_data, key, value)
        self.write_pst(pst=pst, version=version)

    def run(self, arg_str: str = ""):
        pyemu.os_utils.run(
            f"{self.exe_name.name} pest.pst{arg_str}", cwd=self.pf.new_d, verbose=True
        )


class PestGlmSolver(PestSolver):
    """PESTPP-GLM (Gauss-Levenberg-Marquardt) solver"""

    def __init__(
        self,
        exe_name: Union[str, Path] = "pestpp-glm",
        model_ws: Union[str, Path] = Path("model"),
        temp_ws: Union[str, Path] = Path("temp"),
        noptmax: int = 0,
        control_data: Optional[dict] = None,
        pcov: Optional[DataFrame] = None,
        nfev: Optional[int] = None,
        **kwargs,
    ) -> None:
        PestSolver.__init__(
            self,
            exe_name=exe_name,
            model_ws=model_ws,
            temp_ws=temp_ws,
            noptmax=noptmax,
            control_data=control_data,
            pcov=pcov,
            nfev=nfev,
            long_names=True,
            **kwargs,
        )

    def solve(self, **kwargs) -> Tuple[bool, np.ndarray, np.ndarray]:
        self.setup_model()
        self.setup_files()
        self.run()

        # optimal paramters
        ipar = pd.read_csv(self.temp_ws / "pest.ipar", index_col=0).transpose()
        ipar.index = self.ml.parameters.index[self.vary]
        optimal = self.ml.parameters["initial"].copy().values
        self.nfev = ipar.columns[-1]
        optimal[self.vary] = ipar.loc[:, self.nfev].values

        # covariance
        pcov = pd.read_csv(
            self.temp_ws / f"pest.{self.nfev}.post.cov",
            sep="\s+",
            skiprows=[0],
            nrows=len(ipar.index),
            header=None,
        )
        pcov.index = ipar.index
        pcov.columns = ipar.index
        self.pcov = pcov
        stderr = np.full(len(optimal), np.nan)
        stderr[self.vary] = np.sqrt(np.diag(self.pcov.values))

        # objective function value (phi)
        iobj = pd.read_csv(self.temp_ws / "pest.iobj", index_col=0)
        self.obj_func = iobj.at[self.nfev, "total_phi"]
        success = True  # always :)
        return success, optimal, stderr


class PestHpSolver(PestSolver):
    """PEST_HP (highly parallelized) solver"""

    def __init__(
        self,
        exe_name: Union[str, Path] = "pest_hp",
        exe_agent: Union[str, Path] = "agent_hp",
        model_ws: Union[str, Path] = Path("model"),
        temp_ws: Union[str, Path] = Path("temp"),
        noptmax: int = 0,
        control_data: Optional[dict] = None,
        pcov: Optional[DataFrame] = None,
        nfev: Optional[int] = None,
        port_number: int = 4004,
        **kwargs,
    ) -> None:
        PestSolver.__init__(
            self,
            exe_name=exe_name,
            model_ws=model_ws,
            temp_ws=temp_ws,
            pcov=pcov,
            nfev=nfev,
            long_names=False,
            noptmax=noptmax,
            control_data=control_data,
            **kwargs,
        )
        self.exe_agent = Path(exe_agent)
        self.port_number = port_number
        self.computername = get_computername()
        copy_file(self.exe_agent, self.temp_ws)  # copy agent executable

    def solve(self, **kwargs) -> Tuple[bool, np.ndarray, np.ndarray]:
        self.setup_model()
        self.setup_files(version=1)
        # start consecutive thread for pest_hp and agent_hp excutable
        threads = [
            Thread(target=self.run, args=(f" /h :{self.port_number}",)),
            Thread(target=self.run_agent),
        ]
        for t in threads:
            t.start()
            sleep(1.0)
        for t in threads:
            t.join()

        par = pd.read_csv(
            self.temp_ws / "pest.par", index_col=0, sep="\s+", skiprows=[0], header=None
        )
        par.index = self.ml.parameters.index[self.vary]
        optimal = self.ml.parameters["initial"].copy().values
        optimal[self.vary] = par.iloc[:, 0].values

        ofr = pd.read_csv(self.temp_ws / "pest.ofr", index_col=0, sep="\s+", skiprows=2)
        self.nfev = ofr.index[-1]
        self.obj_func = ofr.at[self.nfev, "total"]

        return True, optimal, np.full_like(optimal, np.nan)

    def run_agent(self):
        pyemu.os_utils.run(
            f"{self.exe_agent.name} pest.pst /h {self.computername}:{self.port_number}",
            cwd=self.pf.new_d,
            verbose=True,
        )


class PestIesSolver(PestSolver):
    """PESTPP-IES (Iterative Ensemble Smoother) solver"""

    def __init__(
        self,
        exe_name: Union[str, Path] = "pestpp-ies",
        model_ws: Union[str, Path] = Path("model"),
        temp_ws: Union[str, Path] = Path("temp"),
        master_ws: Union[str, Path] = Path("master"),
        noptmax: int = 0,
        control_data: Optional[dict] = None,
        pcov: Optional[DataFrame] = None,
        nfev: Optional[int] = None,
        port_number: int = 4004,
        num_workers: Optional[int] = None,
        **kwargs,
    ) -> None:
        PestSolver.__init__(
            self,
            exe_name=exe_name,
            model_ws=model_ws,
            temp_ws=temp_ws,
            pcov=pcov,
            nfev=nfev,
            **kwargs,
        )
        self.master_ws = master_ws
        self.noptmax = noptmax
        self.control_data = control_data
        self.port_number = port_number
        self.num_workers = (
            cpu_count(logical=False) if num_workers is None else num_workers
        )

    def run_ensembles(self, ies_num_reals: int = 50, obs_std: float=0.0) -> None:
        self.setup_model()
        self.setup_files(obs_std=obs_std)

        # change ies_num_reals
        pst = self.load_pst()
        pst.pestpp_options["ies_num_reals"] = ies_num_reals
        if obs_std == 0.0:
            pst.pestpp_options["ies_no_noise"] = True
        self.write_pst(pst=pst, version=2)

        pyemu.os_utils.start_workers(
            worker_dir=self.temp_ws,  # the folder which contains the "template" PEST dataset
            exe_rel_path=self.exe_name.name,  # the PEST software version we want to run
            pst_rel_path="pest.pst",  # the control file to use with PEST
            num_workers=self.num_workers,  # how many agents to deploy
            worker_root=self.master_ws.parent,  # where to deploy the agent directories; relative to where python is running
            port=self.port_number,  # the port to use for communication
            master_dir=self.master_ws,  # the manager directory
        )

    def solve(self) -> None:
        raise NotImplementedError("Currently not implemented. Need to check how to implement this properly with optimal parameters and stderr.")
        # self.setup_model()
        # self.setup_files()
        # self.run()

        # return success, optimal, stderr
