import json
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache, partial
from logging import getLogger
from os import cpu_count
from pathlib import Path
from platform import node as get_computername
from shutil import copy as copy_file
from threading import Thread
from time import sleep
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyemu
from numpy.typing import NDArray
from pandas import DataFrame
from pastas.solver import BaseSolver
from pastas.typing import ArrayLike, Model, TimestampType
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative
from scipy.stats import norm, truncnorm
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

logger = getLogger(__name__)


def run() -> None:
    """Run function for PEST (from files)"""
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
    simulation.loc[ml.observations().index].to_csv(fpath / "simulation.csv")


def run_pypestworker(
    pst: str | pyemu.Pst,
    host: int,
    port: int,
    ml: Model,
    timeout: float = 0.1,
) -> None:
    """Run function for PEST using the PyPestWorker (in memory)"""
    ppw = pyemu.os_utils.PyPestWorker(
        pst=pst,
        host=host,
        port=port,
        timeout=timeout,
        verbose=False,
    )
    pvals = ppw.get_parameters()
    if pvals is None:
        return None

    while True:
        for pname, val in pvals.items():
            pname = pname.split(":")[-1] if ":" in pname else pname
            pname = pname.replace("_g", "_A") if pname.endswith("_g") else pname
            ml.set_parameter(pname, optimal=val)
        sim = ml.simulate()
        obsvals = sim.loc[ml.observations().index]
        obsvals.index = ppw._pst.observation_data.index
        ppw.send_observations(obsvals=obsvals)
        pvals = ppw.get_parameters()
        if pvals is None:
            break


class PestSolver(BaseSolver):
    """PEST solver base class"""

    def __init__(
        self,
        exe_name: str | Path,
        model_ws: str | Path = Path("model"),
        temp_ws: str | Path = Path("temp"),
        noptmax: int = 0,
        control_data: dict[str, Any] | None = None,
        pcov: DataFrame | None = None,
        nfev: int | None = None,
        long_names: bool = True,
        port_number: int = 4004,
        use_pypestworker: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the PEST solver.

        Parameters
        ----------
        exe_name : str | Path
            The name or path to the PEST executable.
        model_ws : str | Path, optional
            The model workspace directory for Pastas files. Default is "model".
        temp_ws : str | Path, optional
            The template workspace directory for PEST files. Default is "temp".
        noptmax : int, optional
            The maximum number of optimization iterations. Default is 0.
        control_data : dict[str, Any] | None, optional
            Control data for the PEST solver. Default is None.
        pcov : DataFrame | None, optional
            The parameter covariance matrix. Default is None.
        nfev : int | None, optional
            The number of function evaluations. Default is None.
        long_names : bool, optional
            Whether to use long names in the PEST control file. Default is True.
        port_number : int, optional
            The port number for communication. Default is 4004.
        use_pypestworker : bool, optional
            Whether to use the PyPestWorker for Python processing. Default is True.
        **kwargs : dict
            Additional keyword arguments passed to the BaseSolver.

        Returns
        -------
        None
        """
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
        self.noptmax: int = noptmax
        self.control_data: dict[str, Any] = control_data
        self.port_number = port_number
        self.use_pypestworker: bool = use_pypestworker
        self.run_function: Callable = run
        self.ppw_function: Callable = run_pypestworker

    def setup_model(self):
        """Setup and export Pastas model for PEST optimization"""
        # observations
        observations = self.ml.observations()
        observations.name = "Observations"
        observations.index.name = "Datetime"
        observations.to_csv(self.model_ws / "simulation.csv")
        copy_file(self.model_ws / "simulation.csv", self.temp_ws)
        self.observations = observations

        # setup parameters
        # initial_parameters = self.ml.parameters["initial"].copy()
        # for pname, val in initial_parameters.items():
        #     self.ml.set_parameter(pname, optimal=val)
        self.ml.parameters.loc[:, "optimal"] = self.ml.parameters.loc[:, "initial"]
        self.vary = self.ml.parameters.vary.values.astype(bool)
        parameters = self.ml.parameters[self.vary].copy()
        parameters.index = [
            p.replace("_A", "_g") if p.endswith("_A") else p for p in parameters.index
        ]
        parameters.index.name = "parnames"
        if "constant_d" in parameters.index:
            if np.isnan(parameters.at["constant_d", "pmin"]):
                self.ml.set_parameter(
                    "constant_d",
                    pmin=np.min(observations.values) - np.std(observations.values),
                )
            if np.isnan(parameters.at["constant_d", "pmax"]):
                self.ml.set_parameter(
                    "constant_d",
                    pmax=np.max(observations.values) + np.std(observations.values),
                )
        if self.ml.parameters.index.str.rsplit("_").str[0].str.isupper().any():
            logger.error(
                "pestpp is case insensitive so any capitalized parameters (stress model names) can cause issues in the solver."
            )
        par_sel = parameters.loc[:, ["optimal"]]
        par_sel.to_csv(self.model_ws / "parameters_sel.csv")
        copy_file(self.model_ws / "parameters_sel.csv", self.temp_ws)
        self.par_sel = par_sel

        # model
        self.ml.to_file(self.model_ws / "model.pas")
        copy_file(self.model_ws / "model.pas", self.temp_ws)

    def write_pst(self, pst: pyemu.Pst, version: int = 2) -> None:
        """Write pest control file

        Parameters
        ----------
        pst : pyemu.Pst
            Pyemu pest control file object.
        version : int, optional
            Version of the control file, by default version 2
        """
        pst.write(self.pf.new_d / "pest.pst", version=version)

    def setup_files(self, version: int = 2):
        """Setup PEST file structure for optimization

        Parameters
        ----------
        version : int, optional
            Version of the control file, by default version 2
        """

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
        self.pf.add_py_function(self.run_function, "run()", is_pre_cmd=None)
        self.pf.mod_py_cmds.append("run()")

        # create control file
        pst = self.pf.build_pst(self.pf.new_d / "pest.pst", version=version)
        # parameter bounds
        pst.parameter_data.loc[:, ["parlbnd"]] = self.ml.parameters.loc[
            self.vary, "pmin"
        ].values
        pst.parameter_data.loc[:, ["parubnd"]] = self.ml.parameters.loc[
            self.vary, "pmax"
        ].values
        pst.parameter_data.loc[:, ["parchglim"]] = "relative"
        pst.parameter_data.loc[:, ["pargp"]] = self.par_sel.columns.to_list()
        pst.control_data.noptmax = self.noptmax  # optimization runs
        if self.control_data is not None:
            for key, value in self.control_data.items():
                if key == "noptmax":
                    logger.warning(
                        "noptmax is set as an attribute and can't be set using the `control_data` dictionary"
                    )
                else:
                    setattr(pst.control_data, key, value)
        self.write_pst(pst=pst, version=version)

        # save parameter and observation index for going back and forth between pastas and pest names
        self.parameter_index = dict(
            zip(pst.parameter_data.index, self.ml.parameters[self.vary].index)
        )
        with (self.temp_ws / "parameter_index.json").open("w") as f:
            json.dump(obj=self.parameter_index, fp=f, default=str)
        self.observation_index = dict(
            zip(pst.observation_data.index, self.ml.observations().index)
        )
        with (self.temp_ws / "observation_index.json").open("w") as f:
            json.dump(obj=self.observation_index, fp=f, default=str)

    def run(self, arg_str: str = "", silent: bool = False):
        pyemu.os_utils.run(
            f"{self.exe_name.name} pest.pst{arg_str}", cwd=self.pf.new_d, verbose=silent
        )

    def initialize(self, version: int = 2) -> None:
        """Initialize the solver by setting up the model and files."""
        if self.ml is None:
            raise ValueError("No Pastas model assigned to the solver.")
        if self.pf.pst is None:
            self.setup_model()
            self.setup_files(version=version)
        else:
            logger.info("Solver is already initialized.")

    @staticmethod
    def download_executable(path: Path | str, subset: list[str] | None) -> None:
        """Download the PEST++ executable if it does not exist.
        Parameters
        ----------
        path : Path
            The directory where the executable should be located.
        subset : str | list[str] | None
            A list of strings to filter the executable download,
            e.g.: ["pestpp-glm", "pestpp-ies"]. If None, no filtering
            is applied and all available executables are downloaded.
        Returns
        -------
        None
        """
        pyemu.utils.get_pestpp(str(path), subset=subset, force=True)


class PestGlmSolver(PestSolver):
    """PESTPP-GLM (Gauss-Levenberg-Marquardt) solver"""

    def __init__(
        self,
        exe_name: str | Path = "pestpp-glm",
        model_ws: str | Path = Path("model"),
        temp_ws: str | Path = Path("temp"),
        noptmax: int = 0,
        control_data: dict[str, Any] | None = None,
        pcov: DataFrame | None = None,
        nfev: int | None = None,
        port_number: int = 4004,
        use_pypestworker: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize the PESTPP-GLM solver.

        Parameters
        ----------
        exe_name : str | Path, optional
            The name or path to the PESTPP-GLM executable. Default is "pestpp-glm".
        model_ws : str | Path, optional
            The model workspace directory for Pastas files. Default is "model".
        temp_ws : str | Path, optional
            The template workspace directory for PEST files. Default is "temp".
        noptmax : int, optional
            The maximum number of optimization iterations. Default is 0.
        control_data : dict[str, Any] | None, optional
            Control data for the PEST solver. Default is None.
        pcov : DataFrame | None, optional
            The parameter covariance matrix. Default is None.
        nfev : int | None, optional
            The number of function evaluations. Default is None.
        port_number : int, optional
            The port number for communication. Default is 4004.
        use_pypestworker : bool, optional
            Whether to use the PyPestWorker for Python processing. Default is True.
        **kwargs : dict
            Additional keyword arguments passed to the PestSolver.

        Returns
        -------
        None
        """
        PestSolver.__init__(
            self,
            exe_name=exe_name,
            model_ws=model_ws,
            temp_ws=temp_ws,
            noptmax=noptmax,
            control_data=control_data,
            pcov=pcov,
            nfev=nfev,
            port_number=port_number,
            use_pypestworker=use_pypestworker,
            long_names=True,
            **kwargs,
        )

    def solve(self, **kwargs) -> tuple[bool, NDArray[np.float64], NDArray[np.float64]]:
        """
        Solves the optimization problem using the pestpp-glm solver.
        This method sets up the model and necessary files, runs the solver, and
        retrieves the optimal parameters, their covariance, and the objective
        function value.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for the solver.

        Returns
        -------
        success : bool
            Indicates whether the solver ran successfully.
        optimal : NDArray[np.float64]
            The optimal parameters obtained from the solver.
        stderr : NDArray[np.float64]
            The standard errors of the optimal parameters.
        """

        self.initialize(version=2)

        if self.use_pypestworker:
            pyemu.os_utils.start_workers(
                worker_dir=self.temp_ws,  # the folder which contains the "template" PEST dataset
                exe_rel_path=self.exe_name.name,  # the PEST software version we want to run
                pst_rel_path="pest.pst",  # the control file to use with PEST
                num_workers=1,  # how many agents to deploy
                port=self.port_number,  # the port to use for communication
                worker_root=self.temp_ws.parent,  # where to deploy the agent directories; relative to where python is running
                master_dir=self.temp_ws,  # the manager directory
                reuse_master=self.use_pypestworker,
                ppw_function=self.ppw_function,
                ppw_kwargs={"ml": self.ml},
            )
        else:
            self.run()

        # optimal parameters
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
        exe_name: str | Path = "pest_hp",
        exe_agent: str | Path = "agent_hp",
        model_ws: str | Path = Path("model"),
        temp_ws: str | Path = Path("temp"),
        noptmax: int = 0,
        control_data: dict[str, Any] | None = None,
        pcov: DataFrame | None = None,
        nfev: int | None = None,
        port_number: int = 4004,
        **kwargs,
    ) -> None:
        """
        Initialize the PEST_HP solver.

        Parameters
        ----------
        exe_name : str | Path, optional
            The name or path to the PEST_HP executable. Default is "pest_hp".
        exe_agent : str | Path, optional
            The name or path to the agent_HP executable. Default is "agent_hp".
        model_ws : str | Path, optional
            The model workspace directory for Pastas files. Default is "model".
        temp_ws : str | Path, optional
            The template workspace directory for PEST files. Default is "temp".
        noptmax : int, optional
            The maximum number of optimization iterations. Default is 0.
        control_data : dict[str, Any] | None, optional
            Control data for the PEST solver. Default is None.
        pcov : DataFrame | None, optional
            The parameter covariance matrix. Default is None.
        nfev : int | None, optional
            The number of function evaluations. Default is None.
        port_number : int, optional
            The port number for communication. Default is 4004.
        **kwargs : dict
            Additional keyword arguments passed to the PestSolver.

        Returns
        -------
        None
        """
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
            port_number=port_number,
            use_pypestworker=False,
            **kwargs,
        )
        self.exe_agent = Path(exe_agent)
        self.computername = get_computername()
        copy_file(self.exe_agent, self.temp_ws)  # copy agent executable

    def solve(
        self, silent: bool = False, **kwargs
    ) -> tuple[bool, NDArray[np.float64], NDArray[np.float64]]:
        """
        Solve the optimization problem using the pest_hp solver.

        This method sets up the model and necessary files, runs the solver, and
        retrieves the optimal parameters and the objective function value.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for the solver.

        Returns
        -------
        success : bool
            Indicates whether the solver ran successfully.
        optimal : NDArray[np.float64]
            The optimal parameters obtained from the solver.
        stderr : NDArray[np.float64]
            The standard errors of the optimal parameters.
        """
        self.initialize(version=1)
        # start consecutive thread for pest_hp and agent_hp excutable
        threads = [
            Thread(target=self.run, args=(f" /h :{self.port_number}", silent)),
            Thread(target=self.run_agent, args=(silent,)),
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

        # TODO: Obtain stderror from pest.hp output and covariance matrix
        stderr = np.full_like(optimal, np.nan)
        return True, optimal, stderr

    def run_agent(self, silent: bool = False) -> None:
        """
        Executes the agent using the specified executable and configuration.
        This method runs the agent with the given executable name, pest control file,
        and host configuration (computer name and port number). The execution is done
        in the directory specified by `self.pf.new_d`.
        """

        pyemu.os_utils.run(
            f"{self.exe_agent.name} pest.pst /h {self.computername}:{self.port_number}",
            cwd=self.pf.new_d,
            verbose=silent,
        )


class PestIesSolver(PestSolver):
    """PESTPP-IES (Iterative Ensemble Smoother) solver"""

    def __init__(
        self,
        exe_name: str | Path = "pestpp-ies",
        model_ws: str | Path = Path("model"),
        temp_ws: str | Path = Path("temp"),
        master_ws: str | Path = Path("master"),
        noptmax: int = 0,
        ies_num_reals: int = 50,
        control_data: dict[str, Any] | None = None,
        pcov: DataFrame | None = None,
        nfev: int | None = None,
        port_number: int = 4004,
        num_workers: int | None = None,
        use_pypestworker: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize the PESTPP-iES solver.

        Parameters
        ----------
        exe_name : str | Path, optional
            The name of the executable to run, by default "pestpp-ies".
        model_ws : str | Path, optional
            The working directory for the model, by default Path("model").
        temp_ws : str | Path, optional
            The temporary working directory, by default Path("temp").
        master_ws : str | Path, optional
            The master working directory, by default Path("master") unless
            use_pypestworker is True, then master_ws is equal to temp_ws.
        noptmax : int, optional
            The maximum number of optimization iterations, by default 0.
        ies_num_reals : int, optional
            The number of realizations to draw in order to form parameter and observation ensembles, by default 50.
        control_data : dict[str, Any] | None, optional
            Additional control data for the solver, by default None.
        pcov : DataFrame | None, optional
            The parameter covariance matrix, by default None.
        nfev : int | None, optional
            The number of function evaluations, by default None.
        port_number : int, optional
            The port number for communication, by default 4004.
        num_workers : int | None, optional
            The number of worker processes, by default the number of physical CPU cores.
        use_pypestworker : bool, optional
            Whether to use the PyPestWorker for Python processing. Default is True.
        **kwargs
            Additional keyword arguments passed to the base class initializer.

        Returns
        -------
        None
        """

        PestSolver.__init__(
            self,
            exe_name=exe_name,
            model_ws=model_ws,
            temp_ws=temp_ws,
            pcov=pcov,
            nfev=nfev,
            control_data=control_data,
            port_number=port_number,
            use_pypestworker=use_pypestworker,
            **kwargs,
        )

        self.master_ws = temp_ws if self.use_pypestworker else master_ws
        self.noptmax = noptmax
        self.ies_num_reals = ies_num_reals
        self.num_workers = cpu_count() if num_workers is None else num_workers

    def run_ensembles(
        self,
        ies_add_base: bool = True,
        par_sigma_range: float = 4.0,
        observation_noise_standard_deviation: float = 0.0,
        observation_noise_correlation_coefficient: float = 0.0,
        ies_parameter_ensemble_method: Literal["norm", "truncnorm", "uniform"]
        | None = None,
        pestpp_options: dict[str, Any] | None = None,
        silent: bool = False,
        seed: int = pyemu.en.SEED,
    ) -> None:
        """
        Run ensemble simulations using pestpp-ies.

        Parameters
        ----------
        ies_add_base : bool, optional
            Whether to add the base parameter set to the ensemble, by default
            True. The base ensemble uses the initial parameter values as
            provided by Pastas and does not add noise on the observations.
        par_sigma_range : float, optional
            The difference between a parameters upper and lower bounds
            expressed as standard deviations, by default 4.0.
        observation_noise_standard_deviation : float, optional
            The standard deviation of the observation noise, by default 0.0.
        observation_noise_correlation_coefficient : float, optional
            The correlation coefficient of the observation noise, by default 0.0.
        ies_parameter_ensemble_method : Literal["norm", "truncnorm", "uniform"] | None, optional
            The method to distribution of the prior for the parameter ensemble, by default None.
            If None the parameter distribution is drawn by pestpp-ies itself.
        pestpp_options : dict | None, optional
            Additional PEST++ options, by default None.
        Returns
        -------
        None
        """
        self.initialize(version=2)

        # change ies_num_reals
        pst = pyemu.Pst(str(self.temp_ws / "pest.pst"))
        pst.pestpp_options["ies_num_reals"] = self.ies_num_reals
        pst.pestpp_options["ies_add_base"] = ies_add_base
        pst.pestpp_options["par_sigma_range"] = par_sigma_range
        if observation_noise_standard_deviation == 0.0:
            pst.pestpp_options["ies_no_noise"] = True
        else:
            self.write_ensemble_observation_noise(
                standard_deviation=observation_noise_standard_deviation,
                correlation_coefficient=observation_noise_correlation_coefficient,
                seed=seed,
            )
            pst.pestpp_options["ies_observation_ensemble"] = (
                "pest_starting_obs_ensemble.csv"
            )
        if ies_parameter_ensemble_method is not None:
            self.write_ensemble_parameter_distribution(
                method=ies_parameter_ensemble_method,
                par_sigma_range=par_sigma_range,
                ies_add_base=ies_add_base,
                seed=seed,
            )
            pst.pestpp_options["ies_parameter_ensemble"] = (
                "pest_starting_par_ensemble.csv"
            )

        pestpp_options = {} if pestpp_options is None else pestpp_options
        pst.pestpp_options.update(pestpp_options)

        self.write_pst(pst=pst, version=2)

        pyemu.os_utils.start_workers(
            worker_dir=self.temp_ws,  # the folder which contains the "template" PEST dataset
            exe_rel_path=self.exe_name.name,  # the PEST software version we want to run
            pst_rel_path="pest.pst",  # the control file to use with PEST
            num_workers=self.num_workers,  # how many agents to deploy
            worker_root=self.master_ws.parent,  # where to deploy the agent directories; relative to where python is running
            master_dir=self.master_ws,  # the manager directory
            port=self.port_number,  # the port to use for communication
            verbose=silent,
            silent_master=silent,
            reuse_master=self.use_pypestworker,
            ppw_function=self.ppw_function
            if self.use_pypestworker
            else None,  # the function to run in the agent
            ppw_kwargs={"ml": self.ml}
            if self.use_pypestworker
            else {},  # the arguments to pass to the ppw_function
        )

        phidf = pd.read_csv(self.master_ws / "pest.phi.meas.csv", index_col=0)
        self.nfev = phidf.index[-1]
        if self.noptmax > 0:
            self.obj_func = phidf.at[
                self.nfev, "base"
            ]  # could also get mean of all ensembles?

    @staticmethod
    def parameter_distribution(
        ies_num_reals: int,
        initial: float,
        pmin: float,
        pmax: float,
        par_sigma_range: float,
        method: Literal["norm", "truncnorm", "uniform"],
    ) -> NDArray[np.float64]:
        """Generate a distribution of parameter values based on the specified method.

        Parameters
        ----------
        ies_num_reals : int
            Number of ensembles/realizations.
        initial : float
            Initial parameter value.
        pmin : float
            Minimum parameter value.
        pmax : float
            Maximum parameter value.
        par_sigma_range : float
            Range for the parameter sigma.
        method : {'norm', 'truncnorm', 'uniform'}
            Method to use for generating the distribution. 'norm' generates a
            normal distribution, 'truncnorm' generates a truncated normal
            distribution, and 'uniform' generates a uniform distribution.

        Returns
        -------
        np.array
            Array of generated parameter values.
        """
        if method == "norm":
            scale = min(initial - pmin, pmax - initial) / (par_sigma_range / 2)
            rvs = np.sort(norm(loc=initial, scale=scale).rvs(size=ies_num_reals))
            rvs[rvs < pmin] = pmin
            rvs[rvs > pmax] = pmax
        elif method == "truncnorm":
            scale_left = (initial - pmin) / (par_sigma_range / 2)
            tnorm_left = truncnorm(
                a=(pmin - initial) / scale_left, b=0.0, loc=initial, scale=scale_left
            )
            scale_right = (pmax - initial) / (par_sigma_range / 2)
            tnorm_right = truncnorm(
                a=0.0, b=(pmax - initial) / scale_right, loc=initial, scale=scale_right
            )

            left_ies_num_reals = int(
                np.ceil((initial - pmin) / (pmax - pmin) * ies_num_reals)
            )
            right_ies_num_reals = int(
                np.ceil((pmax - initial) / (pmax - pmin) * ies_num_reals)
            )
            rvs_left = tnorm_left.rvs(size=left_ies_num_reals)
            rvs_right = tnorm_right.rvs(size=right_ies_num_reals)
            rvs = np.sort(np.append(rvs_left, rvs_right)[:ies_num_reals])
            rvs[rvs < pmin] = pmin
            rvs[rvs > pmax] = pmax
        elif method == "uniform":
            rvs = np.linspace(
                start=pmin, stop=pmax, num=ies_num_reals
            )  # linspace ensures pmin and pmax are in the ensembles
        else:
            raise ValueError(f"{method=} should be 'norm', 'truncnorm' or 'uniform'.")
        return rvs

    @staticmethod
    def generate_observation_noise(
        ies_num_reals: int,
        nobs: int,
        standard_deviation: float,
        correlation_coefficient: float = 0.0,
        seed: int = pyemu.en.SEED,
    ) -> NDArray[np.float64]:
        """Generate a matrix of normally distributed and optionally correlated noise

        Parameters
        ----------
        ies_num_reals : int
            Number of ensembles/realizations.
        nobs : int
            Number of observations (length of each noise series).
        standard_deviation : float
            Standard deviation of the noise.
        rho : float, optional
            Autoregressive coefficient. Default is 0.0 (pure white noise).
        seed : int, optional
            Random seed for reproducibility, by default pyemu.en.SEED.

        Returns
        -------
        NDArray[np.float64] (nobs, ies_num_reals) matrix
        """
        drng = np.random.default_rng(seed)

        x = drng.normal(loc=0.0, scale=standard_deviation, size=(nobs, ies_num_reals))
        if correlation_coefficient != 0.0:
            sige = np.sqrt(1 - correlation_coefficient**2) * standard_deviation
            e = drng.normal(loc=0.0, scale=sige, size=(nobs, ies_num_reals))
            for j in range(1, nobs):
                x[j] = correlation_coefficient * x[j - 1] + e[j]
        return x

    def write_ensemble_parameter_distribution(
        self,
        method: Literal["norm", "truncnorm", "uniform"] = "norm",
        par_sigma_range: float = 4.0,
        ies_add_base: bool = True,
        seed: int = pyemu.en.SEED,
    ) -> None:
        """
        Generate and write an ensemble of parameter distributions to a CSV file.

        Parameters
        ----------
        method : Literal["norm", "truncnorm", "uniform"], optional
            The method to use for generating the parameter distribution.
            Options are "norm" for normal distribution, "truncnorm" for
            truncated normal distribution, and "uniform" for uniform
            distribution. Default is "norm".
        par_sigma_range : float, optional
            The range of the parameter sigma for the distribution. Default is
            4.0.
        ies_add_base : bool, optional
            If True, add the base parameter values to the ensemble. Default is
            True.
        seed : int, optional
            Random seed for reproducibility, by default pyemu.en.SEED.

        Returns
        -------
        None
        """
        pst = pyemu.Pst(str(self.temp_ws / "pest.pst"))
        par_df = pd.DataFrame(
            index=pd.Index(range(self.ies_num_reals)), columns=pst.parameter_data.index
        )
        for pname, pdata in pst.parameter_data.iterrows():
            rvs = PestIesSolver.parameter_distribution(
                ies_num_reals=self.ies_num_reals,
                initial=pdata.at["parval1"],
                pmin=pdata.at["parlbnd"],
                pmax=pdata.at["parubnd"],
                par_sigma_range=par_sigma_range,
                method=method,
            )
            par_df[pname] = rvs
        # shuffle each column with the initial parameters independently
        par_df.loc[:, :] = np.random.default_rng(seed=seed).permuted(
            par_df.values, axis=0
        )
        if ies_add_base:
            par_df.loc[self.ies_num_reals - 1] = pst.parameter_data.loc[
                :, "parval1"
            ].values
            par_df = par_df.rename(index={self.ies_num_reals - 1: "base"})
        par_df.to_csv(self.temp_ws / "pest_starting_par_ensemble.csv")

    def write_ensemble_observation_noise(
        self,
        standard_deviation: float = 0.0,
        correlation_coefficient: float = 0.0,
        ies_add_base: bool = True,
        seed: int = pyemu.en.SEED,
    ) -> None:
        """
        Generate and write an ensemble of observation noise to a CSV file.

        Parameters
        ----------
        standard_deviation : float, optional
            The standard deviation of the observation noise. Default is 0.0.
        correlation_coefficient : float, optional
            The correlation coefficient of the observation noise. Default is
            0.0.
        ies_add_base : bool, optional
            If True, add the base observation values to the ensemble. Default
            is True.

        Returns
        -------
        None
        """
        pst = pyemu.Pst(str(self.temp_ws / "pest.pst"))
        noise = PestIesSolver.generate_observation_noise(
            ies_num_reals=self.ies_num_reals,
            nobs=len(pst.observation_data.index),
            standard_deviation=standard_deviation,
            correlation_coefficient=correlation_coefficient,
            seed=seed,
        )
        obs_data = pst.observation_data.loc[:, ["obsval"]].values
        obs_noise_df = pd.DataFrame(
            obs_data + noise,
            index=pst.observation_data.index,
            columns=pd.Index(range(self.ies_num_reals)),
        ).transpose()
        if ies_add_base:
            obs_noise_df.loc[self.ies_num_reals - 1] = obs_data.flatten()
            obs_noise_df = obs_noise_df.rename(index={self.ies_num_reals - 1: "base"})
        obs_noise_df.to_csv(self.temp_ws / "pest_starting_obs_ensemble.csv")

    def parameter_ensemble(self, iteration: int = 0) -> pyemu.ParameterEnsemble:
        """
        Read a parameter ensemble for a given iteration.

        Parameters:
        -----------
        iteration : int, optional
            The iteration number for which to read the parameter ensemble.
            Default is 0.

        Returns:
        --------
        pyemu.ParameterEnsemble
            The parameter ensemble for the specified iteration.
        """

        pst = pyemu.Pst(str(self.master_ws / "pest.pst"))
        pe = pyemu.ParameterEnsemble.from_csv(
            pst=pst, filename=self.master_ws / f"pest.{iteration}.par.csv"
        )
        return pe

    @lru_cache()
    def simulation_ensemble(
        self,
        iteration: int = 0,
        from_file: bool = False,
        tmin: TimestampType = None,
        tmax: TimestampType = None,
    ) -> pd.DataFrame:
        """
        Generate or read a simulation ensemble.

        Parameters
        ----------
        iteration : int, optional
            The iteration number for which to read the simulation ensemble.
            Default is 0.
        from_file : bool, optional
            If True, read the simulation ensemble from a file. If False,
            generate it with Pastas from the parameter ensemble. Default is
            False.
        tmin : TimestampType, optional
            The minimum timestamp for the simulation period. If None, use the
            model's tmin setting. Default is None.
        tmax : TimestampType, optional
            The maximum timestamp for the simulation period. If None, use the
            model's tmax setting. Default is None.

        Returns
        -------
        pd.DataFrame
            The simulation ensemble as a DataFrame.
        """
        if from_file:
            pst = pyemu.Pst(str(self.master_ws / "pest.pst"))
            se = (
                pyemu.ObservationEnsemble.from_csv(
                    pst=pst, filename=self.master_ws / f"pest.{iteration}.obs.csv"
                )
                .transpose()
                .set_index(self.ml.observations().index)
            )
        else:
            ipar = self.parameter_ensemble(iteration=iteration).transpose()
            ipar.index = self.ml.parameters.index[self.vary]

            tmin = self.ml.settings["tmin"] if tmin is None else pd.Timestamp(tmin)
            tmax = self.ml.settings["tmax"] if tmax is None else pd.Timestamp(tmax)
            freq = (
                "D"
                if self.ml.settings["freq"] is not None
                else self.ml.settings["freq"]
            )
            se = pd.DataFrame(
                np.nan,
                columns=ipar.columns,
                index=pd.date_range(start=tmin, end=tmax, freq=freq),
            )

            for idx in ipar.columns:
                self.ml.parameters.loc[ipar.index, "optimal"] = ipar.loc[:, idx].values
                se.loc[:, idx] = (
                    self.ml.simulate(tmin=tmin, tmax=tmax).loc[se.index].values
                )

        return se

    def observation_ensemble(self) -> pyemu.ObservationEnsemble:
        """
        Generate an observation ensemble from a CSV file. This method reads a
        PEST control file and a corresponding observation noise CSV file to
        create an observation ensemble.

        Returns
        -------
        pyemu.ObservationEnsemble
            The generated observation ensemble.
        """

        pst = pyemu.Pst(str(self.master_ws / "pest.pst"))
        oe = pyemu.ObservationEnsemble.from_csv(
            pst=pst, filename=self.master_ws / "pest.obs+noise.csv"
        )
        return oe

    def jacobian(self, iteration: int = 0) -> pd.DataFrame:
        """
        Calculate the Jacobian matrix for the given iteration.
        The Jacobian matrix is computed using the simulation ensemble and
        parameter ensemble for the specified iteration. The calculation
        involves normalizing the ensembles and using the pseudo-inverse
        of the parameter ensemble.

        Parameters:
        -----------
        iteration : int, optional
            The iteration number for which the Jacobian is calculated.
            Default is 0.

        Returns:
        --------
        pd.DataFrame
            A DataFrame representing the Jacobian matrix, with the same
            indices as the observation ensemble and the same columns as
            the parameter ensemble.
        """
        # jac_ies needs to be calculated manually
        obs_ies = self.simulation_ensemble(
            iteration=iteration, from_file=True
        ).transpose()
        par_ies = self.parameter_ensemble(iteration=iteration)
        jac = PestIesSolver.jacobian_empirical(obs_ies.values, par_ies.values)
        jac_ies = pd.DataFrame(jac, index=obs_ies.index, columns=par_ies.columns)
        return jac_ies

    @staticmethod
    def jacobian_empirical(
        simulation_ensembles: NDArray[np.float64],
        parameter_ensembles: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate the approximate Jacobian matrix for the given ensembles.

        Parameters
        ----------
        simulation_ensembles : NDArray[np.float64]
            Ensembles of the simulated values of shape (nobs, nreals)
        parameter_ensembles : NDArray[np.float64]
            Ensembles of the paramters of shape (nreals, npar)

        Returns
        -------
        NDArray[np.float64]
            Approximate, empirical Jacobian matrix
        """
        _, ies_num_reals_sim = simulation_ensembles.shape
        ies_num_reals_par, _ = parameter_ensembles.shape
        if ies_num_reals_par != ies_num_reals_sim:
            raise AssertionError(
                f"Number of realizations in parameter {ies_num_reals_par} and"
                f"simulation {ies_num_reals_sim} ensembles must be equal"
            )
        else:
            ies_num_reals = ies_num_reals_sim

        deviations_sim = (
            simulation_ensembles.T - np.mean(simulation_ensembles, axis=1)
        ).T / np.sqrt(ies_num_reals - 1)
        deviations_par = (
            parameter_ensembles - np.mean(parameter_ensembles, axis=0)
        ).T / np.sqrt(ies_num_reals - 1)
        jac = deviations_sim @ np.linalg.pinv(deviations_par)
        return jac

    def solve(
        self, run_ensembles: bool = False, **kwargs
    ) -> tuple[bool, NDArray[np.float64], NDArray[np.float64]]:
        """
        Gets the base realization of the parameter ensemble.

        Parameters
        ----------
        run_ensembles : bool, optional
            If True, runs the ensembles with the provided keyword arguments (default is False).
        **kwargs : dict
            Additional keyword arguments to pass to the `run_ensembles` method.

        Returns
        -------
        tuple
            A tuple containing:
            - bool: Always returns True.
            - numpy.ndarray: The optimal parameters.
            - numpy.ndarray: The standard error of the parameters.
        """
        if "noise" in kwargs:
            del kwargs["noise"]  # remove noise from kwargs, not used in PestIesSolver
        if "weights" in kwargs:
            del kwargs["weights"]

        if run_ensembles:
            self.run_ensembles(**kwargs)

        optimal, stderr = self.get_solve_results()

        return True, optimal, stderr

    def get_solve_results(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Get the results of the last solve operation.

        Returns
        -------
        tuple
            - numpy.ndarray: The optimal parameters.
            - numpy.ndarray: The standard error of the parameters.
        """
        # optimal parameters
        ipar = self.parameter_ensemble(iteration=self.nfev).transpose()
        ipar.index = self.ml.parameters.index[self.vary]
        optimal = self.ml.parameters["initial"].copy().values
        optimal[self.vary] = ipar.loc[:, "base"].values

        # standard error (could be totally the wrong way to think about/calculate this)
        stderr = np.full_like(optimal, np.nan)
        stderr[self.vary] = ipar.std(axis=1) / np.sqrt(len(ipar.columns))

        return optimal, stderr


class PestSenSolver(PestSolver):
    """PESTPP-SEN (Global Sensitivity Analysis) solver"""

    def __init__(
        self,
        exe_name: str | Path = "pestpp-sen",
        model_ws: str | Path = Path("model"),
        temp_ws: str | Path = Path("temp"),
        master_ws: str | Path = Path("master"),
        noptmax: int = 0,
        control_data: dict[str, Any] | None = None,
        pcov: DataFrame | None = None,
        nfev: int | None = None,
        port_number: int = 4004,
        num_workers: int | None = None,
        use_pypestworker: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize the PESTPP-SEN class. This class is used to run the
        PESTPP-SEN analysis and is not really a solver.

        Parameters
        ----------
        exe_name : str | Path, optional
            The name of the executable to run, by default "pestpp-sen".
        model_ws : str | Path, optional
            The working directory for the model, by default Path("model").
        temp_ws : str | Path, optional
            The temporary working directory, by default Path("temp").
        master_ws : str | Path, optional
            The master working directory, by default Path("master") unless
            use_pypestworker is True, then master_ws is equal to temp_ws.
        noptmax : int, optional
            The maximum number of optimization iterations, by default 0.
        control_data : dict[str, Any] | None, optional
            Control data for the solver, by default None.
        pcov : DataFrame | None, optional
            The parameter covariance matrix, by default None.
        nfev : int | None, optional
            The number of function evaluations, by default None.
        port_number : int, optional
            The port number for communication, by default 4004.
        num_workers : int | None, optional
            The number of worker processes, by default the number of physical CPU cores.
        use_pypestworker : bool, optional
            Whether to use the PyPestWorker for Python processing. Default is True.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        None
        """
        PestSolver.__init__(
            self,
            exe_name=exe_name,
            model_ws=model_ws,
            temp_ws=temp_ws,
            pcov=pcov,
            nfev=nfev,
            control_data=control_data,
            port_number=port_number,
            use_pypestworker=use_pypestworker,
            **kwargs,
        )
        self.master_ws = temp_ws if self.use_pypestworker else master_ws
        self.noptmax = noptmax
        self.num_workers = cpu_count() if num_workers is None else num_workers

    def start(
        self, pestpp_options: dict[str, Any] | None = None, silent: bool = False
    ) -> None:
        """
        Start the PESTPP-SEN analysis.

        This method sets up the model and files, updates the PEST control file with
        the provided PEST++ options, and starts the PESTPP-SEN workers.

        Parameters
        ----------
        pestpp_options : dict[str, Any], optional
            Additional PEST++ options to update in the PEST control file, by default None.

        Returns
        -------
        None
        """

        self.initialize(version=2)

        # change ies_num_reals
        pst = pyemu.Pst(str(self.temp_ws / "pest.pst"))
        pestpp_options = {} if pestpp_options is None else pestpp_options
        pst.pestpp_options.update(pestpp_options)

        self.write_pst(pst=pst, version=2)

        pyemu.os_utils.start_workers(
            worker_dir=self.temp_ws,  # the folder which contains the "template" PEST dataset
            exe_rel_path=self.exe_name.name,  # the PEST software version we want to run
            pst_rel_path="pest.pst",  # the control file to use with PEST
            num_workers=self.num_workers,  # how many agents to deploy
            worker_root=self.master_ws.parent,  # where to deploy the agent directories; relative to where python is running
            port=self.port_number,  # the port to use for communication
            master_dir=self.master_ws,  # the manager directory
            reuse_master=self.use_pypestworker,
            verbose=silent,
            silent_master=silent,
            ppw_function=self.ppw_function
            if self.use_pypestworker
            else None,  # the function to run in the agent
            ppw_kwargs={"ml": self.ml}
            if self.use_pypestworker
            else {},  # the arguments to pass to the ppw_function
        )

    def solve() -> None:
        raise NotImplementedError(
            "PestSenSolver does not have a solve method. Run the sensitivity"
            "analysis using the `start` method."
        )


class RandomizedMaximumLikelihoodSolver(BaseSolver):
    _name = "RandomizedMaximumLikelihoodSolver"

    def __init__(
        self,
        num_reals: int,
        jacobian_method: Literal["2-point", "3-point", "empirical"] = "3-point",
        noptmax: int | None = None,
        seed: int | None = pyemu.en.SEED,
        add_base: bool = True,
        num_workers: int | None = None,
        pcov: pd.DataFrame | None = None,
        nfev: int | None = None,
        **kwargs,
    ):
        super().__init__(pcov=pcov, nfev=nfev, **kwargs)
        self.num_reals = num_reals
        self.jacobian_method = jacobian_method
        if noptmax is None and jacobian_method == "empirical":
            logger.error(
                "noptmax must be specified when using 'empirical' jacobian method."
            )
        self.noptmax = noptmax
        self.seed = seed
        self.add_base = add_base
        self.num_workers = cpu_count() if num_workers is None else num_workers
        self.parameter_ensemble: pd.DataFrame | None = None
        self.observation_noise: pd.DataFrame | None = None
        self.simulation_ensemble: pd.DataFrame | None = None

    def __repr__(self) -> str:
        _repr = (
            "RandomizedMaximumLikelihoodSolver("
            f"num_reals={self.num_reals},"
            f"jacobian_method={self.jacobian_method})"
        )
        return _repr

    def to_dict(self) -> dict:
        data = {
            "class": self._name,
            "num_reals": self.num_reals,
            "jacobian_method": self.jacobian_method,
            "noptmax": self.noptmax,
            "seed": self.seed,
            "add_base": self.add_base,
        }

        # TODO: Use RMLSolver attributes, now go for BaseSolver otherwise can't be stored in PastaStore
        self.nfev = self.noptmax
        self.obj_func = 0.0
        data = super().to_dict()
        return data

    def initialize(
        self,
        standard_deviation: float = 0.0,
        correlation_coefficient: float = 0.0,
        par_sigma_range: float = 4.0,
        method: Literal["norm", "truncnorm", "uniform"] = "norm",
    ) -> None:
        logger.debug("Initialize: RML solver")
        logger.debug("Initialize: Creating observation noise")
        observations = self.ml.observations()
        observation_noise = pd.DataFrame(
            PestIesSolver.generate_observation_noise(
                ies_num_reals=self.num_reals,
                nobs=len(observations),
                standard_deviation=standard_deviation,
                correlation_coefficient=correlation_coefficient,
                seed=self.seed,
            ),
            columns=pd.Index(range(self.num_reals)),
            index=observations.index,
        )

        logger.debug("Initialize: Creating parameter noise")
        parameter_data = pd.DataFrame(
            index=pd.Index(range(self.num_reals)),
            columns=self.ml.parameters.index,
            dtype=float,
        )
        for pname, pdata in self.ml.parameters.iterrows():
            rvs = PestIesSolver.parameter_distribution(
                ies_num_reals=self.num_reals,
                initial=pdata.at["initial"],
                pmin=pdata.at["pmin"],
                pmax=pdata.at["pmax"],
                par_sigma_range=par_sigma_range,
                method=method,
            )
            parameter_data[pname] = rvs
        parameter_data.loc[:, :] = np.random.default_rng(seed=self.seed).permuted(
            parameter_data.values, axis=0
        )
        if self.add_base:
            logger.debug("Initialize: Adding base realization")
            base_idx = self.num_reals - 1
            observation_noise.loc[:, base_idx] = 0.0
            observation_noise = observation_noise.rename(columns={base_idx: "base"})

            parameter_data.loc[base_idx, :] = self.ml.parameters.loc[
                :, "initial"
            ].values
            parameter_data = parameter_data.rename(index={base_idx: "base"})

        self.parameter_ensemble = parameter_data
        self.observation_noise = observation_noise

    @property
    def observation_ensemble(self) -> pd.DataFrame:
        """Generate the observation ensemble by adding noise to model observations."""
        obs = self.ml.observations().values
        noise = self.observation_noise.values.T

        obs_df = pd.DataFrame(
            noise + obs,
            index=self.observation_noise.columns,
            columns=self.ml.observations().index,
        ).T
        return obs_df

    @staticmethod
    def jacobian_empirical(
        simulation_ensembles: ArrayLike, parameter_ensembles: ArrayLike
    ) -> ArrayLike:
        """Approximate the Jacobian matrix using ensemble perturbations."""
        return PestIesSolver.jacobian_empirical(
            simulation_ensembles=simulation_ensembles,
            parameter_ensembles=parameter_ensembles,
        )

    @staticmethod
    def jacobian_finite_difference(
        fun: Callable[[ArrayLike], ArrayLike],
        p: ArrayLike,
        jacobian_method: Literal["2-point", "3-point"],
        bounds=None,
    ) -> ArrayLike:
        """Compute Jacobian via finite differences."""
        return approx_derivative(
            fun=fun,
            x0=p,
            method=jacobian_method,
            bounds=bounds,
        )

    @staticmethod
    def _least_squares_fd(
        real: int,
        parameter_ensemble: pd.DataFrame,
        observation_ensemble: pd.DataFrame,
        ml: Model,
        jacobian_method: Literal["2-point", "3-point"],
        **kwargs,
    ) -> pd.Series:
        """Perform least squares optimization for a single realization (finite diff)."""
        logger.debug(f"RML: Starting least squares for realization {real}")

        observations = observation_ensemble.iloc[:, real]
        p = parameter_ensemble.iloc[real]

        def fun(p: ArrayLike) -> ArrayLike:
            sim = ml.simulate(p)
            res = observations - sim.loc[observations.index]
            return res.values

        bounds = (
            ml.parameters.loc[p.index, "pmin"].values,
            ml.parameters.loc[p.index, "pmax"].values,
        )

        def jac(p: ArrayLike) -> ArrayLike:
            return RandomizedMaximumLikelihoodSolver.jacobian_finite_difference(
                fun=fun, p=p, jacobian_method=jacobian_method, bounds=bounds
            )

        result = least_squares(fun, p, jac=jac, bounds=bounds, **kwargs)

        return pd.Series(result.x, index=parameter_ensemble.columns, name=real)

    @staticmethod
    def _least_squares_em(
        real: int,
        simulations: pd.DataFrame,
        parameter_ensemble: pd.DataFrame,
        observation_ensemble: pd.DataFrame,
        ml: Model,
        jacobian: ArrayLike | None = None,
    ) -> pd.Series:
        """Perform one empirical least squares update."""
        obs = observation_ensemble.iloc[:, real]
        sims = simulations.loc[obs.index, :]
        sim = sims.iloc[:, real]
        p0 = parameter_ensemble.iloc[real]
        bounds = (
            ml.parameters.loc[p0.index, "pmin"].values,
            ml.parameters.loc[p0.index, "pmax"].values,
        )
        if jacobian is None:
            jacobian = RandomizedMaximumLikelihoodSolver.jacobian_empirical(
                simulation_ensembles=sims.values,
                parameter_ensembles=parameter_ensemble.values,
            )

        # Option 1: Gauss–Newton / Levenberg–Marquardt step
        JTJ = jacobian.T @ jacobian
        g = jacobian.T @ (obs - sim).values

        # Solve for delta p
        lam = 1e-8 * np.max(np.diag(JTJ))  # could adapt per-iteration
        delta, *_ = np.linalg.lstsq(JTJ + lam * np.eye(JTJ.shape[0]), g, rcond=None)

        p_new = np.clip(p0.values + delta, bounds[0], bounds[1])

        # Option 2: SciPy least squares with linearized residuals
        # def jac(_) -> ArrayLike:
        #     return jacobian

        # def fun(p) -> ArrayLike:
        #     return (obs - sim).values + jac(None) @ (p - p0).values

        # result = least_squares(fun, x0=p0.values, jac=jac, max_nfev=2, bounds=bounds)
        # p_new = result.x

        # print(f"Real {real}: {p0.values}, {p_new},  Δp = {p_new - p0.values}")
        return pd.Series(p_new, index=parameter_ensemble.columns, name=real)

    @staticmethod
    def _simulate(real: int, parameters: pd.DataFrame, ml: Model) -> pd.Series:
        """Run the model simulation for one realization."""
        p = parameters.iloc[real].values
        return ml.simulate(p=p).rename(real)

    def solve(self, **kwargs) -> tuple[bool, pd.Series, None]:
        """Solve the RML problem using least squares optimization."""

        if "noise" in kwargs:
            _ = kwargs.pop("noise")

        if "weights" in kwargs:
            _ = kwargs.pop("weights")

        if self.jacobian_method in ("2-point", "3-point"):
            func = partial(
                RandomizedMaximumLikelihoodSolver._least_squares_fd,
                parameter_ensemble=self.parameter_ensemble,
                observation_ensemble=self.observation_ensemble,
                ml=self.ml,
                jacobian_method=self.jacobian_method,
            )

            results = process_map(
                func,
                range(self.num_reals),
                max_workers=self.num_workers,
                desc="RML looping over realizations",
                chunksize=1,
            )

            self.parameter_ensemble = pd.DataFrame(
                results, index=self.parameter_ensemble.index
            )
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                sims = [
                    executor.submit(
                        RandomizedMaximumLikelihoodSolver._simulate,
                        r,
                        self.parameter_ensemble,
                        self.ml,
                    )
                    for r in range(self.num_reals)
                ]
                self.simulation_ensemble = pd.concat([f.result() for f in sims], axis=1)

        elif self.jacobian_method == "empirical":
            parameter_ensemble = self.parameter_ensemble.copy()
            for _ in tqdm(range(self.noptmax), desc="RML looping over noptmax"):
                # simulate ensembles in parallel
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = [
                        executor.submit(
                            RandomizedMaximumLikelihoodSolver._simulate,
                            r,
                            parameter_ensemble,
                            self.ml,
                        )
                        for r in range(self.num_reals)
                    ]
                    simulations = pd.concat([f.result() for f in futures], axis=1)
                    if self.add_base:
                        simulations.columns = list(range(self.num_reals - 1)) + ["base"]

                # one least squares update
                jacobian = RandomizedMaximumLikelihoodSolver.jacobian_empirical(
                    simulation_ensembles=simulations.loc[
                        self.observation_ensemble.index
                    ].values,
                    parameter_ensembles=parameter_ensemble.values,
                )
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = [
                        executor.submit(
                            RandomizedMaximumLikelihoodSolver._least_squares_em,
                            real=r,
                            simulations=simulations,
                            parameter_ensemble=parameter_ensemble,
                            observation_ensemble=self.observation_ensemble,
                            ml=self.ml,
                            jacobian=jacobian,
                        )
                        for r in range(self.num_reals)
                    ]
                    parameter_ensemble = pd.DataFrame(
                        [f.result() for f in futures], index=parameter_ensemble.index
                    )

            self.simulation_ensemble = simulations
            self.parameter_ensemble = parameter_ensemble

        res = self.observation_ensemble - self.simulation_ensemble
        self.obj_func = float(np.mean((np.sum(res.values**2, axis=0))))
        self.nfev = self.num_reals if self.noptmax is None else self.noptmax

        if self.add_base:
            optimal = self.parameter_ensemble.loc["base"]
        else:
            optimal = self.parameter_ensemble.mean(axis=0)

        stderr = self.parameter_ensemble.std().values

        return True, optimal, stderr
