import json
import logging
from functools import lru_cache
from pathlib import Path
from platform import node as get_computername
from shutil import copy as copy_file
from threading import Thread
from time import sleep
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyemu
from pandas import DataFrame
from pastas.solver import BaseSolver
from pastas.typing import TimestampType
from psutil import cpu_count
from scipy.stats import norm, truncnorm

logger = logging.getLogger(__name__)


class PestSolver(BaseSolver):
    """PEST solver base class"""

    def __init__(
        self,
        exe_name: Union[str, Path],
        model_ws: Union[str, Path] = Path("model"),
        temp_ws: Union[str, Path] = Path("temp"),
        noptmax: int = 0,
        control_data: Optional[dict] = None,
        pcov: Optional[DataFrame] = None,
        nfev: Optional[int] = None,
        long_names: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the PEST solver.

        Parameters
        ----------
        exe_name : Union[str, Path]
            The name or path to the PEST executable.
        model_ws : Union[str, Path], optional
            The model workspace directory for Pastas files. Default is "model".
        temp_ws : Union[str, Path], optional
            The template workspace directory for PEST files. Default is "temp".
        noptmax : int, optional
            The maximum number of optimization iterations. Default is 0.
        control_data : Optional[dict], optional
            Control data for the PEST solver. Default is None.
        pcov : Optional[DataFrame], optional
            The parameter covariance matrix. Default is None.
        nfev : Optional[int], optional
            The number of function evaluations. Default is None.
        long_names : bool, optional
            Whether to use long names in the PEST control file. Default is True.
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
        """Setup and export Pastas model for PEST optimization"""
        # observations
        observations = self.ml.observations()
        observations.name = "Observations"
        observations.to_csv(self.model_ws / "simulation.csv")
        copy_file(self.model_ws / "simulation.csv", self.temp_ws)
        self.observations = observations

        # setup parameters
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

        par_sel = parameters.loc[:, ["optimal"]]
        par_sel.to_csv(self.model_ws / "parameters_sel.csv")
        copy_file(self.model_ws / "parameters_sel.csv", self.temp_ws)
        self.par_sel = par_sel

        # model
        self.ml.to_file(self.model_ws / "model.pas")
        copy_file(self.model_ws / "model.pas", self.temp_ws)

        # write run function
        self.write_run_function()

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
        self.pf.add_py_function(
            self.model_ws / "_run_pastas_model.py", "run()", is_pre_cmd=None
        )
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
                if key == "control_data":
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
        """
        Initialize the PESTPP-GLM solver.

        Parameters
        ----------
        exe_name : Union[str, Path], optional
            The name or path to the PESTPP-GLM executable. Default is "pestpp-glm".
        model_ws : Union[str, Path], optional
            The model workspace directory for Pastas files. Default is "model".
        temp_ws : Union[str, Path], optional
            The template workspace directory for PEST files. Default is "temp".
        noptmax : int, optional
            The maximum number of optimization iterations. Default is 0.
        control_data : Optional[dict], optional
            Control data for the PEST solver. Default is None.
        pcov : Optional[DataFrame], optional
            The parameter covariance matrix. Default is None.
        nfev : Optional[int], optional
            The number of function evaluations. Default is None.
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
            long_names=True,
            **kwargs,
        )

    def solve(self, **kwargs) -> Tuple[bool, np.ndarray, np.ndarray]:
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
        optimal : np.ndarray
            The optimal parameters obtained from the solver.
        stderr : np.ndarray
            The standard errors of the optimal parameters.
        """

        self.setup_model()
        self.setup_files()
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
        """
        Initialize the PEST_HP solver.

        Parameters
        ----------
        exe_name : Union[str, Path], optional
            The name or path to the PEST_HP executable. Default is "pest_hp".
        exe_agent : Union[str, Path], optional
            The name or path to the agent_HP executable. Default is "agent_hp".
        model_ws : Union[str, Path], optional
            The model workspace directory for Pastas files. Default is "model".
        temp_ws : Union[str, Path], optional
            The template workspace directory for PEST files. Default is "temp".
        noptmax : int, optional
            The maximum number of optimization iterations. Default is 0.
        control_data : Optional[dict], optional
            Control data for the PEST solver. Default is None.
        pcov : Optional[DataFrame], optional
            The parameter covariance matrix. Default is None.
        nfev : Optional[int], optional
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
            **kwargs,
        )
        self.exe_agent = Path(exe_agent)
        self.port_number = port_number
        self.computername = get_computername()
        copy_file(self.exe_agent, self.temp_ws)  # copy agent executable

    def solve(
        self, silent: bool = False, **kwargs
    ) -> Tuple[bool, np.ndarray, np.ndarray]:
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
        optimal : np.ndarray
            The optimal parameters obtained from the solver.
        stderr : np.ndarray
            The standard errors of the optimal parameters.
        """
        self.setup_model()
        self.setup_files(version=1)
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

    def run_agent(self, silent: bool = False):
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
        exe_name: Union[str, Path] = "pestpp-ies",
        model_ws: Union[str, Path] = Path("model"),
        temp_ws: Union[str, Path] = Path("temp"),
        master_ws: Union[str, Path] = Path("master"),
        noptmax: int = 0,
        ies_num_reals: int = 50,
        control_data: Optional[dict] = None,
        pcov: Optional[DataFrame] = None,
        nfev: Optional[int] = None,
        port_number: int = 4004,
        num_workers: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the PESTPP-iES solver.

        Parameters
        ----------
        exe_name : Union[str, Path], optional
            The name of the executable to run, by default "pestpp-ies".
        model_ws : Union[str, Path], optional
            The working directory for the model, by default Path("model").
        temp_ws : Union[str, Path], optional
            The temporary working directory, by default Path("temp").
        master_ws : Union[str, Path], optional
            The master working directory, by default Path("master").
        noptmax : int, optional
            The maximum number of optimization iterations, by default 0.
        ies_num_reals : int, optional
            The number of realizations to draw in order to form parameter and observation ensembles, by default 50.
        control_data : Optional[dict], optional
            Additional control data for the solver, by default None.
        pcov : Optional[DataFrame], optional
            The parameter covariance matrix, by default None.
        nfev : Optional[int], optional
            The number of function evaluations, by default None.
        port_number : int, optional
            The port number for communication, by default 4004.
        num_workers : Optional[int], optional
            The number of worker processes, by default the number of physical CPU cores.
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
            **kwargs,
        )
        self.master_ws = master_ws
        self.noptmax = noptmax
        self.ies_num_reals = ies_num_reals
        self.control_data = control_data
        self.port_number = port_number
        self.num_workers = (
            cpu_count(logical=False) if num_workers is None else num_workers
        )

    def run_ensembles(
        self,
        ies_add_base: bool = True,
        par_sigma_range: float = 4.0,
        observation_noise_standard_deviation=0.0,
        observation_noise_correlation_coefficient=0.0,
        ies_parameter_ensemble_method: Optional[
            Literal["norm", "truncnorm", "uniform"]
        ] = None,
        pestpp_options: Optional[Dict] = None,
        silent: bool = False,
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
        ies_parameter_ensemble_method : Optional[Literal["norm", "truncnorm", "uniform"]], optional
            The method to distribution of the prior for the parameter ensemble, by default None.
            If None the parameter distribution is drawn by pestpp-ies itself.
        pestpp_options : Optional[Dict], optional
            Additional PEST++ options, by default None.
        Returns
        -------
        None
        """
        self.setup_model()
        self.setup_files()

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
            )
            pst.pestpp_options["ies_observation_ensemble"] = (
                "pest_starting_obs_ensemble.csv"
            )
        if ies_parameter_ensemble_method is not None:
            self.write_ensemble_parameter_distribution(
                method=ies_parameter_ensemble_method,
                par_sigma_range=par_sigma_range,
                ies_add_base=ies_add_base,
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
            port=self.port_number,  # the port to use for communication
            master_dir=self.master_ws,  # the manager directory
            verbose=silent,
            silent_master=silent,
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
    ) -> np.ndarray[float]:
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
    ) -> np.ndarray[float]:
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
        np.ndarray (nobs, ies_num_reals) matrix
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
    ):
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
            seed=pyemu.en.SEED,
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
        jac = PestIesSolver.jacobian_emperical(obs_ies.values, par_ies.values)
        jac_ies = pd.DataFrame(jac, index=obs_ies.index, columns=par_ies.columns)
        return jac_ies

    @staticmethod
    def jacobian_emperical(
        simulation_ensembles: np.ndarray, parameter_ensembles: np.ndarray
    ) -> np.ndarray:
        """Calculate the approximate Jacobian matrix for the given ensembles.

        Parameters
        ----------
        simulation_ensembles : np.ndarray
            Ensembles of the simulated values of shape (nobs, nreals)
        parameter_ensembles : np.ndarray
            Ensembles of the paramters of shape (nreals, npar)

        Returns
        -------
        np.ndarray
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
    ) -> Tuple[bool, np.ndarray, np.ndarray]:
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
        if run_ensembles:
            self.run_ensembles(**kwargs)

        # optimal parameters
        ipar = self.parameter_ensemble(iteration=self.nfev).transpose()
        ipar.index = self.ml.parameters.index[self.vary]
        optimal = self.ml.parameters["initial"].copy().values
        optimal[self.vary] = ipar.loc[:, "base"].values

        # standard error (could be totally the wrong way to think about/calculate this)
        stderr = np.full_like(optimal, np.nan)
        stderr[self.vary] = ipar.std(axis=1) / np.sqrt(len(ipar.columns))

        return True, optimal, stderr


class PestSenSolver(PestSolver):
    """PESTPP-SEN (Global Sensitivity Analysis) solver"""

    def __init__(
        self,
        exe_name: Union[str, Path] = "pestpp-sen",
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
        """
        Initialize the PESTPP-SEN class. This class is used to run the
        PESTPP-SEN analysis and is not really a solver.

        Parameters
        ----------
        exe_name : Union[str, Path], optional
            The name of the executable to run, by default "pestpp-sen".
        model_ws : Union[str, Path], optional
            The working directory for the model, by default Path("model").
        temp_ws : Union[str, Path], optional
            The temporary working directory, by default Path("temp").
        master_ws : Union[str, Path], optional
            The master working directory, by default Path("master").
        noptmax : int, optional
            The maximum number of optimization iterations, by default 0.
        control_data : Optional[dict], optional
            Control data for the solver, by default None.
        pcov : Optional[DataFrame], optional
            The parameter covariance matrix, by default None.
        nfev : Optional[int], optional
            The number of function evaluations, by default None.
        port_number : int, optional
            The port number for communication, by default 4004.
        num_workers : Optional[int], optional
            The number of worker processes, by default the number of physical CPU cores.
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
            **kwargs,
        )
        self.master_ws = master_ws
        self.noptmax = noptmax
        self.control_data = control_data
        self.port_number = port_number
        self.num_workers = (
            cpu_count(logical=False) if num_workers is None else num_workers
        )

    def start(
        self, pestpp_options: Optional[Dict] = None, silent: bool = False
    ) -> None:
        """
        Start the PESTPP-SEN analysis.

        This method sets up the model and files, updates the PEST control file with
        the provided PEST++ options, and starts the PESTPP-SEN workers.

        Parameters
        ----------
        pestpp_options : Optional[Dict], optional
            Additional PEST++ options to update in the PEST control file, by default None.

        Returns
        -------
        None
        """

        self.setup_model()
        self.setup_files()

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
            verbose=silent,
            silent_master=silent,
        )

    def solve() -> None:
        raise NotImplementedError(
            "PestSenSolver does not have a solve method. Run the sensitivity"
            "analysis using the `start` method."
        )
