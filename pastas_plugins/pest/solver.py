from pathlib import Path
from shutil import copy as copy_file
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyemu
from pandas import DataFrame
from pastas.solver import BaseSolver

np.random.seed(pyemu.en.SEED)  # set seed


class PestSolver(BaseSolver):
    """PEST solver base class"""

    def __init__(
        self,
        exe_name: Union[str, Path] = "pestpp",
        model_ws: Union[str, Path] = Path("model"),
        temp_ws: Union[str, Path] = Path("temp"),
        pcov: Optional[DataFrame] = None,
        nfev: Optional[int] = None,
        **kwargs,
    ) -> None:
        BaseSolver.__init__(self, pcov=pcov, nfev=nfev, **kwargs)
        np.random.seed(pyemu.en.SEED)
        self.temp_ws = Path(temp_ws)
        self.model_ws = Path(model_ws)
        if not self.model_ws.exists():
            self.model_ws.mkdir(parents=True)
        self.exe_name = Path(exe_name)
        self.pf = pyemu.utils.PstFrom(
            original_d=self.model_ws, new_d=self.temp_ws, remove_existing=True
        )

    def setup_model(self):
        """Setup and export Pastas model for optimization"""
        # setup parameters
        parameters = self.ml.parameters.copy()
        parameters.index = [p.replace("_A", "_g") for p in parameters.index]
        parameters.index.name = "parnames"
        parameters.loc[:, "optimal"] = parameters.loc[:, "initial"]
        self.ml.set_parameter(
            "constant_d",
            pmin=parameters.at["constant_d", "initial"] - 10,
            pmax=parameters.at["constant_d", "initial"] + 10,
        )

        par_sel = parameters.loc[:, ["optimal"]]
        par_sel.to_csv(self.model_ws / "parameters_sel.csv")
        copy_file(self.model_ws / "parameters_sel.csv", self.temp_ws)
        self.par_sel = par_sel

        # observations
        observations = self.ml.observations()
        observations.name = "Observations"
        observations.to_csv(self.model_ws / "simulation.csv")
        copy_file(self.model_ws / "parameters_sel.csv", self.temp_ws)
        self.observations = observations

        # model
        self.ml.to_file(self.model_ws / "model.pas")

    def setup_files(self):
        """Setup PEST structure for optimization"""
        # parameters
        self.pf.add_parameters(
            self.model_ws / "parameters_sel.csv",
            index_cols=[self.par_sel.index.name],
            use_cols=self.par_sel.columns.to_list(),
            par_type="grid",
            par_style="d",
        )
        # observations and simulation
        self.pf.add_observations(
            "simulation.csv",
            index_cols=[self.observations.index.name],
            use_cols=[self.observations.name],
        )

        # python scripts to run
        self.pf.add_py_function(
            Path(__file__).parent / "_run_pastas_model.py", "run()", is_pre_cmd=None
        )
        self.pf.mod_py_cmds.append("run()")

        # create control file
        pst = self.pf.build_pst()
        pst.parameter_data.loc[:, ["parlbnd", "parubnd"]] = self.ml.parameters.loc[
            :, ["pmin", "pmax"]
        ].values  # parameter bounds
        pst.control_data.noptmax = 100  # optimization runs
        pst.write(self.pf.new_d / "pest.pst", version=2)

    def run(self):
        copy_file(self.exe_name, self.temp_ws)
        pyemu.os_utils.run(f"{self.exe_name.name} pest.pst", cwd=self.pf.new_d)

    def obj_func(self, **kwargs) -> float:
        """dummy objective function"""
        return 0.0


class PestGlmSolver(PestSolver):
    """PESTPP-GLM (Gauss-Levenberg-Marquardt) solver"""

    def __init__(
        self,
        exe_name: Union[str, Path] = "pestpp-glm",
        model_ws: Union[str, Path] = Path("model"),
        temp_ws: Union[str, Path] = Path("temp"),
        pcov: Optional[DataFrame] = None,
        nfev: Optional[int] = None,
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

    def solve(self, **kwargs) -> Tuple[bool, np.ndarray, np.ndarray]:
        self.setup_model()
        self.setup_files()
        self.run()

        # optimal paramters
        ipar = pd.read_csv(self.temp_ws / "pest.ipar", index_col=0).transpose()
        ipar.index = self.ml.parameters.index
        optimal = ipar.iloc[:, -1].values

        # covariance
        # pcov = pd.read_csv(
        #     self.temp_ws / f"pest.{ipar.columns[-1]}.post.cov",
        #     sep="\s+",
        #     skiprows=[0],
        #     nrows=4,
        #     header=None,
        # )
        # self.pcov = pcov

        # dummy return values
        success = True  # always :)
        stderr = np.zeros_like(optimal)  # replace by good covariance later
        return success, optimal, stderr


class PestIemSolver(PestSolver):
    """PESTPP-IEM (Iterative Ensemble Smoother) solver"""

    def __init__(
        self,
        exe_name: Union[str, Path] = "pestpp-iem",
        model_ws: Union[str, Path] = Path("model"),
        temp_ws: Union[str, Path] = Path("temp"),
        pcov: Optional[DataFrame] = None,
        nfev: Optional[int] = None,
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

    def solve(self, **kwargs) -> None:
        # does not work somehow
        self.setup_model()
        self.setup_files()
        self.run()

        # return success, optimal, stderr
