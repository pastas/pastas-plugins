from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from packaging.version import parse as parse_version
from pastas import __version__ as pastas_version
from pastas.typing import ArrayLike, Model
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize._numdiff import approx_derivative
from tqdm.auto import tqdm, trange

PASTAS_VERSION = parse_version(parse_version(pastas_version).base_version)


class DataWorth:
    def __init__(
        self,
        ml: Model,
        J: ArrayLike | None = None,
        objfun_target: Literal["noise", "residuals"] = "noise",
    ):
        """Class for computing data worth of observations in a Pastas model.

        Parameters
        ----------
        ml: pastas.Model
            Pastas model for which to compute data worth.
        J: np.ndarray, optional
            Jacobian matrix (N_obs, N_params) to use for data
            worth analysis. If None, the Jacobian from the model's
            solver result is used.
        objfun_target: str, optional
            Objective function target, either "noise" or "residuals".
            This determines how the observation noise covariance matrix
            is computed. If "noise", the variance of the noise process
            is used, and observations are assumed to be independent. If
            "residuals", the variance of the residuals is used, and an
            AR(1) correlation structure can be included to account for
            temporal correlation between observations. Defaults to "noise".
        """
        self.ml = ml
        if J is not None:
            self.J0 = J

        else:
            self.J0 = ml.solver.result.jac

        self.objfun_target = objfun_target

    def observation_noise_covariance(
        self,
        obs: pd.Series | None = None,
        var: float | None = None,
        obs_std: float = 1e-3,
        noise_alpha: float | None = None,
        objfun_target: Literal["noise", "residuals"] | None = None,
    ) -> ArrayLike:
        """Get observation noise covariance matrix.

        When noise_alpha is supplied this is used to quantify the covariance between
        observation noise on the off-diagonals. This is needed when observations are
        not independent and the Jacobian used for the data worth analysis is computed
        in residual space (i.e. not in noise space).

        Parameters
        ----------
        obs: pd.Series
            Observations for which to compute the noise covariance matrix, if None, use
            ml.observations().
        var: float
            Variance (sigma^2) of the noise (or residual) process, if None, computes
            the variance of the weighted noise used in the pastas calibration.
        obs_std: float
            Standard deviation of the pure measurement noise (nugget).
            Defaults to a tiny value for stability.
        noise_alpha: float
            Decay parameter for the AR(1) noise model, in days. Higher values indicate
            stronger temporal correlation. If None, no temporal correlation is assumed,
            i.e. all observations are considered to be independent.

        Returns
        -------
        C_eps: np.ndarray
            Observation noise covariance matrix (N_obs, N_obs)
        """
        if obs is None:
            obs = self.ml.observations()

        if objfun_target is None:
            objfun_target = self.objfun_target

        if var is None:
            if objfun_target == "noise":
                noise_weights = (
                    self.ml.noise_weights()
                    if PASTAS_VERSION < parse_version("2.0.0")
                    else self.ml._noise_weights()
                )
                var = np.var(self.ml.noise() * noise_weights)
            elif objfun_target == "residuals":
                var = np.var(self.ml.residuals())
            else:
                raise ValueError(f"Unknown objfun_target: {objfun_target}")

        n_obs = obs.index.size

        if objfun_target == "residuals" and noise_alpha is not None:
            # Time difference calculation (in days)
            t_days = obs.index.values.astype("datetime64[ns]").astype(float) / (
                1e9 * 3600 * 24
            )

            # Broadcasting: (N, 1) - (1, N) creates the (N, N) matrix of differences
            dt = np.abs(t_days[:, None] - t_days[None, :])

            # Build AR(1) Matrix
            C_eps = var * np.exp(-dt / noise_alpha)

            # 5. Add the "nugget" (observation_std) to the diagonal only
            # This represents pure measurement error and ensures invertibility
            if obs_std > 0:
                C_eps += np.eye(n_obs) * (obs_std**2)

        else:
            # Diagonal matrix using noise variance, every observation is independent.
            C_eps = np.eye(n_obs) * (var + obs_std**2)

        return C_eps

    @staticmethod
    def fisher_information_matrix(J: ArrayLike, C_eps_inv: ArrayLike) -> ArrayLike:
        """Computes the Fisher information matrix.

        J : np.ndarray
            Jacobian matrix (N_obs, N_params)
        C_eps_inv : np.ndarray
            Inverse of the measurement noise covariance matrix (N_obs, N_obs)

        Returns
        -------
        FIM : np.ndarray
            Fisher information matrix (N_params, N_params)
        """
        return J.T @ C_eps_inv @ J

    @staticmethod
    def compute_covariance(
        J: ArrayLike, C_eps: ArrayLike, mask: ArrayLike | None = None
    ):
        """Computes the posterior parameter covariance matrix (Cp).

        J : np.ndarray
            Jacobian matrix (N_obs, N_params)
        C_eps_inv : np.ndarray
            Inverse of the measurement noise covariance matrix (N_obs, N_obs)
        mask : np.ndarray, optional
            Boolean array of length N_obs indicating which observations to include in
            the analysis. If None, all observations are included.

        Returns
        -------
        Cp : np.ndarray
            Posterior parameter covariance matrix (N_params, N_params)
        """
        if mask is not None:
            Jm = J[mask]
            C_sub = C_eps[np.ix_(mask, mask)]
        else:
            Jm = J
            C_sub = C_eps
        # Use Cholesky factorisation to solve C_sub^{-1} J without forming the
        # explicit inverse.  This is numerically stable even when C_sub is nearly
        # singular (large noise_alpha / high temporal correlation).
        c, low = cho_factor(C_sub)
        CinvJ = cho_solve((c, low), Jm)
        FIM = Jm.T @ CinvJ
        try:
            # Prefer a factorization-based inverse; fallback to pseudo-inverse for
            # rank-deficient FIMs (e.g. aggressive thinning / collinear sensitivities).
            cfim, lowfim = cho_factor(FIM)
            Cp = cho_solve((cfim, lowfim), np.eye(FIM.shape[0]))
        except np.linalg.LinAlgError:
            Cp = np.linalg.pinv(FIM, hermitian=True)
        return 0.5 * (Cp + Cp.T)

    def data_worth(
        self,
        J: ArrayLike | None = None,
        C_eps: ArrayLike | None = None,
        mask: ArrayLike | None = None,
    ) -> tuple[float, np.ndarray]:
        if J is None:
            J = self.J0
        if C_eps is None:
            C_eps = self.observation_noise_covariance()

        C_k = self.compute_covariance(J, C_eps, mask=mask)

        # worth overall, log-det of covariance when leaving out masked observations
        _, logdet_k = np.linalg.slogdet(C_k)

        # worth per parameter, e.g. variance of parameters when leaving out masked
        # observations
        var_params_k = np.diag(C_k)
        return logdet_k, var_params_k

    def data_worth_per_observation(self, as_percentage: bool = False) -> pd.DataFrame:
        """Compute data worth of each observation.

        Computes the data worth of each observation by quantifying the change in
        parameter covariance when leaving out that observation. Overall data worth is
        quantified by the change in log-determinant of the parameter covariance, while
        relative data worth per parameter is quantified by the change in variance of
        each parameter when leaving out the observation, relative to the variance when
        all observations are included.

        Returns
        -------
        dw : pd.DataFrame
            DataFrame of length n_obs with data worth. First column contains overall
            data worth (change in log-determinant). Subsequent columsn contain relative
            data worth per parameter (change in variance relative to full model).
        as_percentage : bool
            If True, convert the data worth metrics to percentage increase in standard
            error. For the overall metric (log-det): r = exp(Δlogdet / 2N), expressed as
            (r-1)*100 %. For per-parameter: r_j = sqrt(1 + δσ²_j), expressed as (r_j-1)*100 %.
            A value of 0 % means the observation had no effect on the parameter uncertainty;
            positive values indicate how much the std error would inflate if that observation
            were removed.
        """
        J = self.J0
        n_obs, n_params = J.shape

        # observations
        obs = self.ml.observations()

        # get observation noise covariance matrix
        C_eps = self.observation_noise_covariance()

        # full model covariance
        logdet_full, base_parameter_variance = self.data_worth(J, C_eps)

        worth_overall = np.zeros(n_obs)
        relative_worth_per_param = np.zeros((n_obs, n_params))

        # leave out one analysis
        for k in trange(n_obs, desc="Data Worth (existing obs)"):
            mask = np.ones(n_obs, dtype=bool)
            mask[k] = False
            logdet_k, var_params_k = self.data_worth(J, C_eps, mask)
            # worth overall, change in log-det of covariance when
            # leaving out observation k
            worth_overall[k] = logdet_k - logdet_full
            # worth per parameter, e.g. change in variance of parameter when
            # leaving out observation k
            relative_worth_per_param[k, :] = (
                var_params_k - base_parameter_variance
            ) / base_parameter_variance

        name = "Increase in gen. std. err. (%)" if as_percentage else "Δlogdet"
        worth_overall = pd.Series(worth_overall, index=obs.index, name=name)
        columns = [
            f"Δσ {ipar} (%)" if as_percentage else rf"Δσ$^2$ {ipar}"
            for ipar in self.ml.parameters.index
            if self.ml.parameters.loc[ipar, "vary"]
        ]
        relative_worth_per_param = pd.DataFrame(
            relative_worth_per_param, index=obs.index, columns=columns
        )

        if as_percentage:
            # Convert raw data worth metrics to percentage increase in standard error.
            # For the overall metric (log-det): r = exp(Δlogdet / 2N), expressed as (r-1)*100 %.
            # For per-parameter: r_j = sqrt(1 + δσ²_j), expressed as (r_j-1)*100 %.
            # A value of 0 % means the observation had no effect; positive values indicate
            # how much the std error would inflate if that observation were removed.

            N = self.ml.parameters.index.size
            worth_overall = (np.exp(worth_overall / (2 * N)) - 1) * 100
            relative_worth_per_param = (
                (1 + relative_worth_per_param).pow(0.5) - 1
            ) * 100

        df = pd.concat([worth_overall, relative_worth_per_param], axis=1)

        return df

    def data_worth_thinning(
        self, thinning_intervals: list[int]
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Compute data worth of thinning observations.

        Computes the data worth of thinning observations by quantifying the change in
        parameter covariance when including only a subset of the observations. Overall
        data worth is quantified by the change in log-determinant of the parameter
        covariance, while relative data worth per parameter is quantified by the change
        in variance of each parameter when including only the thinned observations,
        relative to the variance when all observations are included.

        Parameters
        ----------
        thinning_intervals: list of int
            List of thinning intervals (in days) for which to compute data worth. For
            example, a value of 7 means that only every 7th observation is included in
            the analysis.

        Returns
        -------
        dw_thinning: pd.Series
            Series containing overall data worth (change in log-determinant) for each
            thinning interval, indexed by the number of observations per year.
        dw_thinning_per_param: pd.DataFrame
            DataFrame containing relative data worth per parameter (change in variance
            relative to full model) for each thinning interval, indexed by the number
            of observations per year and with columns for each parameter.
        """
        dw_thinning = pd.Series(index=thinning_intervals, data=np.nan)
        dw_thinning_per_param = pd.DataFrame(
            index=thinning_intervals, columns=self.ml.parameters.index, data=np.nan
        )

        C_eps = self.observation_noise_covariance()
        J = self.J0
        logdet_full, var_params_full = self.data_worth(J=J, C_eps=C_eps)

        for i in thinning_intervals:
            mask = np.zeros(self.ml.observations().index.size, dtype=bool)
            mask[::i] = True

            logdet_k, var_params_k = self.data_worth(J=J, C_eps=C_eps, mask=mask)

            dw_thinning.loc[i] = logdet_k - logdet_full
            dw_thinning_per_param.loc[i] = (
                var_params_k - var_params_full
            ) / var_params_full
        return dw_thinning, dw_thinning_per_param

    def recompute_jacobian(
        self,
        new_observations: pd.Series,
        objfun_target: Literal["noise", "residuals"] | None = None,
        method: Literal["2-point", "3-point"] = "2-point",
        **kwargs,
    ) -> ArrayLike:
        """Recompute the Jacobian matrix for a given set of new observations.

        Parameters
        ----------
        observations: pd.Series
            Series of observations for which to recompute the Jacobian.
        objfun_target: str, optional
            Objective function target, either "noise" or "residuals". This determines
            how the objective function is defined for the numerical differentiation.
        method: str, optional
            Method to use for numerical differentiation. Passed to
            `scipy.optimize.approx_derivative`.
            TODO: assert from ml.solver.jac what is used for calibration and
            use the same method here by default. Can be done in Pastas 2.0.
        **kwargs:
            Additional keyword arguments to pass to scipy.optimize.approx_derivative.

        Returns
        -------
        J: np.ndarray
            Jacobian matrix (N_obs, N_params) for the given observations.
        """

        obs_calib = self.ml.observations()
        obs = pd.concat([obs_calib, new_observations], axis=0).sort_index()

        # drop duplicates
        obs = obs.loc[~obs.index.duplicated(keep="first")]

        # define objective function
        def fobj(p, ml, obs, objfun_target=None):
            # TODO: replace by pastas.solver.objerctive_function.misfit
            p_full = ml.parameters["initial"].to_numpy(copy=True)
            p_full[ml.parameters["vary"].to_numpy()] = p
            sim = ml.simulate(p=p_full)
            # Use interpolation to handle obs timestamps not on the simulation grid,
            # mirroring the approach used in ml.residuals().
            sim_at_obs = np.interp(obs.index.asi8, sim.index.asi8, sim.values)
            residuals = pd.Series(obs.values - sim_at_obs, index=obs.index)
            if ml.noisemodel is not None and objfun_target != "residuals":
                noise_weights = ml.noisemodel.weights(
                    residuals, p[-ml.noisemodel.nparam :]
                )
                noise = ml.noisemodel.simulate(residuals, p=p[-ml.noisemodel.nparam :])
                # Return numpy array so approx_fprime doesn't align on pandas index
                return (noise * noise_weights).values
            else:
                return residuals.values

        param_names = (
            self.ml.parameters["vary"].index[self.ml.parameters["vary"]].to_list()
        )
        if objfun_target == "residuals":
            param_names = [par for par in param_names if par != "noise_alpha"]
        Jadj = approx_derivative(
            fobj,
            self.ml.parameters.loc[param_names, "optimal"].values,
            method=method,
            rel_step=None,
            abs_step=None,
            bounds=(
                self.ml.parameters.loc[param_names, "pmin"].values,
                self.ml.parameters.loc[param_names, "pmax"].values,
            ),
            args=(self.ml, obs),
            kwargs={"objfun_target": objfun_target},
            **kwargs,
        )

        return Jadj

    def data_worth_per_added_observation(
        self,
        new_observations: pd.Series,
        objfun_target: Literal["noise", "residuals"] | None = None,
        noise_alpha: float | None = None,
        as_percentage: bool = False,
    ) -> pd.DataFrame:
        """Compute data worth of new observations.

        Computes the data worth of new observations by quantifying the change in
        parameter covariance when including those observations. Overall data worth is
        quantified by the change in log-determinant of the parameter covariance, while
        relative data worth per parameter is quantified by the change in variance of
        each parameter when including the new observations, relative to the variance
        when only existing observations are included.

        Note
        ----
        The objective function either uses the noise or the residuals. It might be more
        appropriate here to use the residuals, since this allows us to include the
        correlation between observations, penalizing new observations that are close in
        time to existing observations. When computing data worth in noise space, we
        assume that all observations are independent, which can lead to overestimation
        of the worth of new observations that are close in time to existing
        observations.

        Parameters
        ----------
        new_observations: pd.Series
            Series of new observations for which to compute data worth.

        Returns
        -------
        dw_new: pd.DataFrame
            DataFrame of length n_new_obs with data worth for new observations. First
            column contains overall data worth (change in log-determinant). Subsequent
            columns contain relative data worth per parameter (change in variance
            relative to full model).
        """
        if objfun_target is None:
            objfun_target = self.objfun_target

        # get existing observations and combine with new observations
        obs_calib = self.ml.observations()
        # mask out existing observations from new_observations to avoid duplicates
        if obs_calib.index.intersection(new_observations.index).size > 0:
            new_observations = new_observations.loc[
                ~new_observations.index.isin(obs_calib.index)
            ]

        obs = pd.concat([obs_calib, new_observations], axis=0).sort_index()
        mask = obs.index.isin(obs_calib.index)
        new_obs_idx = np.nonzero(~mask)[0]

        # recompute Jacobian and observation noise covariance
        # for combined set of observations
        Jadj = self.recompute_jacobian(new_observations, objfun_target)
        if objfun_target == "noise":
            C_eps = self.observation_noise_covariance(
                obs=obs, objfun_target=objfun_target
            )
        else:
            if noise_alpha is None:
                noise_alpha = self.ml.parameters.loc["noise_alpha", "optimal"]
            # include correlation in off-diagonals when objfun_target is "residuals"
            C_eps = self.observation_noise_covariance(
                obs=obs, noise_alpha=noise_alpha, objfun_target=objfun_target
            )

        # full model covariance, use only observations used in calibration
        logdet_base, var_param_base = self.data_worth(Jadj, C_eps, mask=mask)

        # compute data worth for new observations
        n_obs, n_params = Jadj.shape
        worth_overall = np.zeros(n_obs)
        relative_worth_per_param = np.zeros((n_obs, n_params))

        for k in tqdm(new_obs_idx, desc="Data Worth (new obs)"):
            # use mask to include calibration observations + 1 new observation k
            mask_k = mask.copy()
            mask_k[k] = True

            logdet_k, var_params_k = self.data_worth(Jadj, C_eps, mask=mask_k)

            worth_overall[k] = logdet_base - logdet_k

            relative_worth_per_param[k, :] = (
                var_param_base - var_params_k
            ) / var_param_base

        name = "Reduction std. err. (%)" if as_percentage else "Δlogdet"
        worth_overall = pd.Series(worth_overall, index=obs.index, name=name)
        if objfun_target == "residuals":
            parameter_names = [
                ipar for ipar in self.ml.parameters.index if ipar != "noise_alpha"
            ]
        else:
            parameter_names = self.ml.parameters.index
        columns = [
            rf"$P_σ$ {ipar} (%)" if as_percentage else rf"Δσ$^2$ {ipar}"
            for ipar in parameter_names
        ]
        relative_worth_per_param = pd.DataFrame(
            relative_worth_per_param, index=obs.index, columns=columns
        )
        if as_percentage:
            worth_overall = (1 - np.exp(-worth_overall / (2 * n_params))).multiply(100)
            relative_worth_per_param = (
                1 - (1 - relative_worth_per_param).pow(0.5)
            ) * 100

        df = pd.concat([worth_overall, relative_worth_per_param], axis=1).iloc[
            new_obs_idx
        ]

        return df

    def data_worth_new_observations(
        self,
        new_observations: pd.Series,
        objfun_target: Literal["noise", "residuals"] | None = None,
        noise_alpha: float | None = None,
        as_percentage: bool = False,
    ) -> tuple[float, pd.DataFrame]:
        """Compute data worth of adding new observations.

        Computes the data worth of adding new observations by quantifying the change in
        parameter covariance when including those observations. The Jacobian is
        recomputed to include new observations, assuming observation noise variance
        remains constant. Overall data worth is quantified by the change in
        log-determinant of the parameter covariance, while relative data worth per
        parameter is quantified by the change in variance of each parameter when
        including the new observations, relative to the variance when only existing
        observations are included.

        Parameters
        ----------
        new_observations: pd.Series
            Series of new observations for which to compute data worth.
        objfun_target: str, optional
            Objective function target, either "noise" or "residuals".
        noise_alpha: float, optional
            Decay parameter for the AR(1) correlation model (days). Only used when
            ``objfun_target="residuals"``. If None, the fitted model value is used.
        as_percentage: bool, optional
            If True, the data worth is returned as a percentage of the original variance.

        Returns
        -------
        worth_overall: float
            Overall data worth of adding the new observations, quantified by the change
            in log-determinant of the parameter covariance.
        relative_worth_per_param: pd.DataFrame
            Relative data worth per parameter of adding the new observations, quantified
            by the change in variance of each parameter relative to the variance when
            only existing observations are included.

        """
        if objfun_target is None:
            objfun_target = self.objfun_target

        # get existing observations and combine with new observations
        obs_calib = self.ml.observations()
        # mask out existing observations from new_observations to avoid duplicates
        if obs_calib.index.intersection(new_observations.index).size > 0:
            new_observations = new_observations.loc[
                ~new_observations.index.isin(obs_calib.index)
            ]

        obs = pd.concat([obs_calib, new_observations], axis=0).sort_index()
        mask = obs.index.isin(obs_calib.index)

        # recompute Jacobian and observation noise covariance
        # for combined set of observations
        Jadj = self.recompute_jacobian(new_observations, objfun_target)
        if objfun_target == "noise":
            C_eps = self.observation_noise_covariance(
                obs=obs, objfun_target=objfun_target
            )
        else:
            if noise_alpha is None:
                noise_alpha = self.ml.parameters.loc["noise_alpha", "optimal"]
            # include correlation in off-diagonals when objfun_target is "residuals"
            C_eps = self.observation_noise_covariance(
                obs=obs, noise_alpha=noise_alpha, objfun_target=objfun_target
            )

        # base model covariance, use only observations used in calibration
        logdet_base, var_param_base = self.data_worth(Jadj, C_eps, mask=mask)

        # compute data worth for new observations
        logdet_new, var_param_new = self.data_worth(Jadj, C_eps)

        worth_overall = logdet_base - logdet_new
        relative_worth_per_param = (var_param_base - var_param_new) / var_param_base
        if as_percentage:
            worth_overall = (
                1 - np.exp(-worth_overall / (2 * self.ml.parameters.index.size - 1))
            ) * 100
            relative_worth_per_param = (
                1 - np.power(1 - relative_worth_per_param, 0.5)
            ) * 100

        relative_worth_per_param = pd.DataFrame(
            relative_worth_per_param,
            index=self.ml.parameters.index,
            columns=["Relative Worth (%)"] if as_percentage else ["Relative Worth"],
        )
        return worth_overall, relative_worth_per_param


def plot_data_worth_series(
    observations: pd.Series,
    data_worth: pd.DataFrame,
    compute_sizes_per_plot: bool = False,
    **kwargs,
) -> plt.Axes:
    """Plot data worth as a function of time.

    Parameters
    ----------
    observations : pd.Series
        Series of observations, with datetime index.
    data_worth : pd.DataFrame
        DataFrame containing data worth values, with datetime index and columns for
        overall and per-parameter worth.
    compute_sizes_per_plot : bool
        Whether to compute the size of the scatter points separately for each subplot
        based on min/max per column (True) or to use the same scaling across all
        subplots (False) using the absolute min/max.
    kwargs : dict
        Additional keyword arguments to pass to the scatter plot.


    Returns
    -------
    axes: np.ndarray
        Array of matplotlib axes objects for the subplots.
    """
    assert data_worth.index.difference(observations.index).size == 0, (
        "Observations must contain all observations included in data worth dataframe."
    )
    fig, axes = plt.subplots(
        data_worth.columns.size,
        1,
        figsize=(10, 2 * data_worth.columns.size),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    size_min = 1.0
    size_span = 100.0
    gmin = data_worth.min().min()
    grange = data_worth.max().max() - gmin
    if np.isclose(grange, 0.0):
        s_abs = pd.DataFrame(
            size_min + 0.5 * size_span,
            index=data_worth.index,
            columns=data_worth.columns,
        )
    else:
        s_abs = size_min + size_span * (data_worth - gmin) / grange

    for i, col in enumerate(data_worth.columns):
        if compute_sizes_per_plot:
            cmin = data_worth[col].min()
            crange = data_worth[col].max() - cmin
            if np.isclose(crange, 0.0):
                s = pd.Series(size_min + 0.5 * size_span, index=data_worth.index)
            else:
                s = size_min + size_span * (data_worth[col] - cmin) / crange
        else:
            s = s_abs[col]
        iax = axes.flat[i]
        iax.plot(observations.index, observations.values, lw=0.5, color="k")
        sc = axes.flat[i].scatter(
            data_worth.index,
            observations.loc[data_worth.index].values,
            s=s,
            c=data_worth[col],
            marker="o",
            cmap="RdYlBu_r",
            edgecolor="k",
            label=col,
            zorder=5,
            **kwargs,
        )
        iax.grid(True)
        fig.colorbar(sc, ax=iax, label=col)
    return axes


def plot_data_worth_heatmap(data_worth_series: pd.Series, **kwargs) -> plt.Axes:
    """Plot data worth as a heatmap over the year.

    Plots years on the y-axis and day of year on the x-axis, with color representing
    data worth. This allows for visual identification of seasonal patterns in data
    worth, such as certain times of the year consistently having higher or lower worth.

    Parameters
    ----------
    data_worth_series: pd.Series
        Series containing data worth values, with datetime index.
    kwargs: dict
        Additional keyword arguments to pass to the matshow function for the heatmap.

    Returns
    -------
    ax: matplotlib.axes.Axes
        Matplotlib axes object containing the heatmap.
    """
    dwi = data_worth_series.to_frame().copy()
    dwi["year"] = dwi.index.year
    dwi["dayofyear"] = dwi.index.dayofyear
    dwi = dwi.pivot(index="year", columns="dayofyear", values=dwi.columns[0])
    dwi = dwi.reindex(np.arange(366), axis=1)
    # Fill missing values by forward and backward filling, to create a continuous
    # heatmap.
    dwi = dwi.ffill(axis=1).bfill(axis=1)
    dwi.iloc[0, : data_worth_series.index[0].dayofyear] = np.nan
    dwi.iloc[-1, data_worth_series.index[-1].dayofyear :] = np.nan

    month_starts = pd.date_range("2001-01-01", periods=12, freq="MS")
    month_ticks = month_starts.dayofyear - 1  # 0-indexed columns
    month_labels = month_starts.strftime("%b")

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.matshow(dwi, aspect="auto", cmap=kwargs.pop("cmap", "RdYlBu_r"), **kwargs)
    ax.set_yticks(range(dwi.index.size))
    ax.set_yticklabels(dwi.index)
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(data_worth_series.name)
    ax.set_title("Data Worth Heatmap")
    return ax
