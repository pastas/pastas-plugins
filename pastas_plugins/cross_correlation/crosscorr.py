from typing import Tuple, Union

import numpy as np
import pandas as pd
import pastas as ps
import scipy as sc
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.filters.filtertools import convolution_filter


def ccf(
    x: pd.Series,
    y: pd.Series,
    nlags: Union[int, None] = None,
    adjusted: bool = True,
    alpha: Union[float, None] = None,
) -> Union[pd.Series, pd.DataFrame]:
    """Cross-correlation of two time series.

    Parameters
    ----------
    x : pd.Series
        Time series
    y : pd.Series
        Time series, len(y) should be equal to len(x).
    nlags : int or None, optional
        Number of lags to return cross-correlations for, by default None which
        uses nlags equal to len(x).
    adjusted : bool, optional
        If True, denominators for cross-correlation are len(x)-k, otherwise
        len(x), by default True
    alpha : float or None, optional
        If a float between 0 and 1 is given, the confidence intervals for the
        given level are returned in a DataFrame. For instance if alpha=0.05,
        95% confidence intervals are returned where the standard deviation is
        computed according to 1/sqrt(len(x)).

    Returns
    -------
    pandas Series or DataFrame
    """
    # check if lengths are equal
    assert len(x) == len(y), "Length of series x and y should be equal"
    # check if series are equidistant
    for series in (x, y):
        if pd.infer_freq(series.index) is None:
            msg = (
                "The frequency of the index of time series %s could not be "
                "inferred. Please provide a time series with a equidistant time step."
            )
            raise ValueError(msg % series.name)

    n = len(x)

    xbar = x - x.mean()
    ybar = y - y.mean()

    d = np.arange(n, 0, -1) if adjusted else n
    cc = sc.signal.correlate(xbar, ybar, mode="full", method="fft")
    cvf = cc[n - 1 :] / (np.std(x) * np.std(y) * d)

    nlags = n if nlags is None else nlags
    index = pd.Index(np.arange(nlags), name="Lags")
    ret = cvf[:nlags]

    if alpha is not None:
        interval = sc.stats.norm.ppf(1.0 - alpha / 2.0) / np.sqrt(n)
        crosscorr = pd.DataFrame(
            data=np.vstack([ret, ret - interval, ret + interval]).T,
            index=index,
            columns=["Cross-correlation", f"CI {alpha / 2}", f"CI {1 - alpha / 2}"],
        )
    else:
        crosscorr = pd.Series(
            data=ret,
            index=index,
            name="Cross-correlation",
        )
    return crosscorr


def prewhiten(
    x: pd.Series, y: Union[pd.Series, None] = None, ar: int = 20, arima: bool = False
) -> Union[pd.Series, Tuple[pd.Series]]:
    """Prewhiten time series using AR(ar) model.

    An AR(ar) model is fitted on time series x. The goal is to obtain residuals that
    adhere to a white noise process. Next, the AR(ar) model is applied to time series Y.

    Note
    ----
    If prewhitened time series for x still shows significant autocorrelation or partial
    autocorrelation, try increasing the number of autoregressive parameters.

    Parameters
    ----------
    x : pd.Series
        time series on which AR(ar) model will be fitted
    y : pd.Series, optional
        time series that will be filtered using the AR(ar) model fitted on x
    ar : int, optional
        number of autoregressive parameters (sometimes called `p`), by default 20
    arima: bool, optional
        use an ARIMA(ar,0,0) model instead of an AR(ar) model, by default False
        which causes a significant speedup at the cost of a very small accuracy
        penalty

    Returns
    -------
    pwx : pd.Series
        prewhitened time series for x (should no longer show significant
        autocorrelation or partial autocorrelation)
    pwy : pd.Series, optional
        prewhitened time series for y, if y is provided
    """

    # fit AR model on x
    if arima:
        ml = ARIMA(x.values, order=(ar, 0, 0), trend="c").fit()
    else:
        ml = AutoReg(x.values, lags=ar, trend="c").fit()

    # get model filtered model residuals
    residuals = ml.resid[ar:] if arima else ml.resid
    pwx = pd.Series(residuals, index=x.index[ar:])

    if y is not None:
        # apply same filter on y
        arparams = ml.arparams if arima else ml.params[1:]
        filt = np.append(1.0, -arparams)
        pwy = convolution_filter(y.values, filt=filt, nsides=1)
        pwy = pd.Series(data=pwy[ar:], index=y.index[ar:])
        return pwx, pwy
    else:
        return pwx


def fit_response(
    ccf: pd.Series,
    rfunc: ps.typing.RFunc,
    scale_factor: float = 1.0,
    dt: float = 1.0,
) -> np.ndarray[float]:
    """Fit the response function to the cross-correlation function using least
    squares optimization.

    Parameters:
    -----------
    ccf : pd.Series
        The cross-correlation function.
    rfunc : ps.typing.RFunc
        The response function to fit on the impulse response.
    scale_factor : float, optional
        Scale factor applied to the cross-correlation function to obtain the
        impulse response, by default 1.0.
    dt : float, optional
        Time step of the response function, by default 1.0.

    Returns:
    --------
    np.ndarray[float]
        The optimized parameters of the response function.

    """

    def obj_func(p):
        """Objective function for least squares optimization."""
        impulse_response = (ccf * scale_factor).values
        blockr = rfunc.block(p, dt=dt, cutoff=rfunc.cutoff)

        # make sure length of residuals is constant
        if len(blockr) > len(impulse_response):
            blockr = blockr[: len(impulse_response)]
        elif len(blockr) < len(impulse_response):
            blockr = np.append(blockr, np.zeros(len(impulse_response) - len(blockr)))

        return impulse_response - blockr

    params = rfunc.get_init_parameters(rfunc._name)
    pini = params["initial"].values
    bounds = (
        params["pmin"].fillna(-np.inf).values,
        params["pmax"].fillna(np.inf).values,
    )

    res = sc.optimize.least_squares(obj_func, x0=pini, bounds=bounds)
    return res.x
