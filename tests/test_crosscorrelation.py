import numpy as np
import pandas as pd
import pastas as ps
import pytest

import pastas_plugins.cross_correlation.cross_correlation as ppcc

sdate = pd.Timestamp("2022-01-01")
x = pd.Series(
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    index=pd.date_range(start=sdate, periods=6, freq="D"),
)
y = pd.Series(
    [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
    index=pd.date_range(start=sdate, periods=6, freq="D"),
)
cc = np.array([1.0, 0.6, 0.08571429, -0.54285714, -1.28571429, -2.14285714])
gpar = np.array([3.0, 1.2, 5.0])
rfunc = ps.Gamma()
ccf = pd.Series(ps.Gamma().block(gpar))


def test_ccf():
    nlags = 3
    result = ppcc.ccf(x, y, nlags=nlags)
    assert isinstance(result, pd.Series)
    assert len(result) == nlags
    assert np.allclose(cc[:nlags], result.values)


def test_ccf_alpha():
    alpha = 0.05
    result = ppcc.ccf(x, y, alpha=alpha)
    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) == 3
    assert result.columns[1] == f"CI {alpha / 2}"
    assert result.columns[2] == f"CI {1 - alpha / 2}"


def test_ccf_invalid_length():
    with pytest.raises(AssertionError):
        ppcc.ccf(x.iloc[:-1], y)


def test_ccf_invalid_frequency():
    x_copy = x.copy()
    x_idx = x_copy.index.values
    x_idx[0] = sdate - pd.Timedelta(days=1)
    x_copy.index = x_idx
    with pytest.raises(ValueError):
        ppcc.ccf(x_copy, y)


def test_prewhiten():
    ar = 2
    pwx = ppcc.prewhiten(x, ar=ar)
    assert isinstance(pwx, pd.Series)
    assert len(pwx) == len(x) - ar


def test_prewhiten_with_y():
    ar = 2
    pwx, pwy = ppcc.prewhiten(x, y, ar=ar)
    assert isinstance(pwx, pd.Series)
    assert len(pwx) == len(x) - ar
    assert isinstance(pwy, pd.Series)
    assert len(pwy) == len(y) - ar


def test_prewhiten_arima():
    ar = 4
    pwx = ppcc.prewhiten(x, ar=ar, arima=True)
    assert isinstance(pwx, pd.Series)
    assert len(pwx) == len(x) - ar


def test_prewhiten_with_y_arima():
    ar = 4
    pwx, pwy = ppcc.prewhiten(x, y, ar=ar, arima=True)
    assert isinstance(pwx, pd.Series)
    assert len(pwx) == len(x) - ar
    assert isinstance(pwy, pd.Series)
    assert len(pwy) == len(y) - ar


def test_fit_response():
    params = ppcc.fit_response(ccf, rfunc)
    assert isinstance(params, np.ndarray)
    assert len(params) == len(rfunc.get_init_parameters(rfunc._name)["initial"])


def test_fit_response_scale_factor():
    scale_factor = 2.0
    params = ppcc.fit_response(ccf, rfunc, scale_factor=scale_factor)
    assert isinstance(params, np.ndarray)
    assert len(params) == len(rfunc.get_init_parameters(rfunc._name)["initial"])


def test_fit_response_dt():
    dt = 0.5
    params = ppcc.fit_response(ccf, rfunc, dt=dt)
    assert isinstance(params, np.ndarray)
    assert len(params) == len(rfunc.get_init_parameters(rfunc._name)["initial"])


def test_fit_response_scale_factor_dt():
    scale_factor = 2.0
    dt = 0.5
    params = ppcc.fit_response(ccf, rfunc, scale_factor=scale_factor, dt=dt)
    assert isinstance(params, np.ndarray)
    assert len(params) == len(rfunc.get_init_parameters(rfunc._name)["initial"])
