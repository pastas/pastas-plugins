import pandas as pd
import pytest
from pastas_plugins.crosscorrelation import crosscorr as ppccf

def test_ccf_equidistant_equal_length():
    x = pd.Series([1, 2, 3, 4, 5])
    y = pd.Series([2, 4, 6, 8, 10])
    result = ppccf.ccf_equidistant(x, y)
    assert len(result) == len(x)

def test_ccf_equidistant_equidistant_series():
    x = pd.Series([1, 2, 3, 4, 5], index=pd.date_range(start='2022-01-01', periods=5, freq='D'))
    y = pd.Series([2, 4, 6, 8, 10], index=pd.date_range(start='2022-01-01', periods=5, freq='D'))
    result = ppccf.ccf_equidistant(x, y)
    assert len(result) == len(x)

def test_ccf_equidistant_alpha():
    x = pd.Series([1, 2, 3, 4, 5])
    y = pd.Series([2, 4, 6, 8, 10])
    alpha = 0.05
    result = ppccf.ccf_equidistant(x, y, alpha=alpha)
    assert len(result) == len(x)
    assert len(result.columns) == 3
    assert result.columns[1] == f"CI {alpha / 2}"
    assert result.columns[2] == f"CI {1 - alpha / 2}"

def test_ccf_equidistant_invalid_length():
    x = pd.Series([1, 2, 3, 4, 5])
    y = pd.Series([2, 4, 6, 8])
    with pytest.raises(AssertionError):
        ppccf.ccf_equidistant(x, y)

def test_ccf_equidistant_invalid_frequency():
    x = pd.Series([1, 2, 3, 4, 5], index=pd.date_range(start='2022-01-01', periods=5, freq='D'))
    y = pd.Series([2, 4, 6, 8, 10], index=pd.date_range(start='2022-01-01', periods=5, freq='H'))
    with pytest.raises(ValueError):
        ppccf.ccf_equidistant(x, y)