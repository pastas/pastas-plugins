import pandas as pd
import pytest

from pastas_plugins.reservoirs.reservoir import Reservoir1, Reservoir2, ReservoirBase


def test_reservoir_base_init():
    reservoir = ReservoirBase()
    assert not reservoir.temp
    assert reservoir.nparam == 0


def test_reservoir_base_get_init_parameters():
    reservoir = ReservoirBase()
    parameters = reservoir.get_init_parameters()
    assert parameters.empty


def test_reservoir_base_simulate():
    reservoir = ReservoirBase()
    prec = [1, 2, 3]
    evap = [0.5, 1.0, 1.5]
    p = [0.1, 0.2, 0.3, 0.4]
    result = reservoir.simulate(prec, evap, p)
    assert result is None


def test_reservoir1_init():
    initialhead = 10
    reservoir = Reservoir1(initialhead)
    assert reservoir.initialhead == initialhead
    assert reservoir.nparam == 4


def test_reservoir1_get_init_parameters():
    reservoir = Reservoir1(10)
    parameters = reservoir.get_init_parameters()
    assert len(parameters) == 4
    assert parameters.loc["reservoir_S"]["initial"] == 0.1
    assert parameters.loc["reservoir_c"]["initial"] == 100
    assert parameters.loc["reservoir_d"]["initial"] == 10
    assert parameters.loc["reservoir_f"]["initial"] == -1.0


def test_reservoir1_simulate():
    reservoir = Reservoir1(10)
    prec = pd.Series([1.0, 2.0, 3.0])
    evap = pd.Series([0.5, 1.0, 1.5])
    p = [0.1, 0.2, 0.3, 0.4]
    result = reservoir.simulate(prec, evap, p)
    assert len(result) == 3
    assert result[0] == pytest.approx(12.3)
    assert result[1] == pytest.approx(-563.7)
    assert result[2] == pytest.approx(27672.3)


def test_reservoir2_init():
    initialhead = 10
    reservoir = Reservoir2(initialhead)
    assert reservoir.initialhead == initialhead
    assert reservoir.nparam == 6


def test_reservoir2_get_init_parameters():
    reservoir = Reservoir2(10)
    parameters = reservoir.get_init_parameters()
    assert len(parameters) == 6
    assert parameters.loc["reservoir_S"]["initial"] == 0.1
    assert parameters.loc["reservoir_c"]["initial"] == 100
    assert parameters.loc["reservoir_d"]["initial"] == 10
    assert parameters.loc["reservoir_f"]["initial"] == -1.0
    assert parameters.loc["reservoir_c2"]["initial"] == 100
    assert parameters.loc["reservoir_deld"]["initial"] == 0.01


def test_reservoir2_simulate():
    reservoir = Reservoir2(10)
    prec = pd.Series([1.0, 2.0, 3.0])
    evap = pd.Series([0.5, 1.0, 1.5])
    p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.01]
    result = reservoir.simulate(prec, evap, p)
    assert len(result) == 3
    assert result[0] == pytest.approx(12.3)
    assert result[1] == pytest.approx(-803.5)
    assert result[2] == pytest.approx(39422.5)
