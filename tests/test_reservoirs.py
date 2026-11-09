import numpy as np
import pandas as pd
import pastas as ps
import pytest

from pastas_plugins.reservoirs.reservoir import Reservoir1, Reservoir2, ReservoirBase
from pastas_plugins.reservoirs.stressmodels import ReservoirModel


@pytest.fixture
def reservoir_model():
    """Create a Pastas model with a ReservoirModel for testing."""
    rng = np.random.default_rng(42)
    index = pd.date_range(start="2020-01-01", periods=100, freq="D")
    observations = pd.Series(
        10 + np.cumsum(rng.normal(0, 0.05, 100)), index=index, name="head"
    )
    prec = pd.Series(rng.gamma(2, 2, 100), index=index, name="prec")
    evap = pd.Series(rng.gamma(1, 2, 100), index=index, name="evap")
    ml = ps.Model(observations, name="reservoir_ml")
    rm = ReservoirModel(ml, [prec, evap], Reservoir1, name="res", meanhead=10.0)
    return ml, rm


def test_reservoir_model_added_to_model(reservoir_model):
    ml, rm = reservoir_model
    assert "res" in ml.get_stressmodel_names()
    assert ml.stressmodels["res"] is rm
    assert rm.model is ml


def test_reservoir_model_parameters(reservoir_model):
    ml, _ = reservoir_model
    assert len(ml.parameters) == 5  # 4 reservoir parameters + constant_d
    assert "res_S" in ml.parameters.index
    assert "constant_d" in ml.parameters.index


def test_reservoir_model_stresses(reservoir_model):
    _, rm = reservoir_model
    assert len(rm.stresses) == 2
    assert rm.stresses.prec.name == "prec"
    assert rm.stresses.evap.name == "evap"
    assert rm.nsplit == 1


def test_reservoir_model_simulate(reservoir_model):
    ml, _ = reservoir_model
    sim = ml.simulate()
    assert isinstance(sim, pd.Series)
    assert len(sim) > 0
    assert not sim.hasnans


def test_reservoir_model_set_stress(reservoir_model):
    _, rm = reservoir_model
    prec_new = rm.prec.series * 2.0
    rm.set_stress(prec=prec_new)
    assert np.allclose(rm.prec.series.values, prec_new.values)
    # direct property assignment also works
    evap_orig = rm.evap.series.copy()
    rm.evap = evap_orig * 2.0
    assert np.allclose(rm.evap.series.values, evap_orig.values * 2.0)


def test_reservoir_model_get_stress(reservoir_model):
    _, rm = reservoir_model
    stresses = rm.get_stress()
    assert isinstance(stresses, pd.DataFrame)
    assert isinstance(stresses["prec"], pd.Series)
    assert isinstance(stresses["evap"], pd.Series)


def test_reservoir_model_solve(reservoir_model):
    ml, _ = reservoir_model
    ml.solve(report=False)
    assert not ml.parameters["optimal"].isna().any()


def test_reservoir_model_to_dict(reservoir_model):
    _, rm = reservoir_model
    data = rm.to_dict(series=False)
    assert data["class"] == "ReservoirModel"
    assert data["name"] == "res"
    assert data["reservoir"] == "Reservoir1"
    assert data["meanhead"] == 10.0
    assert len(data["stress"]) == 2


def test_reservoir_model_from_string_name():
    rng = np.random.default_rng(1)
    index = pd.date_range(start="2020-01-01", periods=50, freq="D")
    observations = pd.Series(10 + rng.normal(0, 0.1, 50), index=index, name="head")
    prec = pd.Series(rng.gamma(2, 2, 50), index=index, name="prec")
    evap = pd.Series(rng.gamma(1, 2, 50), index=index, name="evap")
    ml = ps.Model(observations)
    rm = ReservoirModel(ml, [prec, evap], "Reservoir2", name="res", meanhead=10.0)
    assert isinstance(rm.reservoir, Reservoir2)


def test_reservoir_model_no_overlap_raises():
    index = pd.date_range(start="2020-01-01", periods=50, freq="D")
    other = pd.date_range(start="2021-01-01", periods=50, freq="D")
    observations = pd.Series(10 + np.random.randn(50), index=index, name="head")
    prec = pd.Series(np.random.rand(50), index=index, name="prec")
    evap = pd.Series(np.random.rand(50), index=other, name="evap")
    ml = ps.Model(observations)
    with pytest.raises(ValueError, match="no overlapping time indices"):
        ReservoirModel(ml, [prec, evap], Reservoir1, name="res", meanhead=10.0)


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
