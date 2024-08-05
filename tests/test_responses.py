import numpy as np
from pandas import DataFrame

from pastas_plugins.responses.rfunc import Edelman, Theis


def test_theis_init():
    theis = Theis()
    assert theis.cutoff == 0.999
    assert theis.nterms == 10


def test_theis_get_init_parameters():
    theis = Theis()
    parameters = theis.get_init_parameters("test")
    assert len(parameters) == 3
    assert parameters.loc["test_A"]["initial"] == 1 / theis.gain_scale_factor
    assert parameters.loc["test_a"]["initial"] == 1e2
    assert parameters.loc["test_b"]["initial"] == 1e-3


def test_theis_get_tmax():
    theis = Theis()
    p = [1, 1, 1]
    tmax = theis.get_tmax(p)
    assert np.isclose(tmax, -p[1] * np.log(1 - theis.cutoff))


def test_theis_gain():
    theis = Theis()
    p = [2, 1, 1]
    assert theis.gain(p) == p[0]


def test_theis_step():
    theis = Theis()
    p = [1, 1, 1]
    dt = 1.0
    cutoff = 0.999
    maxtmax = None
    s = theis.step(p, dt, cutoff, maxtmax)
    assert isinstance(s, np.ndarray)


def test_theis_to_dict():
    theis = Theis()
    data = theis.to_dict()
    assert data["class"] == "Theis"
    assert data["cutoff"] == theis.cutoff
    # assert data["nterms"] == theis.nterms


def test_edelman_init():
    cutoff = 0.999
    rfunc = Edelman(cutoff=cutoff)
    assert rfunc.cutoff == cutoff


def test_edelman_get_init_parameters():
    rfunc = Edelman()
    params = rfunc.get_init_parameters("Edelman")
    assert isinstance(params, DataFrame)
    assert len(params) == 1
    assert params.loc["Edelman_beta"]["initial"] == 1.0


def test_edelman_get_tmax():
    p = np.array([1.0])
    cutoff = 0.999
    rfunc = Edelman(cutoff=cutoff)
    tmax = rfunc.get_tmax(p, cutoff=cutoff)
    assert isinstance(tmax, float)


def test_edelman_gain():
    p = np.array([1.0])
    rfunc = Edelman()
    gain = rfunc.gain(p)
    assert isinstance(gain, float)
    assert gain == 1.0


def test_edelman_impulse():
    t = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0])
    rfunc = Edelman()
    impulse = rfunc.impulse(t, p)
    assert isinstance(impulse, np.ndarray)
    assert len(impulse) == len(t)


def test_edelman_step():
    p = np.array([1.0])
    dt = 1.0
    cutoff = 0.999
    rfunc = Edelman(cutoff=cutoff)
    step = rfunc.step(p, dt=dt, cutoff=cutoff)
    assert isinstance(step, np.ndarray)
