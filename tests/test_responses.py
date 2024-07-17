import numpy as np

from pastas_plugins.responses.rfunc import Theis


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
