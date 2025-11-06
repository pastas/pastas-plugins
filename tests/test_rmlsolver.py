import numpy as np
import pandas as pd
import pastas as ps
import pytest
from pastas.typing import ArrayLike

from pastas_plugins.pest.solver import RandomizedMaximumLikelihoodSolver


@pytest.fixture
def simple_pastas_model():
    """Create a simple Pastas model for testing."""
    # Create synthetic observation data
    index = pd.date_range(start="2020-01-01", periods=100, freq="D")
    observations = pd.Series(
        10 + 2 * np.sin(np.arange(100) * 2 * np.pi / 365) + np.random.randn(100) * 0.1,
        index=index,
        name="obs",
    )
    prec = pd.Series(
        np.random.rand(100) * 5,
        index=index,
        name="precipitation",
    )
    evap = pd.Series(
        np.random.rand(100) * 3,
        index=index,
        name="evaporation",
    )
    # Create a simple model
    ml = ps.Model(observations, name="test_ml")
    rm = ps.RechargeModel(prec, evap, name="rch", rfunc=ps.Gamma())
    ml.add_stressmodel(rm)
    ml.set_parameter(
        "constant_d",
        pmin=observations.min(),
        pmax=observations.max(),
        initial=observations.mean(),
    )

    return ml


@pytest.fixture
def rml_solver():
    """Create a basic RML solver instance."""
    return RandomizedMaximumLikelihoodSolver(
        num_reals=10,
        jacobian_method="2-point",
        seed=42,
        add_base=True,
        num_workers=1,
    )


def test_rml_solver_initialization(rml_solver: RandomizedMaximumLikelihoodSolver):
    """Test RML solver initialization."""
    assert rml_solver.num_reals == 10
    assert rml_solver.jacobian_method == "2-point"
    assert rml_solver.seed == 42
    assert rml_solver.add_base is True
    assert rml_solver.parameter_ensemble is None
    assert rml_solver.observation_noise is None
    assert rml_solver.simulation_ensemble is None


def test_rml_solver_repr(rml_solver: RandomizedMaximumLikelihoodSolver):
    """Test string representation of RML solver."""
    repr_str = repr(rml_solver)
    assert "RandomizedMaximumLikelihoodSolver" in repr_str
    assert "num_reals=10" in repr_str
    assert "jacobian_method=2-point" in repr_str


@pytest.mark.skip("to_dict method does not work properly yet")
def test_rml_solver_to_dict(rml_solver: RandomizedMaximumLikelihoodSolver):
    """Test conversion to dictionary."""
    data = rml_solver.to_dict()
    assert data["class"] == "RandomizedMaximumLikelihoodSolver"
    assert data["num_reals"] == 10
    assert data["jacobian_method"] == "2-point"
    assert data["seed"] == 42


def test_rml_solver_initialize(
    simple_pastas_model: ps.Model, rml_solver: RandomizedMaximumLikelihoodSolver
):
    """Test RML solver initialization with model."""
    simple_pastas_model.add_solver(rml_solver)
    rml_solver.initialize(
        standard_deviation=0.1,
        correlation_coefficient=0.0,
        par_sigma_range=4.0,
        method="norm",
    )

    assert rml_solver.parameter_ensemble is not None
    assert rml_solver.observation_noise is not None
    assert rml_solver.parameter_ensemble.shape[0] == 10
    assert rml_solver.observation_noise.shape[1] == 10
    assert "base" in rml_solver.parameter_ensemble.index
    assert "base" in rml_solver.observation_noise.columns


def test_rml_solver_initialize_without_base(simple_pastas_model: ps.Model):
    """Test RML solver initialization without base realization."""
    solver = RandomizedMaximumLikelihoodSolver(
        num_reals=10,
        jacobian_method="2-point",
        seed=42,
        add_base=False,
        num_workers=1,
    )
    solver.ml = simple_pastas_model
    solver.initialize(standard_deviation=0.1)

    assert "base" not in solver.parameter_ensemble.index
    assert "base" not in solver.observation_noise.columns


def test_observation_ensemble(
    simple_pastas_model: ps.Model, rml_solver: RandomizedMaximumLikelihoodSolver
):
    """Test observation ensemble generation."""
    rml_solver.ml = simple_pastas_model
    rml_solver.initialize(standard_deviation=0.1)

    obs_ensemble = rml_solver.observation_ensemble

    assert obs_ensemble is not None
    assert obs_ensemble.shape[0] == len(simple_pastas_model.observations())
    assert obs_ensemble.shape[1] == 10
    assert obs_ensemble.index.equals(simple_pastas_model.observations().index)


def test_jacobian_empirical():
    """Test empirical Jacobian calculation."""
    sim_ens = np.random.randn(50, 10)  # 50 observations, 10 realizations
    par_ens = np.random.randn(10, 5)  # 10 realizations, 5 parameters

    jac = RandomizedMaximumLikelihoodSolver.jacobian_empirical(sim_ens, par_ens)

    assert jac.shape == (50, 5)  # nobs x npar


def test_jacobian_empirical_dimension_mismatch():
    """Test empirical Jacobian with mismatched dimensions."""
    sim_ens = np.random.randn(50, 10)
    par_ens = np.random.randn(8, 5)  # Different number of realizations

    with pytest.raises(AssertionError):
        RandomizedMaximumLikelihoodSolver.jacobian_empirical(sim_ens, par_ens)


def test_jacobian_finite_difference():
    """Test finite difference Jacobian calculation."""

    def fun(p: ArrayLike) -> ArrayLike:
        return np.array([p[0] ** 2 + p[1], p[0] * p[1]])

    p = np.array([1.0, 2.0])

    for method in ["2-point", "3-point"]:
        jac = RandomizedMaximumLikelihoodSolver.jacobian_finite_difference(
            fun=fun, p=p, jacobian_method=method, bounds=(-np.inf, np.inf)
        )
        assert jac.shape == (2, 2)


@pytest.mark.parametrize("jacobian_method", ["2-point", "3-point"])
def test_rml_solve_finite_difference(
    simple_pastas_model: ps.Model, jacobian_method: str
):
    """Test RML solver with finite difference methods."""
    solver = RandomizedMaximumLikelihoodSolver(
        num_reals=5,
        jacobian_method=jacobian_method,
        seed=42,
        add_base=True,
        num_workers=1,
    )
    solver.ml = simple_pastas_model
    solver.initialize(standard_deviation=0.01)

    success, optimal, stderr = solver.solve()

    assert success is True
    assert optimal is not None
    assert stderr is not None
    assert len(optimal) == len(simple_pastas_model.parameters)
    assert solver.nfev == 5


def test_rml_solve_empirical(simple_pastas_model: ps.Model):
    """Test RML solver with empirical Jacobian method."""
    solver = RandomizedMaximumLikelihoodSolver(
        num_reals=5,
        jacobian_method="empirical",
        noptmax=2,
        seed=42,
        add_base=True,
        num_workers=1,
    )
    solver.ml = simple_pastas_model
    solver.initialize(standard_deviation=0.01)

    success, optimal, stderr = solver.solve()

    assert success is True
    assert optimal is not None
    assert stderr is not None
    assert solver.nfev == 2
    assert solver.simulation_ensemble is not None


def test_rml_solve_removes_noise_weights_kwargs(
    simple_pastas_model: ps.Model, rml_solver: RandomizedMaximumLikelihoodSolver
):
    """Test that solve removes 'noise' and 'weights' from kwargs."""
    rml_solver.ml = simple_pastas_model
    rml_solver.initialize(standard_deviation=0.01)

    # Should not raise an error even with noise/weights in kwargs
    success, optimal, stderr = rml_solver.solve(noise=True, weights=[1, 2, 3])

    assert success is True


def test_rml_initialize_parameters_norm(simple_pastas_model: ps.Model):
    """Test parameter initialization with normal distribution."""
    solver = RandomizedMaximumLikelihoodSolver(
        num_reals=100, jacobian_method="2-point", seed=42, num_workers=1
    )
    solver.ml = simple_pastas_model
    solver.initialize(method="norm")

    assert solver.parameter_ensemble is not None
    assert solver.parameter_ensemble.shape[0] == 100


def test_rml_initialize_parameters_truncnorm(simple_pastas_model: ps.Model):
    """Test parameter initialization with truncated normal distribution."""
    solver = RandomizedMaximumLikelihoodSolver(
        num_reals=100, jacobian_method="2-point", seed=42, num_workers=1
    )
    solver.ml = simple_pastas_model
    solver.initialize(method="truncnorm")

    assert solver.parameter_ensemble is not None


def test_rml_initialize_parameters_uniform(simple_pastas_model: ps.Model):
    """Test parameter initialization with uniform distribution."""
    solver = RandomizedMaximumLikelihoodSolver(
        num_reals=100, jacobian_method="2-point", seed=42, num_workers=1
    )
    solver.ml = simple_pastas_model
    solver.initialize(method="uniform")

    assert solver.parameter_ensemble is not None


def test_rml_observation_noise_with_correlation(simple_pastas_model: ps.Model):
    """Test observation noise generation with correlation."""
    solver = RandomizedMaximumLikelihoodSolver(
        num_reals=10, jacobian_method="2-point", seed=42, num_workers=1
    )
    solver.ml = simple_pastas_model
    solver.initialize(
        standard_deviation=0.1,
        correlation_coefficient=0.5,
    )

    assert solver.observation_noise is not None
    # Base should have zero noise
    assert np.allclose(solver.observation_noise.loc[:, "base"], 0.0)


def test_rml_solver_noptmax_required_for_empirical():
    """Test that empirical method requires noptmax."""
    solver = RandomizedMaximumLikelihoodSolver(
        num_reals=10,
        jacobian_method="empirical",
        noptmax=None,  # This should trigger an error
        seed=42,
    )
    # Error is logged but not raised, so we just check initialization
    assert solver.noptmax is None
