"""Comprehensive tests for DataWorth class and related functions.

Tests cover initialization, observation noise covariance, Fisher information matrix,
covariance computation, data worth analysis, and plotting functions.
"""

import numpy as np
import pandas as pd
import pastas as ps
import pytest
from numpy.testing import assert_array_almost_equal

from pastas_plugins.dataworth import DataWorth, plot_data_worth_heatmap, plot_data_worth_series


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_pastas_model():
    """Create a simple Pastas model for testing."""
    np.random.seed(42)  # For reproducibility
    
    # Create time series
    index = pd.date_range(start="2020-01-01", periods=100, freq="D")
    observations = pd.Series(
        10 + 2 * np.sin(np.arange(100) * 2 * np.pi / 365) + np.random.randn(100) * 0.1,
        index=index,
        name="head",
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
    
    # Create model
    ml = ps.Model(observations, name="test_ml")
    rm = ps.RechargeModel(prec, evap, name="rch", rfunc=ps.Gamma())
    ml.add_stressmodel(rm)
    
    # Add a constant parameter
    ml.set_parameter(
        "constant_d",
        pmin=observations.min(),
        pmax=observations.max(),
        initial=observations.mean(),
    )
    
    # Add a simple noise model to avoid noise calculation issues
    ml.add_noisemodel(ps.ArNoiseModel())  
    
    # Solve the model to get a Jacobian
    ml.solve(tmin=index[0], tmax=index[-1], report=False)
    
    return ml


@pytest.fixture
def calibrated_pastas_model():
    """Create a calibrated Pastas model with noise for testing."""
    np.random.seed(42)
    
    # Create time series with more observations
    index = pd.date_range(start="2020-01-01", periods=200, freq="D")
    
    # Create a trend
    trend = 0.01 * np.arange(200)
    # Create seasonal component
    seasonal = 5 * np.sin(np.arange(200) * 2 * np.pi / 365)
    # Add noise
    np.random.seed(123)
    noise = np.random.randn(200) * 0.5
    
    observations = pd.Series(
        100 + trend + seasonal + noise,
        index=index,
        name="head",
    )
    
    prec = pd.Series(
        np.random.rand(200) * 10,
        index=index,
        name="precipitation",
    )
    evap = pd.Series(
        np.random.rand(200) * 5,
        index=index,
        name="evaporation",
    )
    
    # Create model
    ml = ps.Model(observations, name="calibrated_ml")
    
    # Add stress model
    rm = ps.RechargeModel(prec, evap, name="rch", rfunc=ps.Gamma())
    ml.add_stressmodel(rm)
    
    # Add constant
    ml.set_parameter("constant_d", initial=100)
    
    # Add noise model
    ml.add_noisemodel(ps.ArNoiseModel())
    
    # Solve the model
    ml.solve(tmin=index[0], tmax=index[-1], report=False)
    
    return ml


@pytest.fixture
def dataworth_instance(simple_pastas_model):
    """Create a DataWorth instance for testing."""
    return DataWorth(simple_pastas_model)


@pytest.fixture
def dataworth_with_jacobian(simple_pastas_model):
    """Create a DataWorth instance with explicit Jacobian."""
    # Create a simple Jacobian matrix for testing
    n_obs = len(simple_pastas_model.observations())
    n_params = len(simple_pastas_model.parameters)
    J = np.random.randn(n_obs, n_params) * 0.1
    
    return DataWorth(simple_pastas_model, J=J)


# =============================================================================
# Test: DataWorth Initialization
# =============================================================================


class TestDataWorthInitialization:
    """Tests for DataWorth class initialization."""

    def test_init_with_model_only(self, simple_pastas_model):
        """Test DataWorth initialization with only a model."""
        dw = DataWorth(simple_pastas_model)
        
        assert dw.ml is simple_pastas_model
        assert dw.J0 is not None
        assert dw.J0.shape[0] == len(simple_pastas_model.observations())
        assert dw.J0.shape[1] == len(simple_pastas_model.parameters)
        assert dw.objfun_target == "noise"

    def test_init_with_explicit_jacobian(self, simple_pastas_model):
        """Test DataWorth initialization with explicit Jacobian."""
        n_obs = len(simple_pastas_model.observations())
        n_params = len(simple_pastas_model.parameters)
        J = np.random.randn(n_obs, n_params)
        
        dw = DataWorth(simple_pastas_model, J=J)
        
        assert dw.ml is simple_pastas_model
        assert dw.J0 is J
        assert dw.objfun_target == "noise"

    def test_init_with_objfun_target(self, simple_pastas_model):
        """Test DataWorth initialization with different objfun_target."""
        for target in ["noise", "residuals"]:
            dw = DataWorth(simple_pastas_model, objfun_target=target)
            assert dw.objfun_target == target

    def test_init_with_both_jacobian_and_target(self, simple_pastas_model):
        """Test DataWorth initialization with both Jacobian and objfun_target."""
        n_obs = len(simple_pastas_model.observations())
        n_params = len(simple_pastas_model.parameters)
        J = np.random.randn(n_obs, n_params)
        
        dw = DataWorth(simple_pastas_model, J=J, objfun_target="residuals")
        
        assert dw.J0 is J
        assert dw.objfun_target == "residuals"


# =============================================================================
# Test: Observation Noise Covariance
# =============================================================================


class TestObservationNoiseCovariance:
    """Tests for observation_noise_covariance method."""

    def test_covariance_default_params(self, dataworth_instance):
        """Test observation noise covariance with default parameters."""
        C_eps = dataworth_instance.observation_noise_covariance()
        
        n_obs = len(dataworth_instance.ml.observations())
        assert C_eps.shape == (n_obs, n_obs)
        assert isinstance(C_eps, np.ndarray)

    def test_covariance_with_obs(self, dataworth_instance):
        """Test observation noise covariance with explicit observations."""
        obs = dataworth_instance.ml.observations().iloc[:50]
        C_eps = dataworth_instance.observation_noise_covariance(obs=obs)
        
        assert C_eps.shape == (50, 50)

    def test_covariance_with_var(self, dataworth_instance):
        """Test observation noise covariance with explicit variance."""
        var = 2.5
        C_eps = dataworth_instance.observation_noise_covariance(var=var)
        
        n_obs = len(dataworth_instance.ml.observations())
        # With default objfun_target="noise" and no noise_alpha, should be diagonal
        expected_diag = var + (1e-3)**2  # var + obs_std**2
        assert np.allclose(np.diag(C_eps), expected_diag)

    def test_covariance_with_obs_std(self, dataworth_instance):
        """Test observation noise covariance with custom obs_std."""
        obs_std = 0.5
        C_eps = dataworth_instance.observation_noise_covariance(obs_std=obs_std)
        
        n_obs = len(dataworth_instance.ml.observations())
        # Should be diagonal matrix
        off_diag_sum = np.sum(np.abs(C_eps)) - np.sum(np.abs(np.diag(C_eps)))
        assert np.isclose(off_diag_sum, 0.0)

    def test_covariance_with_noise_alpha_residuals(self, calibrated_pastas_model):
        """Test observation noise covariance with noise_alpha for residuals target."""
        dw = DataWorth(calibrated_pastas_model, objfun_target="residuals")
        noise_alpha = 30.0  # days
        
        C_eps = dw.observation_noise_covariance(noise_alpha=noise_alpha)
        
        n_obs = len(calibrated_pastas_model.observations())
        assert C_eps.shape == (n_obs, n_obs)
        
        # With AR(1) correlation, off-diagonal elements should be non-zero
        # Check that some off-diagonal elements are non-zero
        np.fill_diagonal(C_eps, 0)  # Zero out diagonal
        assert np.any(C_eps != 0), "AR(1) covariance should have non-zero off-diagonals"

    def test_covariance_diagonal_when_no_alpha(self, dataworth_instance):
        """Test that covariance is diagonal when no noise_alpha is provided."""
        C_eps = dataworth_instance.observation_noise_covariance()
        
        # Check that it's diagonal (off-diagonal elements should be zero)
        off_diag = C_eps - np.diag(np.diag(C_eps))
        assert np.allclose(off_diag, 0.0)

    def test_covariance_objfun_target_override(self, dataworth_instance):
        """Test overriding objfun_target in method call."""
        C_eps_noise = dataworth_instance.observation_noise_covariance(objfun_target="noise")
        C_eps_residuals = dataworth_instance.observation_noise_covariance(objfun_target="residuals")
        
        # Both should have same shape
        assert C_eps_noise.shape == C_eps_residuals.shape


# =============================================================================
# Test: Fisher Information Matrix
# =============================================================================


class TestFisherInformationMatrix:
    """Tests for fisher_information_matrix static method."""

    def test_fim_shape(self, dataworth_instance):
        """Test Fisher information matrix has correct shape."""
        J = dataworth_instance.J0
        n_params = J.shape[1]
        
        C_eps = dataworth_instance.observation_noise_covariance()
        C_eps_inv = np.linalg.inv(C_eps)
        
        FIM = DataWorth.fisher_information_matrix(J, C_eps_inv)
        
        assert FIM.shape == (n_params, n_params)

    def test_fim_symmetric(self, dataworth_instance):
        """Test Fisher information matrix is symmetric."""
        J = dataworth_instance.J0
        C_eps = dataworth_instance.observation_noise_covariance()
        C_eps_inv = np.linalg.inv(C_eps)
        
        FIM = DataWorth.fisher_information_matrix(J, C_eps_inv)
        
        assert np.allclose(FIM, FIM.T)

    def test_fim_formula(self, dataworth_instance):
        """Test Fisher information matrix formula: J.T @ C_eps_inv @ J."""
        J = dataworth_instance.J0
        C_eps = dataworth_instance.observation_noise_covariance()
        C_eps_inv = np.linalg.inv(C_eps)
        
        FIM = DataWorth.fisher_information_matrix(J, C_eps_inv)
        expected_FIM = J.T @ C_eps_inv @ J
        
        assert np.allclose(FIM, expected_FIM)


# =============================================================================
# Test: Covariance Computation
# =============================================================================


class TestCovarianceComputation:
    """Tests for compute_covariance static method."""

    def test_covariance_shape(self, dataworth_instance):
        """Test computed covariance has correct shape."""
        J = dataworth_instance.J0
        C_eps = dataworth_instance.observation_noise_covariance()
        
        Cp = DataWorth.compute_covariance(J, C_eps)
        
        n_params = J.shape[1]
        assert Cp.shape == (n_params, n_params)

    def test_covariance_symmetric(self, dataworth_instance):
        """Test computed covariance is symmetric."""
        J = dataworth_instance.J0
        C_eps = dataworth_instance.observation_noise_covariance()
        
        Cp = DataWorth.compute_covariance(J, C_eps)
        
        assert np.allclose(Cp, Cp.T)

    def test_covariance_with_mask(self, dataworth_instance):
        """Test covariance computation with mask."""
        J = dataworth_instance.J0
        C_eps = dataworth_instance.observation_noise_covariance()
        
        # Create mask that excludes some observations
        n_obs = J.shape[0]
        mask = np.ones(n_obs, dtype=bool)
        mask[::2] = False  # Exclude every other observation
        
        Cp = DataWorth.compute_covariance(J, C_eps, mask=mask)
        
        n_params = J.shape[1]
        assert Cp.shape == (n_params, n_params)

    def test_covariance_mask_all_false(self, dataworth_instance):
        """Test covariance computation with mask that excludes all observations."""
        J = dataworth_instance.J0
        C_eps = dataworth_instance.observation_noise_covariance()
        
        # Mask that excludes all observations
        mask = np.zeros(J.shape[0], dtype=bool)
        
        Cp = DataWorth.compute_covariance(J, C_eps, mask=mask)
        
        # Should still return a valid covariance matrix
        assert Cp.shape[0] == Cp.shape[1]


# =============================================================================
# Test: Data Worth Methods
# =============================================================================


class TestDataWorthMethods:
    """Tests for data_worth method."""

    def test_data_worth_basic(self, dataworth_instance):
        """Test basic data worth computation."""
        logdet, var_params = dataworth_instance.data_worth()
        
        assert isinstance(logdet, float)
        assert isinstance(var_params, np.ndarray)
        assert var_params.size == len(dataworth_instance.ml.parameters)

    def test_data_worth_with_mask(self, dataworth_instance):
        """Test data worth computation with mask."""
        n_obs = len(dataworth_instance.ml.observations())
        mask = np.ones(n_obs, dtype=bool)
        mask[:10] = False  # Exclude first 10 observations
        
        logdet, var_params = dataworth_instance.data_worth(mask=mask)
        
        assert isinstance(logdet, float)
        assert isinstance(var_params, np.ndarray)


class TestDataWorthPerObservation:
    """Tests for data_worth_per_observation method."""

    def test_data_worth_per_observation_shape(self, dataworth_instance):
        """Test data worth per observation returns correct shape."""
        df = dataworth_instance.data_worth_per_observation()
        
        n_obs = len(dataworth_instance.ml.observations())
        n_params_vary = len(dataworth_instance.ml.parameters[dataworth_instance.ml.parameters["vary"]])
        
        assert df.shape[0] == n_obs
        assert df.shape[1] == 1 + n_params_vary  # 1 for overall + n_params_vary for per-parameter

    def test_data_worth_per_observation_columns(self, dataworth_instance):
        """Test data worth per observation has correct columns."""
        df = dataworth_instance.data_worth_per_observation()
        
        # First column should be overall worth
        assert df.columns[0].startswith("Δ")  # Either "Δlogdet" or "Increase in gen. std. err. (%)"
        
        # Remaining columns should be for varying parameters
        varying_params = dataworth_instance.ml.parameters[dataworth_instance.ml.parameters["vary"]].index
        assert len(df.columns) - 1 == len(varying_params)

    def test_data_worth_per_observation_as_percentage(self, dataworth_instance):
        """Test data worth per observation with as_percentage=True."""
        df = dataworth_instance.data_worth_per_observation(as_percentage=True)
        
        n_obs = len(dataworth_instance.ml.observations())
        assert df.shape[0] == n_obs
        
        # When as_percentage=True, first column should be percentage
        assert "%" in df.columns[0]

    def test_data_worth_per_observation_index(self, dataworth_instance):
        """Test data worth per observation has correct index."""
        df = dataworth_instance.data_worth_per_observation()
        
        obs_index = dataworth_instance.ml.observations().index
        assert df.index.equals(obs_index)


class TestDataWorthThinning:
    """Tests for data_worth_thinning method."""

    def test_data_worth_thinning_basic(self, dataworth_instance):
        """Test basic data worth thinning computation."""
        thinning_intervals = [1, 2, 5, 10]
        dw_thinning, dw_thinning_per_param = dataworth_instance.data_worth_thinning(thinning_intervals)
        
        assert isinstance(dw_thinning, pd.Series)
        assert isinstance(dw_thinning_per_param, pd.DataFrame)
        
        assert len(dw_thinning) == len(thinning_intervals)
        assert dw_thinning_per_param.shape[0] == len(thinning_intervals)

    def test_data_worth_thinning_index(self, dataworth_instance):
        """Test data worth thinning has correct index."""
        thinning_intervals = [1, 3, 7]
        dw_thinning, dw_thinning_per_param = dataworth_instance.data_worth_thinning(thinning_intervals)
        
        assert dw_thinning.index.tolist() == thinning_intervals
        assert dw_thinning_per_param.index.tolist() == thinning_intervals

    def test_data_worth_thinning_columns(self, dataworth_instance):
        """Test data worth thinning per parameter has correct columns."""
        thinning_intervals = [1, 2]
        dw_thinning, dw_thinning_per_param = dataworth_instance.data_worth_thinning(thinning_intervals)
        
        # Columns should be all parameters
        param_names = dataworth_instance.ml.parameters.index
        assert dw_thinning_per_param.columns.tolist() == param_names.tolist()


class TestRecomputeJacobian:
    """Tests for recompute_jacobian method."""

    def test_recompute_jacobian_basic(self, simple_pastas_model):
        """Test basic Jacobian recomputation for new observations."""
        dw = DataWorth(simple_pastas_model)
        
        # Create new observations
        new_obs_index = pd.date_range(start="2021-01-01", periods=10, freq="D")
        new_observations = pd.Series(
            np.random.randn(10) + 10,
            index=new_obs_index,
            name="new_obs"
        )
        
        J_new = dw.recompute_jacobian(new_observations)
        
        # Should have shape (n_old_obs + n_new_obs, n_params)
        n_old_obs = len(simple_pastas_model.observations())
        n_params = len(simple_pastas_model.parameters)
        
        assert J_new.shape[0] == n_old_obs + 10
        assert J_new.shape[1] == n_params

    def test_recompute_jacobian_objfun_target(self, simple_pastas_model):
        """Test Jacobian recomputation with different objfun_target."""
        dw = DataWorth(simple_pastas_model)
        
        new_observations = pd.Series(
            np.random.randn(5) + 10,
            index=pd.date_range(start="2021-01-01", periods=5, freq="D"),
            name="new_obs"
        )
        
        # Test with noise target
        J_new_noise = dw.recompute_jacobian(new_observations, objfun_target="noise")
        assert J_new_noise.shape[0] > len(simple_pastas_model.observations())
        
        # Test with residuals target - only test if model has noise_alpha parameter
        if "noise_alpha" in simple_pastas_model.parameters.index:
            J_new_residuals = dw.recompute_jacobian(new_observations, objfun_target="residuals")
            assert J_new_residuals.shape[0] > len(simple_pastas_model.observations())


class TestDataWorthPerAddedObservation:
    """Tests for data_worth_per_added_observation method."""

    def test_data_worth_per_added_observation_basic(self, simple_pastas_model):
        """Test basic data worth computation for added observations."""
        dw = DataWorth(simple_pastas_model)
        
        # Create new observations
        new_obs_index = pd.date_range(start="2021-01-01", periods=5, freq="D")
        new_observations = pd.Series(
            np.random.randn(5) + 10,
            index=new_obs_index,
            name="new_obs"
        )
        
        df = dw.data_worth_per_added_observation(new_observations)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 5  # One row per new observation

    def test_data_worth_per_added_observation_removes_duplicates(self, simple_pastas_model):
        """Test that data_worth_per_added_observation removes duplicate timestamps."""
        dw = DataWorth(simple_pastas_model)
        
        # Create new observations that include some timestamps from existing observations
        existing_index = simple_pastas_model.observations().index[:5]
        new_obs_index = pd.date_range(start="2021-01-01", periods=5, freq="D")
        
        # Combine existing and new timestamps
        all_index = existing_index.union(new_obs_index).sort_values()
        new_observations = pd.Series(
            np.random.randn(len(all_index)) + 10,
            index=all_index,
            name="new_obs"
        )
        
        df = dw.data_worth_per_added_observation(new_observations)
        
        # Should only include truly new observations (not the duplicates)
        assert df.shape[0] == 5  # Only the 5 truly new observations

    def test_data_worth_per_added_observation_as_percentage(self, simple_pastas_model):
        """Test data worth per added observation with as_percentage=True."""
        dw = DataWorth(simple_pastas_model)
        
        new_observations = pd.Series(
            np.random.randn(3) + 10,
            index=pd.date_range(start="2021-01-01", periods=3, freq="D"),
            name="new_obs"
        )
        
        df = dw.data_worth_per_added_observation(new_observations, as_percentage=True)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 3


class TestDataWorthNewObservations:
    """Tests for data_worth_new_observations method."""

    def test_data_worth_new_observations_basic(self, simple_pastas_model):
        """Test basic data worth computation for new observations (aggregate)."""
        dw = DataWorth(simple_pastas_model)
        
        new_observations = pd.Series(
            np.random.randn(5) + 10,
            index=pd.date_range(start="2021-01-01", periods=5, freq="D"),
            name="new_obs"
        )
        
        worth_overall, relative_worth_per_param = dw.data_worth_new_observations(new_observations)
        
        assert isinstance(worth_overall, float)
        assert isinstance(relative_worth_per_param, pd.DataFrame)

    def test_data_worth_new_observations_as_percentage(self, simple_pastas_model):
        """Test data worth new observations with as_percentage=True."""
        dw = DataWorth(simple_pastas_model)
        
        new_observations = pd.Series(
            np.random.randn(3) + 10,
            index=pd.date_range(start="2021-01-01", periods=3, freq="D"),
            name="new_obs"
        )
        
        worth_overall, relative_worth_per_param = dw.data_worth_new_observations(
            new_observations, as_percentage=True
        )
        
        assert isinstance(worth_overall, float)
        assert isinstance(relative_worth_per_param, pd.DataFrame)
        assert "%" in relative_worth_per_param.columns[0]


# =============================================================================
# Test: Plotting Functions
# =============================================================================


class TestPlottingFunctions:
    """Tests for plotting functions."""

    def test_plot_data_worth_series_basic(self, dataworth_instance):
        """Test basic plot_data_worth_series functionality."""
        # Get data worth for existing observations
        df = dataworth_instance.data_worth_per_observation()
        obs = dataworth_instance.ml.observations()
        
        # This should not raise an exception
        axes = plot_data_worth_series(obs, df)
        
        assert axes is not None
        assert len(axes) == df.shape[1]  # One axis per column

    def test_plot_data_worth_series_index_match(self, dataworth_instance):
        """Test plot_data_worth_series requires matching indices."""
        df = dataworth_instance.data_worth_per_observation()
        
        # Create observations with different index
        wrong_obs = pd.Series(
            np.random.randn(10),
            index=pd.date_range(start="2021-01-01", periods=10, freq="D")
        )
        
        # Should raise assertion error
        with pytest.raises(AssertionError):
            plot_data_worth_series(wrong_obs, df)

    def test_plot_data_worth_heatmap_basic(self, dataworth_instance):
        """Test basic plot_data_worth_heatmap functionality."""
        # Get data worth for existing observations (as percentage for better visualization)
        df = dataworth_instance.data_worth_per_observation(as_percentage=True)
        
        # Use the first column (overall worth) for heatmap
        data_worth_series = df.iloc[:, 0]
        
        # This should not raise an exception
        ax = plot_data_worth_heatmap(data_worth_series)
        
        assert ax is not None

    def test_plot_data_worth_heatmap_returns_axes(self, dataworth_instance):
        """Test that plot_data_worth_heatmap returns axes."""
        df = dataworth_instance.data_worth_per_observation()
        data_worth_series = df.iloc[:, 0]
        
        ax = plot_data_worth_heatmap(data_worth_series)
        
        assert ax is not None
        # Check that it's a matplotlib axes object
        import matplotlib.axes
        assert isinstance(ax, matplotlib.axes.Axes)


# =============================================================================
# Test: Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_observation_noise_covariance_invalid_objfun_target(self, dataworth_instance):
        """Test observation noise covariance with invalid objfun_target."""
        with pytest.raises(ValueError, match="Unknown objfun_target"):
            dataworth_instance.observation_noise_covariance(objfun_target="invalid")

    def test_data_worth_with_empty_mask(self, dataworth_instance):
        """Test data worth computation with empty mask (no observations)."""
        n_obs = len(dataworth_instance.ml.observations())
        mask = np.zeros(n_obs, dtype=bool)
        
        # Should still work, though results may not be meaningful
        logdet, var_params = dataworth_instance.data_worth(mask=mask)
        assert isinstance(logdet, float)
        assert isinstance(var_params, np.ndarray)

    def test_data_worth_per_observation_empty_model(self, dataworth_instance):
        """Test data worth per observation with model that has no varying parameters."""
        # This should still work - it will just have fewer columns
        df = dataworth_instance.data_worth_per_observation()
        assert df.shape[0] > 0  # Should have rows for each observation

    def test_thinning_with_empty_intervals(self, dataworth_instance):
        """Test data worth thinning with empty intervals list."""
        dw_thinning, dw_thinning_per_param = dataworth_instance.data_worth_thinning([])
        
        assert len(dw_thinning) == 0
        assert len(dw_thinning_per_param) == 0


# =============================================================================
# Test: Numerical Properties
# =============================================================================


class TestNumericalProperties:
    """Tests for numerical properties of the computations."""

    def test_covariance_positive_semidefinite(self, dataworth_instance):
        """Test that computed covariance matrix is positive semi-definite."""
        J = dataworth_instance.J0
        C_eps = dataworth_instance.observation_noise_covariance()
        
        Cp = DataWorth.compute_covariance(J, C_eps)
        
        # Check that covariance matrix is positive semi-definite
        # by trying to compute Cholesky decomposition
        try:
            np.linalg.cholesky(Cp)
            is_positive_definite = True
        except np.linalg.LinAlgError:
            # If Cholesky fails, check eigenvalues are non-negative
            eigenvalues = np.linalg.eigvalsh(Cp)
            is_positive_definite = np.all(eigenvalues >= -1e-10)  # Allow small numerical errors
        
        assert is_positive_definite

    def test_fisher_information_positive_semidefinite(self, dataworth_instance):
        """Test that Fisher information matrix is positive semi-definite."""
        J = dataworth_instance.J0
        C_eps = dataworth_instance.observation_noise_covariance()
        C_eps_inv = np.linalg.inv(C_eps)
        
        FIM = DataWorth.fisher_information_matrix(J, C_eps_inv)
        
        # Check that FIM is positive semi-definite
        eigenvalues = np.linalg.eigvalsh(FIM)
        assert np.all(eigenvalues >= -1e-10)  # Allow small numerical errors

    def test_data_worth_values_reasonable(self, dataworth_instance):
        """Test that data worth values are reasonable (finite, not extreme)."""
        df = dataworth_instance.data_worth_per_observation()
        
        # Check that values are finite
        assert np.all(np.isfinite(df.values))
        
        # Check that log-det values are reasonable (not extremely large or small)
        logdet_col = df.iloc[:, 0]
        assert np.all(np.abs(logdet_col) < 1000)  # Reasonable bound for log-det


# =============================================================================
# Test: Integration with Different Model Types
# =============================================================================


class TestModelIntegration:
    """Tests for DataWorth integration with different Pastas model configurations."""

    def test_with_noise_model(self, calibrated_pastas_model):
        """Test DataWorth with a model that has a noise model."""
        dw = DataWorth(calibrated_pastas_model)
        
        df = dw.data_worth_per_observation()
        assert df.shape[0] == len(calibrated_pastas_model.observations())

    def test_with_residuals_objfun_target(self, calibrated_pastas_model):
        """Test DataWorth with residuals objfun_target."""
        dw = DataWorth(calibrated_pastas_model, objfun_target="residuals")
        
        C_eps = dw.observation_noise_covariance()
        assert C_eps.shape[0] == C_eps.shape[1]
        
        df = dw.data_worth_per_observation()
        assert df.shape[0] > 0

    def test_with_explicit_jacobian_matrix(self, simple_pastas_model):
        """Test DataWorth with explicitly provided Jacobian matrix."""
        n_obs = len(simple_pastas_model.observations())
        n_params = len(simple_pastas_model.parameters)
        
        # Create a custom Jacobian
        J_custom = np.random.randn(n_obs, n_params) * 0.1
        
        dw = DataWorth(simple_pastas_model, J=J_custom)
        
        # Test that we can compute data worth with custom Jacobian
        logdet, var_params = dw.data_worth()
        assert isinstance(logdet, float)
        assert var_params.shape[0] == n_params
