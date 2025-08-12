"""
Gaussian Process sampler with CONFOPT acquisition functions for HPO Bench.

This module implements a Gaussian process sampler inspired by Optuna's GPSampler
but supporting acquisition functions from the CONFOPT framework. It provides
a bridge between GP-based Bayesian optimization and the diverse acquisition
strategies available in CONFOPT.

Key Features:
- GP-based surrogate modeling using Optuna's GP infrastructure
- Support for CONFOPT acquisition functions (EI, Thompson, Entropy, Bounds)
- Configurable number of candidate configurations
- Integration with HPO Bench tuning framework
- Closed-form acquisition function optimization where possible
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Sequence, TYPE_CHECKING
from enum import Enum

import numpy as np
import optuna
from optuna.distributions import BaseDistribution
from optuna.samplers._base import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial, TrialState

if TYPE_CHECKING:
    from optuna.study import Study
else:
    from optuna._imports import _LazyImport
    torch = _LazyImport("torch")
    gp_search_space = _LazyImport("optuna._gp.search_space")
    gp = _LazyImport("optuna._gp.gp")
    optim_mixed = _LazyImport("optuna._gp.optim_mixed")
    acqf = _LazyImport("optuna._gp.acqf")
    prior = _LazyImport("optuna._gp.prior")


logger = logging.getLogger(__name__)
EPS = 1e-10


class CONFOPTAcquisitionFunction(Enum):
    """Supported CONFOPT acquisition functions."""
    EXPECTED_IMPROVEMENT = "expected_improvement"
    LOG_EXPECTED_IMPROVEMENT = "log_expected_improvement"  # More comparable to Optuna
    THOMPSON_SAMPLING = "thompson_sampling"
    CONFIDENCE_BOUND = "confidence_bound"  # UCB/LCB unified
    MAX_VALUE_ENTROPY_SEARCH = "max_value_entropy_search"


def _standardize_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize values for GP modeling."""
    clipped_values = gp.warn_and_convert_inf(values)
    means = np.mean(clipped_values, axis=0)
    stds = np.std(clipped_values, axis=0)
    standardized_values = (clipped_values - means) / np.maximum(EPS, stds)
    return standardized_values, means, stds


class CONFOPTGPSampler(BaseSampler):
    """
    Gaussian Process sampler with CONFOPT acquisition functions.
    
    This sampler combines Optuna's GP infrastructure with acquisition functions
    inspired by the CONFOPT framework. It supports various acquisition strategies
    while maintaining the theoretical foundation of Gaussian process-based
    Bayesian optimization.
    
    Args:
        acquisition_function: CONFOPT acquisition function to use
        n_candidates: Number of candidate configurations to evaluate
        seed: Random seed for reproducibility
        independent_sampler: Sampler for initial trials and conditional parameters
        n_startup_trials: Number of initial random trials
        deterministic_objective: Whether the objective is deterministic
        beta: Exploration parameter for confidence bound methods
        xi: Improvement threshold for expected improvement
        n_samples: Number of samples for Monte Carlo acquisition functions
    """
    
    def __init__(
        self,
        *,
        acquisition_function: CONFOPTAcquisitionFunction = CONFOPTAcquisitionFunction.EXPECTED_IMPROVEMENT,
        n_candidates: int = 2000,
        seed: int | None = None,
        independent_sampler: BaseSampler | None = None,
        n_startup_trials: int = 10,
        deterministic_objective: bool = False,
        maximize: bool = False,
        beta: float = 2.0,
        xi: float = 0.01,
        n_samples: int = 1000,
    ) -> None:
        self._acquisition_function = acquisition_function
        self._n_candidates = n_candidates
        self._rng = LazyRandomState(seed)
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._intersection_search_space = optuna.search_space.IntersectionSearchSpace()
        self._n_startup_trials = n_startup_trials
        self._log_prior: Callable[[gp.KernelParamsTensor], torch.Tensor] = prior.default_log_prior
        self._minimum_noise: float = prior.DEFAULT_MINIMUM_NOISE_VAR
        self._kernel_params_cache_list: list[gp.KernelParamsTensor] | None = None
        self._deterministic = deterministic_objective
        
        # Direction handling
        self._maximize = maximize
        self._y_sign = 1.0 if maximize else -1.0  # Transform to always minimize internally
        
        # CONFOPT-specific parameters
        self._beta = beta  # For confidence bounds
        self._xi = xi  # For EI
        self._n_samples = n_samples  # For Monte Carlo methods
        
        logger.info(f"Initialized CONFOPTGPSampler with acquisition function: {acquisition_function.value}")
    
    def reseed_rng(self) -> None:
        """Reseed the random number generator."""
        self._rng.rng.seed()
        self._independent_sampler.reseed_rng()
    
    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        """Infer the search space for multivariate sampling."""
        search_space = {}
        for name, distribution in self._intersection_search_space.calculate(study).items():
            if distribution.single():
                continue
            search_space[name] = distribution
        return search_space
    
    def _expected_improvement_acquisition(
        self, 
        mean: torch.Tensor, 
        var: torch.Tensor, 
        f_best: float
    ) -> torch.Tensor:
        """
        Compute Expected Improvement acquisition function.
        
        This implementation follows the standard EI formula:
        EI(x) = σ(x) * [z * Φ(z) + φ(z)]
        where z = (f_best - μ(x) - ξ) / σ(x)
        """
        std = torch.sqrt(var)
        
        # Avoid division by zero
        std_safe = torch.clamp(std, min=EPS)
        z = (f_best - mean - self._xi) / std_safe
        
        # Compute normal CDF and PDF
        normal_dist = torch.distributions.Normal(0, 1)
        phi = normal_dist.cdf(z)  # CDF
        pdf = torch.exp(normal_dist.log_prob(z))  # PDF
        
        # Expected Improvement formula
        ei = std * (z * phi + pdf)
        
        # Handle numerical issues: EI should be non-negative
        ei = torch.clamp(ei, min=0.0)
        
        return ei
    
    def _log_expected_improvement_acquisition(
        self, 
        mean: torch.Tensor, 
        var: torch.Tensor, 
        f_best: float
    ) -> torch.Tensor:
        """
        Compute Log Expected Improvement acquisition function.
        
        This implementation uses the numerically stable log-space computation
        similar to Optuna's approach.
        """
        std = torch.sqrt(var + EPS)  # Add small epsilon for numerical stability
        z = (f_best - mean - self._xi) / std
        
        # Use log-space computation for numerical stability
        # This follows the approach in Optuna's acqf module
        small_mask = z < -25
        normal_mask = ~small_mask
        
        log_ei = torch.zeros_like(z)
        
        # For normal values, use standard log EI computation
        if torch.any(normal_mask):
            z_normal = z[normal_mask]
            std_normal = std[normal_mask]
            
            # CDF and PDF of standard normal
            cdf = 0.5 * torch.special.erfc(-z_normal / np.sqrt(2))
            pdf = torch.exp(-0.5 * z_normal**2) / np.sqrt(2 * np.pi)
            
            # Log EI = log(std) + log(z * cdf + pdf)
            ei_normal = z_normal * cdf + pdf
            log_ei[normal_mask] = torch.log(std_normal) + torch.log(torch.clamp(ei_normal, min=EPS))
        
        # For small values, use asymptotic expansion
        if torch.any(small_mask):
            z_small = z[small_mask]
            std_small = std[small_mask]
            
            # Asymptotic expansion for very negative z
            sqrt_2pi = np.sqrt(2 * np.pi)
            r = np.sqrt(0.5 * np.pi) * torch.special.erfcx(-z_small / np.sqrt(2))
            log_ei[small_mask] = torch.log(std_small) - 0.5 * z_small**2 + torch.log((z_small * r + 1) / sqrt_2pi)
        
        return log_ei
    
    def _confidence_bound_acquisition(
        self, 
        mean: torch.Tensor, 
        var: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Confidence Bound acquisition function.
        
        For minimization problems (internally): Uses Upper Confidence Bound (UCB)
        For maximization problems: The sign will be handled by y_sign transformation
        """
        return mean + self._beta * torch.sqrt(var)
    
    def _thompson_sampling_acquisition(
        self, 
        mean: torch.Tensor, 
        var: torch.Tensor
    ) -> torch.Tensor:
        """Compute Thompson Sampling acquisition function."""
        # Sample from the posterior
        std = torch.sqrt(var)
        samples = torch.normal(mean, std)
        return samples
    
    def _max_value_entropy_search_acquisition(
        self, 
        mean: torch.Tensor, 
        var: torch.Tensor, 
        y_samples: np.ndarray
    ) -> torch.Tensor:
        """
        Compute Max Value Entropy Search acquisition function.
        
        This is a simplified version that approximates the information gain
        about the maximum value through sampling-based entropy estimation.
        """
        # Sample from posterior
        std = torch.sqrt(var)
        n_samples = min(self._n_samples, 100)  # Limit for computational efficiency
        
        # Generate samples for each candidate
        samples = torch.normal(
            mean.unsqueeze(-1).expand(-1, n_samples), 
            std.unsqueeze(-1).expand(-1, n_samples)
        )
        
        # Estimate entropy reduction (simplified)
        # Higher variance and higher mean both contribute to information gain
        entropy_approx = torch.var(samples, dim=-1) + torch.abs(mean - torch.mean(torch.tensor(y_samples)))
        
        return entropy_approx
    
    def _compute_acquisition_function(
        self,
        normalized_params: torch.Tensor,
        kernel_params: gp.KernelParamsTensor,
        search_space: gp_search_space.SearchSpace,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        y_samples: np.ndarray,
    ) -> torch.Tensor:
        """Compute acquisition function values for candidate points."""
        # Compute kernel matrices
        K_train = gp.kernel(kernel_params, X_train, X_train)
        K_noise = K_train + torch.eye(X_train.shape[0], dtype=K_train.dtype, device=K_train.device) * kernel_params.noise_var
        
        # Compute inverse and solve
        try:
            K_inv = torch.linalg.inv(K_noise)
            alpha = torch.linalg.solve(K_noise, Y_train)
        except torch.linalg.LinAlgError:
            # Add jitter for numerical stability
            jitter = 1e-6
            K_noise_jitter = K_noise + torch.eye(K_noise.shape[0], dtype=K_noise.dtype, device=K_noise.device) * jitter
            K_inv = torch.linalg.inv(K_noise_jitter)
            alpha = torch.linalg.solve(K_noise_jitter, Y_train)
        
        # Compute posterior mean and variance using Optuna's posterior function
        mean, var = gp.posterior(
            kernel_params,
            X_train,
            Y_train,
            K_inv,
            alpha,
            normalized_params,
        )
        
        # Add numerical stability
        var = var + EPS
        
        # Compute acquisition function based on selected method
        if self._acquisition_function == CONFOPTAcquisitionFunction.EXPECTED_IMPROVEMENT:
            f_best = torch.max(Y_train).item() if len(Y_train) > 0 else 0.0
            return self._expected_improvement_acquisition(mean, var, f_best)
        
        elif self._acquisition_function == CONFOPTAcquisitionFunction.LOG_EXPECTED_IMPROVEMENT:
            f_best = torch.max(Y_train).item() if len(Y_train) > 0 else 0.0
            return self._log_expected_improvement_acquisition(mean, var, f_best)
        
        elif self._acquisition_function == CONFOPTAcquisitionFunction.CONFIDENCE_BOUND:
            return self._confidence_bound_acquisition(mean, var)
        
        elif self._acquisition_function == CONFOPTAcquisitionFunction.THOMPSON_SAMPLING:
            return self._thompson_sampling_acquisition(mean, var)
        
        elif self._acquisition_function == CONFOPTAcquisitionFunction.MAX_VALUE_ENTROPY_SEARCH:
            return self._max_value_entropy_search_acquisition(mean, var, y_samples)
        
        else:
            raise ValueError(f"Unknown acquisition function: {self._acquisition_function}")
    
    def _optimize_acquisition_function(
        self,
        kernel_params: gp.KernelParamsTensor,
        search_space: gp_search_space.SearchSpace,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        y_samples: np.ndarray,
    ) -> np.ndarray:
        """Optimize the acquisition function to find the next candidate."""
        # Generate candidate points using quasi-random sampling
        n_dims = len(search_space.scale_types)
        
        # Use Sobol sequence for better space coverage
        sobol_engine = torch.quasirandom.SobolEngine(n_dims, scramble=True, seed=self._rng.rng.randint(2**31))
        candidates = sobol_engine.draw(self._n_candidates).double()
        
        # Compute acquisition function values
        acq_values = self._compute_acquisition_function(
            candidates, kernel_params, search_space, X_train, Y_train, y_samples
        )
        
        # Find the best candidate
        best_idx = torch.argmax(acq_values)
        best_candidate = candidates[best_idx]
        
        return best_candidate.numpy()
    
    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        """Sample parameters using GP with CONFOPT acquisition functions."""
        if search_space == {}:
            return {}
        
        states = (TrialState.COMPLETE,)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)
        
        if len(trials) < self._n_startup_trials:
            return {}
        
        # Convert to internal representation
        (
            internal_search_space,
            normalized_params,
        ) = gp_search_space.get_search_space_and_normalized_params(trials, search_space)
        
        # Transform target values for consistent minimization internally
        # Always transform to minimization problem internally
        raw_values = np.array([trial.values[0] for trial in trials])  # Single objective for now
        
        # Apply sign transformation: maximize -> negate to minimize, minimize -> keep as is
        if self._maximize:
            # If we want to maximize, negate values to convert to minimization
            transformed_values = -raw_values
        else:
            # If we want to minimize, keep values as is
            transformed_values = raw_values
            
        standardized_values, _, _ = _standardize_values(transformed_values.reshape(-1, 1))
        
        # Clear cache if search space changes
        if (self._kernel_params_cache_list is not None and 
            len(self._kernel_params_cache_list[0].inverse_squared_lengthscales) != 
            len(internal_search_space.scale_types)):
            self._kernel_params_cache_list = None
        
        # Fit GP model
        is_categorical = internal_search_space.scale_types == gp_search_space.ScaleType.CATEGORICAL
        cache = self._kernel_params_cache_list[0] if self._kernel_params_cache_list is not None else None
        
        kernel_params = gp.fit_kernel_params(
            X=normalized_params,
            Y=standardized_values[:, 0],
            is_categorical=is_categorical,
            log_prior=self._log_prior,
            minimum_noise=self._minimum_noise,
            initial_kernel_params=cache,
            deterministic_objective=self._deterministic,
        )
        
        self._kernel_params_cache_list = [kernel_params]
        
        # Convert to tensors for acquisition function optimization
        X_train = torch.from_numpy(normalized_params).double()
        Y_train = torch.from_numpy(standardized_values[:, 0]).double()
        
        # Optimize acquisition function
        normalized_param = self._optimize_acquisition_function(
            kernel_params, internal_search_space, X_train, Y_train, transformed_values
        )
        
        # Convert back to external representation
        return gp_search_space.get_unnormalized_param(search_space, normalized_param)
    
    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        """Sample a single parameter independently."""
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )
    
    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        """Called before each trial."""
        self._independent_sampler.before_trial(study, trial)
    
    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        """Called after each trial."""
        self._independent_sampler.after_trial(study, trial, state, values)
