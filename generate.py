import numpy as np
import pandas as pd
from typing import Literal
from hashlib import sha256


def sparsify_features(X_sparsified: np.array, sparsity: float) -> np.array:
    # Apply sparsity by randomly zeroing out elements
    mask = np.random.binomial(1, sparsity, size=X_sparsified.shape)
    X_sparsified = X_sparsified * mask

    return X_sparsified


def create_redundant_features(X, n_redundant_linear, n_x_features):
    # Generate linearly dependent redundant features
    linear_redundant_features = []
    if n_redundant_linear > 0:
        for _ in range(n_redundant_linear):
            # Randomly select features to combine
            selected_features = np.random.choice(n_x_features, size=2, replace=False)
            coeffs = np.random.uniform(-1, 1, size=2)
            redundant_feature = (
                X[:, selected_features[0]] * coeffs[0]
                + X[:, selected_features[1]] * coeffs[1]
            )
            linear_redundant_features.append(redundant_feature)
        linear_redundant_features = np.column_stack(linear_redundant_features)
    return linear_redundant_features


def add_uninformative_features(n_redundant_noise, n_samples):
    # Generate noise-based redundant features
    noise_redundant_features = []
    if n_redundant_noise > 0:
        noise_redundant_features = np.random.normal(size=(n_samples, n_redundant_noise))

    return noise_redundant_features


def add_noise(Y, noise_level):
    noise = np.random.normal(scale=noise_level, size=Y.shape)
    noisy_Y = Y + noise
    return noisy_Y


def create_signal(X, transformer: Literal["linear", "composite"]):
    if transformer == "linear":
        weights = np.random.uniform(-1, 1, size=(X.shape[1], 1))
        bias = np.random.uniform(-1, 1, size=(1, 1))
        Y = X @ weights + bias
    elif transformer == "composite":
        pass  # TODO

    return Y


def generate_data(
    n_samples=1000,
    n_x_features=5,
    n_y_features=1,
    noise_level=0.1,
    n_redundant_linear=0,
    n_redundant_noise=0,
    sparsity=1.0,
    transformer: str = "linear",
    random_state=None,
    to_array: bool = True,
):
    if random_state is not None:
        np.random.seed(random_state)

    # Generate independent features (X) from a normal distribution
    X = np.random.normal(size=(n_samples, n_x_features))
    X = sparsify_features(X_sparsified=X, sparsity=sparsity)

    Y = create_signal(X=X, transformer=transformer)

    if n_redundant_linear > 0:
        linear_redundant_features = create_redundant_features(
            X=X, n_redundant_linear=n_redundant_linear, n_x_features=n_x_features
        )
        X = np.hstack([X, linear_redundant_features])
    if n_redundant_noise > 0:
        uninformative_features = add_uninformative_features(
            n_redundant_noise=n_redundant_noise, n_samples=n_samples
        )
        X = np.hstack([X, uninformative_features])

    # Add noise
    Y = add_noise(Y, noise_level=noise_level)

    # Convert to DataFrame for better usability
    X_columns = (
        [f"X{i + 1}" for i in range(n_x_features)]
        + [f"LinearRedundant{i + 1}" for i in range(n_redundant_linear)]
        + [f"NoiseRedundant{i + 1}" for i in range(n_redundant_noise)]
    )
    df_X = pd.DataFrame(X, columns=X_columns)
    df_Y = pd.DataFrame(Y, columns=[f"Y{i + 1}" for i in range(n_y_features)])

    if to_array:
        df_X = df_X.to_numpy()
        df_Y = df_Y.to_numpy().reshape(-1)

    return df_X, df_Y


def noisy_rastrigin(x, A=10, noise_seed=42, noise=0.1):
    # TODO: check if this performs as intended with the random seed hashing:
    n = len(x)
    # Use a hash of the input x and the seed to generate a unique state for RNG
    x_bytes = x.tobytes()
    combined_bytes = x_bytes + noise_seed.to_bytes(
        4, "big"
    )  # combines the bytes of the seed and x
    hash_value = int.from_bytes(
        sha256(combined_bytes).digest()[:4], "big"
    )  # creates a hash and takes the first four bytes.
    rng = np.random.default_rng(hash_value)

    rastrigin_value = A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

    noise = rng.normal(loc=0.0, scale=noise)

    return rastrigin_value + noise


class ObjectiveSurfaceGenerator:
    def __init__(self, generator: str):
        self.generator = generator

    def predict(self, x):
        if self.generator == "rastrigin":
            y = noisy_rastrigin(x=x)
        return y
