import numpy as np
import pandas as pd
from typing import Literal
from hashlib import sha256

from jahs_bench import Benchmark

from yahpo_gym import local_config
from yahpo_gym import benchmark_set

local_config.init_config()
local_config.set_data_path("yahpo_bench_data")


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


def noisy_rastrigin(x, A=20, noise_seed=42, noise=0):
    n = len(x)
    x_bytes = x.tobytes()
    combined_bytes = x_bytes + noise_seed.to_bytes(4, "big")
    hash_value = int.from_bytes(sha256(combined_bytes).digest()[:4], "big")
    rng = np.random.default_rng(hash_value)
    rastrigin_value = A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    noise = rng.normal(loc=0.0, scale=noise)
    return rastrigin_value + noise


def noisy_ackley(x, a=20, b=0.2, c=2 * np.pi, noise_seed=42, noise=0):
    n = len(x)
    x_bytes = x.tobytes()
    combined_bytes = x_bytes + noise_seed.to_bytes(4, "big")
    hash_value = int.from_bytes(sha256(combined_bytes).digest()[:4], "big")
    rng = np.random.default_rng(hash_value)
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / n))
    term2 = -np.exp(np.sum(np.cos(c * x)) / n)
    ackley_value = term1 + term2 + a + np.exp(1)
    noise = rng.normal(loc=0.0, scale=noise)
    return ackley_value + noise


def noisy_griewank(x, noise_seed=42, noise=0):
    n = len(x)
    x_bytes = x.tobytes()
    combined_bytes = x_bytes + noise_seed.to_bytes(4, "big")
    hash_value = int.from_bytes(sha256(combined_bytes).digest()[:4], "big")
    rng = np.random.default_rng(hash_value)
    term1 = np.sum(x**2) / 4000
    term2 = 1
    for i in range(n):
        term2 *= np.cos(x[i] / np.sqrt(i + 1))
    griewank_value = term1 - term2 + 1
    noise = rng.normal(loc=0.0, scale=noise)
    return griewank_value + noise


def noisy_weierstrass(x, a=0.5, b=3, kmax=20, noise_seed=42, noise=0):
    n = len(x)
    x_bytes = x.tobytes()
    combined_bytes = x_bytes + noise_seed.to_bytes(4, "big")
    hash_value = int.from_bytes(sha256(combined_bytes).digest()[:4], "big")
    rng = np.random.default_rng(hash_value)
    weierstrass_value = 0
    for i in range(n):
        for k in range(kmax + 1):
            weierstrass_value += (a**k) * np.cos(2 * np.pi * (b**k) * (x[i] + 0.5))
        for k in range(kmax + 1):
            weierstrass_value -= (a**k) * np.cos(2 * np.pi * (b**k) * 0.5)
    noise = rng.normal(loc=0.0, scale=noise)
    return weierstrass_value + noise


def noisy_shekel(x, m=10, noise_seed=42, noise=0):  # m is the number of local minima
    n = len(x)
    x_bytes = x.tobytes()
    combined_bytes = x_bytes + noise_seed.to_bytes(4, "big")
    hash_value = int.from_bytes(sha256(combined_bytes).digest()[:4], "big")
    rng = np.random.default_rng(hash_value)
    A = np.random.rand(m, n) * 10  # random A matrix for each run
    C = np.random.rand(m) * 10
    shekel_value = 0
    for i in range(m):
        shekel_value -= 1 / (C[i] + np.sum((x - A[i]) ** 2))
    noise = rng.normal(loc=0.0, scale=noise)
    return -(shekel_value + noise)


def noisy_hartmann6(x, noise_seed=42, noise=0):
    x_bytes = x.tobytes()
    combined_bytes = x_bytes + noise_seed.to_bytes(4, "big")
    hash_value = int.from_bytes(sha256(combined_bytes).digest()[:4], "big")
    rng = np.random.default_rng(hash_value)
    alpha = [1.0, 1.2, 3.0, 3.2]
    A = np.array(
        [
            [1.0, 1.2, 3.0, 3.2],
            [3.6, 1.6, 0.7, 3.9],
            [4.0, 1.6, 0.8, 3.4],
            [1.6, 0.0, 3.6, 0.8],
            [1.6, 0.0, 3.6, 0.8],
        ]
    )
    P = np.array(
        [
            [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5894],
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
            [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
        ]
    )
    hartmann6_value = 0
    for i in range(4):
        inner_sum = 0
        for j in range(6):
            inner_sum += A[i, j] * (x[j] - P[i, j]) ** 2
        hartmann6_value -= alpha[i] * np.exp(-inner_sum)
    noise = rng.normal(loc=0.0, scale=noise)
    return -(hartmann6_value + noise)


class ObjectiveSurfaceGenerator:
    def __init__(self, generator: str):
        self.generator = generator

    def predict(self, params):
        x = np.array(list(params.values()))
        if self.generator == "rastrigin":
            y = noisy_rastrigin(x=x)
        elif self.generator == "ackley":
            y = noisy_ackley(x=x)
        elif self.generator == "griewank":
            y = noisy_griewank(x=x)
        elif self.generator == "weierstrass":
            y = noisy_weierstrass(x=x)
        elif self.generator == "shekel":
            y = noisy_shekel(x=x)
        elif self.generator == "hartmann6":
            y = noisy_hartmann6(x=x)
        else:
            raise ValueError(f"Unknown generator: {self.generator}")
        return y

    def predict_runtime(self, params):
        return 0


class Jahs201Generator:
    def __init__(self, dataset: str, metrics: list[str] = ["valid-acc", "runtime"]):
        self.generator = Benchmark(task=dataset, lazy=False, metrics=metrics)

    def predict(self, params):
        return -self.generator(params)[200]["valid-acc"]

    def predict_runtime(self, params):
        return self.generator(params)[200]["runtime"]


class YahpoGenerator:
    def __init__(self, dataset: str):
        self.generator = benchmark_set.BenchmarkSet(dataset)
        # self.generator.set_instance(self.generator.instances[0])

    def predict(self, params):
        return -self.generator.objective_function(params)[0]["val_accuracy"]

    def predict_runtime(self, params):
        # TODO: Check unit of time
        print(self.generator.objective_function(params)[0])
        return self.generator.objective_function(params)[0]["time"]


# class Jahs201Generator:
#     def __init__(self, dataset: str, metrics: list[str] = ["valid-acc", "runtime"]):
#         self.generator = Benchmark(task=dataset, lazy=False, metrics=metrics)

#     def predict(self, params):
#         response = self.generator(params)[200]
#         return -response["valid-acc"], response["runtime"]
