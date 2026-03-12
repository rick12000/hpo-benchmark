import numpy as np


def rastrigin(input_vector, amplitude=20):
    """Rastrigin function - multimodal optimization benchmark.

    Args:
        input_vector: Input vector with shape (n_dimensions,).
        amplitude: Amplitude parameter (default: 20).

    Returns:
        Function value at point input_vector.
    """
    num_dimensions = len(input_vector)
    rastrigin_value = amplitude * num_dimensions + np.sum(input_vector**2 - amplitude * np.cos(2 * np.pi * input_vector))
    return rastrigin_value


def ackley(input_vector, amplitude=20, exponential_decay=0.2, oscillation_frequency=2 * np.pi):
    """Ackley function - multimodal optimization benchmark.

    Args:
        input_vector: Input vector with shape (n_dimensions,).
        amplitude: Amplitude parameter (default: 20).
        exponential_decay: Exponential decay parameter (default: 0.2).
        oscillation_frequency: Oscillation frequency parameter (default: 2π).

    Returns:
        Function value at point input_vector.
    """
    num_dimensions = len(input_vector)
    term1 = -amplitude * np.exp(-exponential_decay * np.sqrt(np.sum(input_vector**2) / num_dimensions))
    term2 = -np.exp(np.sum(np.cos(oscillation_frequency * input_vector)) / num_dimensions)
    ackley_value = term1 + term2 + amplitude + np.exp(1)
    return ackley_value


def griewank(input_vector):
    """Griewank function - multimodal optimization benchmark.

    Args:
        input_vector: Input vector with shape (n_dimensions,).

    Returns:
        Function value at point input_vector.
    """
    num_dimensions = len(input_vector)
    term1 = np.sum(input_vector**2) / 4000
    term2 = 1
    for dimension_idx in range(num_dimensions):
        term2 *= np.cos(input_vector[dimension_idx] / np.sqrt(dimension_idx + 1))
    griewank_value = term1 - term2 + 1
    return griewank_value


def weierstrass(input_vector, amplitude=0.5, frequency_multiplier=3, max_summation_index=20):
    """Weierstrass function - fractal optimization benchmark.

    Args:
        input_vector: Input vector with shape (n_dimensions,).
        amplitude: Amplitude parameter (default: 0.5).
        frequency_multiplier: Frequency multiplier parameter (default: 3).
        max_summation_index: Maximum summation index (default: 20).

    Returns:
        Function value at point input_vector.
    """
    num_dimensions = len(input_vector)
    weierstrass_value = 0
    for dimension_idx in range(num_dimensions):
        for summation_idx in range(max_summation_index + 1):
            weierstrass_value += (amplitude**summation_idx) * np.cos(2 * np.pi * (frequency_multiplier**summation_idx) * (input_vector[dimension_idx] + 0.5))
        for summation_idx in range(max_summation_index + 1):
            weierstrass_value -= (amplitude**summation_idx) * np.cos(2 * np.pi * (frequency_multiplier**summation_idx) * 0.5)
    return weierstrass_value


def shekel(input_vector, num_local_minima=10):
    """Shekel function - multimodal optimization benchmark with variable local minima.

    Args:
        input_vector: Input vector with shape (n_dimensions,).
        num_local_minima: Number of local minima (default: 10).

    Returns:
        Function value at point input_vector (negated for minimization).
    """
    num_dimensions = len(input_vector)
    matrix_A = np.random.rand(num_local_minima, num_dimensions) * 10  # random matrix for each run
    vector_C = np.random.rand(num_local_minima) * 10
    shekel_value = 0
    for minima_idx in range(num_local_minima):
        shekel_value -= 1 / (vector_C[minima_idx] + np.sum((input_vector - matrix_A[minima_idx]) ** 2))
    return -shekel_value


def hartmann6(input_vector):
    """Hartmann 6-dimensional function - optimization benchmark.

    Args:
        input_vector: Input vector with shape (6,).

    Returns:
        Function value at point input_vector.
    """
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
    return -hartmann6_value
