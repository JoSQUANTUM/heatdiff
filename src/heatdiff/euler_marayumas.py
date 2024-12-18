"""Euler-Marayuma methods for several diffusion equations written as Stochastic Differential Equations"""

import numpy as np


def reverse_wiener_1d(T, N, seed=42):
    """
     Euler-Marayuma algorithm for the reversed Wiener process, V_{t} = W_{T-t} - W_{T}

     Parameters:
    - T (int): The terminal time.
    - N (int): Number of steps.

    Returns:
    - V: An array of simulated values of the process V_t at times from T to 0 with step dt.
    - X: An array of simulated values of the process W_t at times from 0 to T with step dt.
    """
    dt = T / N
    W = np.zeros(N + 1)
    V = np.zeros(N + 1)
    np.random.seed(seed)  # Is this the best way to set a random seed?

    for i in range(1, N + 1):
        W[i] = (
            W[i - 1] + np.sqrt(2 * dt) * np.random.normal()
        )  # step_size = i-(i-1) = 1

    for i in range(N + 1):
        V[N - i] = W[N - i] - W[N]  # changed from i to N-i

    return V, W


def wiener_1d(T, N, seed=4):
    """
     Euler-Marayuma algorithm for the reversed Wiener process, X_{t} = sqrt{2}.W_{T}

     Parameters:
    - T (int): The terminal time.
    - N (int): Number of steps.

    Returns:
    - W: An array of simulated values of the process X_t at times from 0 to T with
    step dt.
    """
    dt = T / N
    W = np.zeros(N + 1)
    np.random.seed(seed)

    for i in range(1, N + 1):
        W[i] = W[i - 1] + np.sqrt(2 * dt) * np.random.normal()

    return W


def reverse_wiener_img(image, T, N, seed=42):
    """
     Euler-Marayuma algorithm for the reversed Wiener process, V_{t} = W_{T-t} - W_{T}
     for images

     Parameters:
    - T (int): The terminal time.
    - N (int): Number of steps.

    Returns:
    - V: A matrix of simulated values of the process V_t at times from T to 0 with step dt.
    - X: An array of simulated values of the process X_t at times from 0 to T with step dt.
    """
    dt = T / N
    h, w = image.shape
    X = np.zeros((N + 1, h, w))
    V = np.zeros((N + 1, h, w))
    X[0] = image

    np.random.seed(seed)

    for i in range(1, N + 1):
        dx = np.random.normal(loc=0.0, scale=1.0, size=(h, w))
        X[i] = X[i - 1] + np.sqrt(2 * dt) * dx  # We could also set loc = img.mean()

    for i in range(N + 1):

        V[N - i] = X[N - i] - X[N]

    return V, X


def reverse_diffusion(image, T, N, seed=42):
    """
     Euler-Marayuma algorithm for the reversed diffuion process a la denoising diffusion models with respect
     to the standard Wiener process

     Parameters:
    - T (int): The terminal time.
    - N (int): Number of steps.

    Returns:
    - V: A matrix of simulated values of the reversed diffusion process at times from
     T to 0 with step dt.
    - X: A matrix of simulated values of the Wiener at times from 0 to T with step dt.
    """

    dt = T / N
    h, w = image.shape
    X = np.zeros((N + 1, h, w))
    X[0] = image
    mean = image.mean()
    V = np.zeros((N + 1, h, w))

    np.random.seed(seed)

    for i in range(1, N + 1):
        dx = np.random.normal(
            loc=mean, scale=1.0, size=(h, w)
        )  # diffusion models set the initial mean=image mean
        X[i] = X[i - 1] + np.sqrt(2 * dt) * dx  # We could also set loc = img.mean()

    V[N] = X[N]
    for i in range(N - 1, -1, -1):

        V[i] = X[i + 1] + dt * ((X[i] - mean) / (i + 1)) - np.sqrt(2 * dt) * dx  #

    return V, X


def reverse_ou(image, theta, T, N, seed=42):
    """
     Euler-Marayuma algorithm for the reversed diffuion process a la denoising diffusion models with respect to
     the Ornstein-Uhlenbeck process

     Parameters:
    - T: The terminal time.
    - N: Number of steps.

    Returns:
    - V (int): A matrix of simulated values of the reversed diffusion process at times from T to 0 with step dt.
    - X (int): A matrix of simulated values of the Ornstein-Uhlenbeck process at times from 0 to T with step dt.
    """

    dt = T / N
    h, w = image.shape
    X = np.zeros((N + 1, h, w))
    X[0] = image
    mean = image.mean()
    std = image.std()
    V = np.zeros((N + 1, h, w))

    np.random.seed(seed)

    for i in range(1, N + 1):
        dx = np.random.normal(
            loc=mean, scale=std, size=(h, w)
        )  # diffusion models set the initial mean to be the image mean
        X[i] = (
            X[0] * np.exp(-theta * i)
            + (std / np.sqrt(2 * theta)) * np.sqrt(1 - np.exp(-2 * theta * i) * dt) * dx
        )  # formal forward solution

    V[N] = X[N]
    for i in range(N - 1, -1, -1):
        V[i] = (
            X[i + 1]
            + dt
            * (
                (X[i + 1] - X[0] * np.exp(-2 * theta * (i + 1)))
                / (1 - np.exp(-2 * theta * (i + 1)))
            )
            - np.sqrt(2 * dt) * dx
        )

    return V, X
