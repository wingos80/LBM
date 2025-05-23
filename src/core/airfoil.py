"""
MIT License

Copyright (c) 2025 Wing Yin Chan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from jax.typing import ArrayLike
import logging, matplotlib.pyplot as plt, numpy as np, jax

logger = logging.getLogger(__package__)  # ← Uses module's name as logger name

MAX_CAMBER = 4  # maximum camber as percentage of chord
MAX_CAMBER_P = 0.4  # location of maximum camber in fraction of chord length
THICKNESS = 10  # maximum thickness of airfoil as percentage of chord

M = MAX_CAMBER / 100
P = MAX_CAMBER_P
T = THICKNESS / 100


def construct_yc(x: ArrayLike) -> jax.Array | np.ndarray:
    """
    Function returning y coordinate of the
    mean camber line as a function of x coordinate.

    Raises:
        ValueError: Value of `x` ∉ [0, 1]
    """
    if np.any(x < 0) or np.any(x > 1):
        raise ValueError('"x" has to be between 0 and 1"')

    y = np.where(
        x <= P,
        M * (2 * P * x - x**2) / P**2,
        M * ((1 - 2 * P) + 2 * P * x - x**2) / (1 - P) ** 2,
    )
    return y


def construct_yt(x: ArrayLike) -> jax.Array | np.ndarray:
    """
    Function returning the thickness of the airfoil,
    that is distance from the airfoil to the camberline,
    as a function of x coordinate.

    Raises:
        ValueError: Value of `x` ∉ [0, 1]
    """
    if np.any(x < 0) or np.any(x > 1):
        raise ValueError('"x" has to be between 0 and 1"')

    yt = (
        5
        * T
        * (
            0.2969 * np.sqrt(x)
            - 0.126 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )
    )
    return yt


def construct_theta(x: ArrayLike) -> jax.Array | np.ndarray:
    """
    Function returning the angle of rotation for yt as a function
    of x coordinate

    Raises:
        ValueError: Value of `x` ∉ [0, 1]
    """
    if np.any(x < 0) or np.any(x > 1):
        raise ValueError('"x" has to be between 0 and 1"')

    theta = np.where(
        x <= P,
        np.arctan(2 * M * (P - x) / P**2),
        np.arctan(2 * M * (P - x) / (1 - P) ** 2),
    )
    return theta


def construct_airfoil(x: ArrayLike) -> tuple[jax.Array | np.ndarray, jax.Array | np.ndarray]:
    """
    Returns upper and lower airfoil coordinates
    """
    y_c = construct_yc(x)
    y_t = construct_yt(x)
    theta = construct_theta(x)

    x_U = x - y_t * np.sin(theta)
    x_L = x + y_t * np.sin(theta)

    y_U = y_c + y_t * np.cos(theta)
    y_L = y_c - y_t * np.cos(theta)

    U = np.stack([x_U, y_U], axis=0)
    L = np.stack([x_L, y_L], axis=0)
    return U, L


if __name__ == "__main__":
    import jax.numpy as jnp
    import jax

    XMAX = 600
    YMAX = 300
    x = np.arange(XMAX)
    y = np.arange(YMAX)
    X, Y = np.meshgrid(x, y)
    X, Y = X.T, Y.T

    airfoil_le = XMAX // 6, YMAX // 2  # location of airfoil leading edge
    chord_length = XMAX // 4  # size of airfoil, scaling factor
    n_pts = 10000
    x = np.linspace(0, 1, n_pts)
    U_n, L_n = construct_airfoil(x)

    rotate = -10 * np.pi / 180
    R = np.array([[np.cos(rotate), -np.sin(rotate)], [np.sin(rotate), np.cos(rotate)]])

    U, L = chord_length * U_n, chord_length * L_n
    step = n_pts // chord_length  # assuming n_pts > chord length
    U = R @ U
    L = R @ L

    top = U[1].max()
    bottom = L[1].min()

    U[1] += abs(bottom)
    L[1] += abs(bottom)

    thickness = int(np.ceil(top - bottom))
    hold_array = jnp.full((chord_length, thickness), False)

    for i in range(chord_length):
        location = i * step
        slice_i = jnp.arange(thickness)
        slice_i = jnp.logical_and(slice_i >= L[1, location], slice_i <= U[1, location])

        hold_array = hold_array.at[i].set(slice_i)

    solids = jnp.full((XMAX, YMAX), False)
    solids = solids.at[:chord_length, :thickness].set(hold_array)
    solids = jnp.roll(jnp.roll(solids, airfoil_le[0], axis=0), airfoil_le[1], axis=1)
    plt.imshow(hold_array.T)
    plt.show()
    d = 1
