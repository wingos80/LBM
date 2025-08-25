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

from settings import *
from core.airfoil import *
import logging, jax, jax.numpy as jnp

logger = logging.getLogger(__package__)  # Use module's name as logger name
libname = "JAX" if USE_LIBRARY == "jax" else "NumPy"
logger.info(f"You are using {USE_LIBRARY} on {jax.devices()[0].device_kind}")


## Initial conditions, TODO make as a class with static methods?
def flow_taylorgreen(
    t: float = 0.0, tau: float = 1.0, rho0: float = 1.0, u_max: float = 0.2
) -> jax.Array:
    nu = (2 * tau - 1) / 6
    x = jnp.arange(XMAX) + 0.5
    y = jnp.arange(YMAX) + 0.5
    [X, Y] = jnp.meshgrid(x, y)
    kx = 2 * jnp.pi / XMAX
    ky = 2 * jnp.pi / YMAX
    td = 1 / (nu * (kx * kx + ky * ky))

    u0 = jnp.array(
        [
            -u_max
            * jnp.sqrt(ky / kx)
            * jnp.cos(kx * X)
            * jnp.sin(ky * Y)
            * jnp.exp(-t / td),
            u_max
            * jnp.sqrt(kx / ky)
            * jnp.sin(kx * X)
            * jnp.cos(ky * Y)
            * jnp.exp(-t / td),
        ]
    )
    P = (
        -0.25
        * rho0
        * u_max
        * u_max
        * ((ky / kx) * jnp.cos(2 * kx * X) + (kx / ky) * jnp.cos(2 * ky * Y))
        * jnp.exp(-2 * t / td)
    )
    rho = rho0 + 3 * P
    u = jnp.zeros((XMAX, YMAX, 2))
    u[:, :, 0] = u0[0, :, :].T
    u[:, :, 1] = u0[1, :, :].T

    f = calculate_f_eq(rho.T, u)
    return f


def flow_up() -> jax.Array:
    rho = jnp.ones((XMAX, YMAX))
    u = jnp.ones((XMAX, YMAX, 2))
    u[:, :, 0] = 0  # make ux 0
    u[:, :-3, 1] = 0  # make ux 0
    u[:, -2:, 1] = 0  # make ux 0
    f = calculate_f_eq(rho, u)
    return f


def flow_right() -> jax.Array:
    vel = INFLOW_VEL
    rho = jnp.ones((XMAX, YMAX))
    # rho[-3,:] += 1
    # u = 0.15*jnp.ones((XMAX,YMAX,2))
    u = jnp.stack((vel * jnp.ones((XMAX, YMAX, 1)), jnp.zeros((XMAX, YMAX, 1))), axis=2)
    # u.at[:,:,1].set(0)  # make uy 0
    # u[:-3,:,0] = 0  # make ux 0
    # u[-2:,:,0] = 0  # make ux 0
    f = calculate_f_eq(rho, u)
    return f


def flow_random() -> jax.Array:
    spread = 0.1
    rho = jnp.ones((XMAX, YMAX))
    u = jnp.ones((XMAX, YMAX, 2))
    u[:, :, 0] = 0  # make ux 0
    u[:, :-3, 1] = 0  # make ux 0
    u[:, -2:, 1] = 0  # make ux 0
    u += spread * (jnp.random.rand(XMAX, YMAX, 2) - 0.5)
    f = calculate_f_eq(rho, u)
    return f


## Moments of f
def calculate_rho(f: jax.Array) -> jax.Array:
    """
    :param f: population, particle vector field, how many particles moving in each direction per cell
    :returns: density, scalar field, particles per cell
    """
    return jnp.sum(f, axis=2)


def calculate_momentum(f: jax.Array, c: jax.Array | None = None) -> jax.Array:
    """
    :param f: population, particle vector field, how many particles moving in each direction per cell
    :param c: basis velocity components
    :returns: momentum, vector field, momentum per cell
    """
    if c is not None:
        ## Generic implementation
        cx, cy = c[0], c[1]
        momentum_x = jnp.sum(f * cx, axis=2)
        momentum_y = jnp.sum(f * cy, axis=2)
    else:
        ## D2Q9 implementation
        momentum_x = (f[:, :, 1] + f[:, :, 5] + f[:, :, 8]) - (
            f[:, :, 3] + f[:, :, 6] + f[:, :, 7]
        )
        momentum_y = (f[:, :, 2] + f[:, :, 5] + f[:, :, 6]) - (
            f[:, :, 4] + f[:, :, 7] + f[:, :, 8]
        )
    return jnp.stack((momentum_x, momentum_y), axis=2)


def calculate_velocity(momentum: jax.Array, rho: jax.Array) -> jax.Array:
    """
    :param momentum: momentum, vector field, how many particles moving in each direction per cell
    :param rho: density, scalar field, particles per cell
    :return: velocity
    """
    ux = momentum[:, :, 0] / rho
    uy = momentum[:, :, 1] / rho
    u = jnp.stack((ux, uy), axis=2)
    return u


def calculate_f_eq(rho: jax.Array, u: jax.Array) -> jax.Array:
    """
    Returns the equillibrium population/particle field.
    Implementation is specific for D2Q9, meaning that the weights of
    D2Q9's velocity set are used and the generic equation for f_eq is
    expanded using those weights, resulting in the hard coded coefficients
    of rho.
    TODO, add generic implementation
    :param rho: density, scalar field, particles per cell
    :param u: velocity, vector field, how many particles moving in each direction per cell
    :return: equillibrium population
    """
    ## D2Q9 implementation
    ux, uy = u[:, :, 0].squeeze(), u[:, :, 1].squeeze()

    # First pre compute commonly used terms
    rho_1 = 2 * rho / 9
    rho_2 = rho / 18
    rho_3 = rho / 36

    u2 = ux**2 + uy**2
    ux2 = ux**2
    uy2 = uy**2
    uxuy = ux * uy

    # Use the pre-computed terms to compute equillibrium distributions
    f_eq_0 = rho_1 * (2 - 3 * u2)
    f_eq_1 = rho_2 * (2 + 6 * ux + 9 * ux2 - 3 * u2)
    f_eq_2 = rho_2 * (2 + 6 * uy + 9 * uy2 - 3 * u2)
    f_eq_3 = rho_2 * (2 - 6 * ux + 9 * ux2 - 3 * u2)
    f_eq_4 = rho_2 * (2 - 6 * uy + 9 * uy2 - 3 * u2)
    f_eq_5 = rho_3 * (1 + 3 * (ux + uy) + 9 * uxuy + 3 * u2)
    f_eq_6 = rho_3 * (1 - 3 * (ux - uy) - 9 * uxuy + 3 * u2)
    f_eq_7 = rho_3 * (1 - 3 * (ux + uy) + 9 * uxuy + 3 * u2)
    f_eq_8 = rho_3 * (1 + 3 * (ux - uy) - 9 * uxuy + 3 * u2)
    f_eq = jnp.stack(
        (f_eq_0, f_eq_1, f_eq_2, f_eq_3, f_eq_4, f_eq_5, f_eq_6, f_eq_7, f_eq_8), axis=2
    )

    return f_eq


## Simulation steps
def collision_step(
    f: jax.Array, f_eq: jax.Array, dt: float = 1.0, tau: float = 1.0
) -> jax.Array:
    """
    BGK collision step
    :param f: array containing the current population
    :param f_eq: array containing the equillibrium population
    :retunr: population after collision
    """
    return f * (1 - dt / tau) + dt / tau * f_eq


def streaming_step(f: jax.Array) -> jax.Array:
    """
    Particle streamed according to their velocities
    """
    # Streaming
    f_streamed = jnp.stack(
        [
            jnp.roll(jnp.roll(f[:, :, i], c[0, i], axis=0), c[1, i], axis=1)
            for i in range(9)
        ],
        axis=2,
    )
    # args = (f, c)
    # def body_fun(i, val):
    #     """
    #     i: iteration
    #     val: tuple with first element being f, second being c"""
    #     f = val[0]
    #     c = val[1]
    #     f = f.at[:,:,i].set(jnp.roll(jnp.roll(f[:,:,i], c[0, i], axis=0), c[1, i], axis=1))
    #     return f, c
    # out_args = jax.lax.fori_loop(0, 9, body_fun, args)
    # f_streamed = out_args[0]
    return f_streamed


def BC_solids(f: jax.Array, solids: jax.Array) -> jax.Array:
    """
    Boundary condition for solids
    """
    _solids = jnp.repeat(
        jnp.expand_dims(solids, axis=2), 9, axis=2
    )  # expanding the solid matrix to be same shape as f:(XMAX, YMAX, 9)
    b = jnp.where(_solids, f, 0)  # select all particles inside solids
    b = b[:, :, [0, 3, 4, 1, 2, 7, 8, 5, 6]]  # flip the velocities of these particles
    f = jnp.where(_solids, b, f)  # insert them back into f
    return f


## Simulation function
@jax.jit
def update(f: jax.Array, solids: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Simulate one step forward in time
    """
    # Prescribe outflow BC
    f = f.at[-1, :, [3, 6, 7]].set(f[-2, :, [3, 6, 7]])

    rho = calculate_rho(f)
    momentum = calculate_momentum(f)
    u = calculate_velocity(momentum, rho)

    # Zou/He inflow scheme part 1
    u = u.at[0, 1:-1, 0].set(INFLOW_VEL)  # set inflow vx to be 0.1
    u = u.at[0, 1:-1, 1].set(0)  # set inflow vy to be 0
    rho_vert_fs = calculate_rho(
        f[:1, :, [0, 2, 4]]
    )  # :1 in the first index to keep shape
    rho_left_fs = calculate_rho(
        f[:1, :, [3, 6, 7]]
    )  # :1 in the first index to keep shape
    rho = rho.at[:1, :].set((rho_vert_fs + 2 * rho_left_fs) / (1 - u[:1, :, 0]))

    # find equillibrium population
    f_eq = calculate_f_eq(rho, u)

    # Zou/He inflow scheme part 1
    f = f.at[0, :, [1, 5, 8]].set(f_eq[0, :, [1, 5, 8]])

    # perform collision, streaming, and solid BCs
    f = collision_step(f, f_eq, dt=DT, tau=TAU)
    f = streaming_step(f)
    f = BC_solids(f, solids)
    return f, u
