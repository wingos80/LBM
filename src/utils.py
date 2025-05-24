"""
File containing helper functions
"""

import settings

USE_LIBRARY, USE_DEVICE = settings.USE_LIBRARY, settings.USE_DEVICE
IMG_TYPE = settings.IMG_TYPE
VIDEO_LENGTH = settings.VIDEO_LENGTH
FRAME_TIME = settings.FRAME_TIME

# Set the numpy backend for jax if jax is being used
if USE_LIBRARY == "jax":
    from core.jnpfuncs import *
elif USE_LIBRARY == "numpy":
    from core.npfuncs import *
import time, os, numpy as np, jax.numpy as jnp, matplotlib.pyplot as plt
from jax import Array as JaxArray
from numpy import ndarray as npndarray
from tqdm import tqdm


class CatchTime:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = time.perf_counter() - self.start
        self.readout = f"Time: {self.time:.3f} seconds"
        print(self.readout)


class Recorder:
    def __init__(self):
        print(
            f"Recording simulation, saving frames to ./recording/{USE_DEVICE}/{IMG_TYPE}/"
        )
        self.start_time = time.time()
        self.elapsed_time = time.time() - self.start_time
        self.last_saved = self.elapsed_time
        self.frame = 0

    def check_terminate(self):
        if self.elapsed_time < VIDEO_LENGTH:
            self.elapsed_time = time.time() - self.start_time
            self.time_since_last_saved = self.elapsed_time - self.last_saved
            print(
                f"{self.elapsed_time:.3f} s: Time since last saved: {self.time_since_last_saved:.3f} s",
                end="\r",
            )
            return True
        else:
            return False

    def record(self, u, solids):
        if self.time_since_last_saved > FRAME_TIME:
            self.frame += 1
            vorticity = get_vorticity(u, solids)
            plot(
                vorticity,
                frame=self.frame,
                save_dir=USE_DEVICE,
                img_type=IMG_TYPE,
                save=True,
            )
            self.last_saved = time.time() - self.start_time


def create_solids() -> JaxArray | npndarray:
    solids = jnp.logical_or(Y == 0, Y == YMAX - 1)  # solid top and bottom walls
    circle_center = XMAX / 4, YMAX / 2 + 4
    circle_radius = YMAX / 10
    # solids += (X - circle_center[0])**2 + (Y - circle_center[1])**2 < (circle_radius)**2

    airfoil_le = XMAX // 6, YMAX // 2  # location of airfoil leading edge
    chord_length = XMAX // 4  # size of airfoil, scaling factor
    n_pts = 10000
    x = np.linspace(0, 1, n_pts)
    U_n, L_n = construct_airfoil(x)
    rotate = -10 * np.pi / 180
    R = np.array([[np.cos(rotate), -np.sin(rotate)], [np.sin(rotate), np.cos(rotate)]])

    U, L = chord_length * U_n, chord_length * L_n
    step = n_pts // chord_length  # assuming n_pts > chord length

    # rotate the airfoil
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

    solids_2 = jnp.full((XMAX, YMAX), False)
    solids_2 = solids_2.at[:chord_length, :thickness].set(hold_array)
    solids_2 = jnp.roll(
        jnp.roll(solids_2, airfoil_le[0], axis=0), airfoil_le[1], axis=1
    )
    # solids += solids_2
    if USE_LIBRARY == "jax":
        return solids_2
    elif USE_LIBRARY == "numpy":
        return np.array(solids_2)


def get_vorticity(u: npndarray | JaxArray, solids: npndarray | JaxArray):
    vorticity = (np.roll(u[:, :, 0], -1, axis=1) - np.roll(u[:, :, 0], 1, axis=1)) - (
        np.roll(u[:, :, 1], -1, axis=0) - np.roll(u[:, :, 1], 1, axis=0)
    )
    vorticity[solids] = np.nan
    return vorticity.T


def plot(array: npndarray, frame=0, save_dir="cpu", img_type="jpg", save: bool = False):
    plt.cla()
    plt.imshow(array, cmap="bwr")
    plt.clim(-0.01, 0.01)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.grid()
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    ax.set_aspect("equal")
    if save:
        dirr = f".recordings/{save_dir}/{img_type}"
        if not os.path.exists(dirr):
            os.makedirs(dirr)
        name = f"{dirr}/frame_{frame}.{img_type}"
        plt.savefig(name)
    else:
        plt.pause(0.0001)
