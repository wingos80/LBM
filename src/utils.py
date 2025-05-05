"""
File containing helper functions
"""

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import os
import time
from settings import *
if USE_LIBRARY == "jax":
    from core.jnpfuncs import *
elif USE_LIBRARY == "numpy":
    from core.npfuncs import *


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


def get_vorticity(u: np.ndarray | jnp.ndarray, solids: np.ndarray | jnp.ndarray):
    vorticity = (np.roll(u[:, :, 0], -1, axis=1) - np.roll(u[:, :, 0], 1, axis=1)) - (
        np.roll(u[:, :, 1], -1, axis=0) - np.roll(u[:, :, 1], 1, axis=0)
    )
    vorticity[solids] = np.nan
    return vorticity.T


def plot(
    array: np.ndarray, frame=0, save_dir="cpu", img_type="jpg", save: bool = False
):
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
