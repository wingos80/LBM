import matplotlib.pyplot as plt
import numpy as np
import jax
import os
import time

class catchtime:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = time.perf_counter() - self.start
        self.readout = f'Time: {self.time:.3f} seconds'
        print(self.readout)


def plot(array: np.ndarray, frame=0, save_dir="cpu", img_type="jpg", save: bool=False):
    plt.cla()
    plt.imshow(array, cmap='bwr')
    plt.clim(-.1, .1)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect("equal")
    if save:
        dirr = f".recordings/{save_dir}/{img_type}"
        if not os.path.exists(dirr):
            os.makedirs(dirr)
        name = f"{dirr}/frame_{frame}.{img_type}"
        plt.savefig(name)
    else:
        plt.pause(0.0001)