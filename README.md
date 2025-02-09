# Lattice Boltzman Method

LBM simulation built in python using JAX and numpy. 

## Requirements

This code was developed with `Python=3.10`, tested on Windows and WSL.

Install required libraries:

`pip install -r requirements.txt`

If you're running on Linux and wish to use the GPU, install JAX with all the cuda helpers:

`pip install -U "jax[cuda12]"`

## Settings.py options

Switch between rendering a plot vs recording renders to disk:
- `RECORD = False` for rendering the simulation
- `RECORD = True` for saving simulation to disk

Switch between the Numpy and JAX implementations:
- `USE_LIBRARY = NP` for Numpy
- `USE_LIBRARY = JAX` for JAX

Switch between using gpu and cpu:
- `USE_DEVICE = CPU` for CPU
- `USE_DEVICE = GPU` for GPU

Note that JAX currently [only offers GPU support on Linux](https://jax.readthedocs.io/en/latest/installation.html#:~:text=libtpu_releases.html-,Supported%20platforms,no,-CPU), so it is not possible to use the GPU if you are running on windows. Numpy on the otherhand, does not support GPU support on any platform.

## LBM ran on GPU vs CPU

On GPU:

![](https://github.com/wingos80/LBM/blob/main/resources/gpu_lbm.gif)

On CPU:

![](https://github.com/wingos80/LBM/blob/main/resources/cpu_lbm.gif)

