# Lattice Boltzman Method

LBM simulation built in python using JAX and numpy. 

## Run on GPU or CPU

Switch between using gpu and cpu in the settings file:
- `USE_DEVICE= CPU` to run on CPU
- `USE_DEVICE= GPU` to run on GPU
Note however, that JAX currently [only offers GPU support on Linux](https://jax.readthedocs.io/en/latest/installation.html#:~:text=libtpu_releases.html-,Supported%20platforms,no,-CPU), so it is not possible to use the GPU if you are running on windows.

# LBM ran on GPU vs CPU

On GPU:

![](https://github.com/wingos80/LBM/blob/main/resources/gpu_lbm.gif)

On CPU:

![](https://github.com/wingos80/LBM/blob/main/resources/cpu_lbm.gif)

