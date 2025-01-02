# Settings
JAX = "jax"
NP = "numpy"
GPU = "gpu"
CPU = "cpu"

# Select library and device
USE_LIBRARY = JAX
USE_DEVICE  = GPU

# Grid size
XMAX  = 400
YMAX  = 100
MSIZE = XMAX*YMAX

# Control variables
TAU  = 0.53
DT   = 1
CS2  = 1/1
TIME = 1000

# Plotting options
PLOT_EVERY = 100  # render every n-th frame

# Recording options
RECORD      = False  # toggle recording
RECORD_TIME = 5  # how many seconds to record for
R_FPS       = 15  # frames per second
R_FT        = 1/R_FPS  # frame time
IMG_TYPE    = "png"

# Pressure difference
DP      = 0.1
P_OUT   = 1
rho_out = P_OUT/CS2
p_in    = P_OUT + DP
rho_in  = p_in/CS2

# D2Q9 velocity set
import numpy as np
ws = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
c = np.array([[0, 1, 0, -1,  0, 1, -1, -1,  1],  # velocities, x components
              [0, 0, 1,  0, -1, 1,  1, -1, -1]]) # velocities, y components

x = np.arange(XMAX)
y = np.arange(YMAX)
X,Y = np.meshgrid(x,y)
X,Y = X.T, Y.T


class COLOR:
   PURPLE = '\033[1;35;48m'
   CYAN = '\033[1;36;48m'
   BOLD = '\033[1;37;48m'
   BLUE = '\033[1;34;48m'
   GREEN = '\033[1;32;48m'
   YELLOW = '\033[1;33;48m'
   RED = '\033[1;31;48m'
   BLACK = '\033[1;30;48m'
   UNDERLINE = '\033[4;37;48m'
   END = '\033[1;37;0m'
