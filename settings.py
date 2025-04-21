"""
All the simulations settings are defined here
"""
# Fluid settings
TAU  = 0.5075
CS2  = 1/1

# Simulation settings
XMAX       = 400
YMAX       = 150
MSIZE      = XMAX*YMAX
DT         = 1
TIME       = 720
INFLOW_VEL = 0.02

# Select library and device
USE_LIBRARY = "jax"
USE_DEVICE  = "cpu"

# Plotting options
PLOT_EVERY = 150  # render every n-th frame

# Recording options
RECORD       = False  # toggle recording
VIDEO_LENGTH = 10  # how many seconds to record for
VIDEO_FPS    = 15  # frames per second
IMG_TYPE     = "png"
FRAME_TIME   = 1/VIDEO_FPS  # frame time

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
