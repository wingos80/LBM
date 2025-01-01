# Grid size
XMAX = 400
YMAX = 100
MSIZE = XMAX*YMAX

# Control variables
TAU = 0.53
DT = 1
CS2 = 1/1
TIME = 16000

# Pressure difference
DP = 0.1
P_OUT = 1
rho_out = P_OUT/CS2
p_in = P_OUT + DP
rho_in = p_in/CS2

# D2Q9 velocity set
import numpy as np
ws = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
c = np.array([[0, 1, 0, -1,  0, 1, -1, -1,  1],  # velocities, x components
              [0, 0, 1,  0, -1, 1,  1, -1, -1]]) # velocities, y components

x = np.arange(XMAX)
y = np.arange(YMAX)
X,Y = np.meshgrid(x,y)
X,Y = X.T, Y.T

from time import perf_counter

class catchtime:
    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        self.readout = f'Time: {self.time:.3f} seconds'
        print(self.readout)