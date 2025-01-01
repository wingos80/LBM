import jax
import matplotlib.pyplot as plt
import numpy as np
from settings import *
from jnpfuncs import *
from tqdm import tqdm

# TODO redo folder struction
# TODO better solids creation/customization
# TODO add functionality to select D2Q9 or supply own velocity set (change moment and f_eq calcs to generic)
# https://github.com/Safiullah-Rahu/Lattice-Boltzmann-Simulation/blob/main/Boltzmann_simu.py

# Pressure difference
calculate_f_eq_out = lambda u_1: calculate_f_eq(rho_out, u_1)[-1,:,:]
calculate_f_eq_in = lambda u_N: calculate_f_eq(rho_in, u_N)[0,:,:]

# Initial Conditions
# f = flow_taylorgreen()
# f = flow_up()
f = flow_right()
solids = create_solids()

@jax.jit
def one_time_march(f):
    # # # absorbing left and right walls
    # # f[0,:,[1,5,8]] = f[1,:,[1,5,8]]
    # # f[-1,:,[3,6,7]] = f[-2,:,[3,6,7]]
    rho = calculate_rho(f)
    momentum = calculate_momentum(f)
    u = calculate_velocity(momentum, rho)

    # f_eq_in = calculate_f_eq_out(u[0,:,:].reshape(-1,YMAX,2))
    # f_eq_out = calculate_f_eq_out(u[-1,:,:].reshape(-1,YMAX,2))
    f_eq = calculate_f_eq(rho, u)
    f = collision_step(f, f_eq, dt=DT, tau=TAU)
    # f_star[0,:,:] = f_eq_in + f_star[-2,:,:] - f_eq[-2,:,:]
    # f_star[-1,:,:] = f_eq_out + f_star[1,:,:] - f_eq[1,:,:]
    f = streaming_step(f)
    f = BC_solids(f, solids)
    return f, u


history = []
cpus = jax.devices()
with catchtime() as timer:
    for t in tqdm(range(TIME)):
        f, u = one_time_march(f)
        
        # plot in real time - color 1/2 particles blue, other half red
        if t%20==0:
            plt.cla()
            vorticity = (np.roll(u[:,:,0], -1, axis=1) - np.roll(u[:,:,0], 1, axis=1)) - (np.roll(u[:,:,1],-1,axis=0) - np.roll(u[:,:,1],1,axis=0))
            # vorticity = u[:,:,0]**2 + u[:,:,1]**2
            # cmap = plt.cm.bwr
            # cmap.set_bad('black')
            # arrowsx, arrowsy = u[:,:,0], u[:,:,1]
            # arrowsx[solids] = np.nan
            # arrowsy[solids] = np.nan
            # plt.quiver(X, Y, arrowsx, arrowsy)
            vorticity[solids] = np.nan
            plt.imshow(vorticity.T, cmap='bwr')
            plt.clim(-.1, .1)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect("equal")
            plt.pause(0.0001)