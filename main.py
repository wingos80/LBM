import numpy as np
import os
from settings import *
os.environ["JAX_PLATFORM_NAME"] = USE_DEVICE

from jnpfuncs import *
from utils import *
from tqdm import tqdm

print(f"You are using device: {jax.devices()[0]}")

# TODO have all libarry imports in one place instead of scattered between e.g. main.py, settings.py, utils.py...?
# TODO add saving class/context manager/something to make saving pictures prettier
# TODO redo folder struction
# TODO better solids creation/customization
# TODO add functionality to select D2Q9 or supply own velocity set (change moment and f_eq calcs to generic)

# Pressure difference
calculate_f_eq_out = lambda u_1: calculate_f_eq(rho_out, u_1)[-1,:,:]
calculate_f_eq_in = lambda u_N: calculate_f_eq(rho_in, u_N)[0,:,:]

# Initial Conditions
# f = flow_taylorgreen()
# f = flow_up()
f = flow_right()
solids = create_solids()

@jax.jit
def one_time_march(f: np.ndarray):
    """
    Simulate one step forward in time
    """
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


with catchtime() as timer:
    if not RECORD:
        print("Rendering simulation")
        for t in tqdm(range(TIME)):
            f, u = one_time_march(f)
        
            # plot in real time
            if t%100==0:
                vorticity = (np.roll(u[:,:,0], -1, axis=1) - np.roll(u[:,:,0], 1, axis=1)) - (np.roll(u[:,:,1],-1,axis=0) - np.roll(u[:,:,1],1,axis=0))
                vorticity[solids] = np.nan
                plot(vorticity.T)
    elif RECORD:
        print(f"Recording simulation, saving frames to ./recording/{USE_DEVICE}/{IMG_TYPE}/")
        start_time = time.time()
        elapsed_time = time.time() - start_time
        last_saved = elapsed_time
        frame = 0
        while (elapsed_time < RECORD_TIME):
            elapsed_time = time.time() - start_time
            f, u = one_time_march(f)

            time_since_last_saved = elapsed_time - last_saved
            print(f"{elapsed_time:.3f} s: Time since last saved: {time_since_last_saved:.3f} s", end="\r")
            # save frames every fps seconds
            if time_since_last_saved > FT:
                    frame += 1
                    vorticity = (np.roll(u[:,:,0], -1, axis=1) - np.roll(u[:,:,0], 1, axis=1)) - (np.roll(u[:,:,1],-1,axis=0) - np.roll(u[:,:,1],1,axis=0))
                    vorticity[solids] = np.nan
                    plot(vorticity.T, frame=frame, save_dir=USE_DEVICE, img_type=IMG_TYPE, save=True)
                    last_saved = time.time() - start_time
        print("\n")

