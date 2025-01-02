"""
MIT License

Copyright (c) 2025 Wing Yin Chan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from settings import *
if USE_LIBRARY==JAX:
    from jnpfuncs import *
elif USE_LIBRARY==NP:
    from npfuncs import *
from utils import *
from tqdm import tqdm

if __name__ == "__main__":
    # Pressure difference?
    calculate_f_eq_out = lambda u_1: calculate_f_eq(rho_out, u_1)[-1,:,:]
    calculate_f_eq_in = lambda u_N: calculate_f_eq(rho_in, u_N)[0,:,:]

    # Initial Conditions
    # f = flow_taylorgreen()
    # f = flow_up()
    f = flow_right()
    solids = create_solids()

    with catchtime() as timer:
        start_time = time.time()
        elapsed_time = time.time() - start_time
        if not RECORD:
            print("Rendering simulation")
            for t in tqdm(range(TIME)):
                f, u = one_time_march(f, solids)
                
                # plot every PLOT_EVERY seconds
                if t % PLOT_EVERY == 0:
                    vorticity = (np.roll(u[:,:,0], -1, axis=1) - np.roll(u[:,:,0], 1, axis=1)) - (np.roll(u[:,:,1],-1,axis=0) - np.roll(u[:,:,1],1,axis=0))
                    vorticity[solids] = np.nan
                    plot(vorticity.T)
        elif RECORD:
            print(f"Recording simulation, saving frames to ./recording/{USE_DEVICE}/{IMG_TYPE}/")
            last_saved = elapsed_time
            frame = 0
            while (elapsed_time < RECORD_TIME):
                elapsed_time = time.time() - start_time
                f, u = one_time_march(f, solids)

                time_since_last_saved = elapsed_time - last_saved
                print(f"{elapsed_time:.3f} s: Time since last saved: {time_since_last_saved:.3f} s", end="\r")
                # save frames every FT seconds
                if time_since_last_saved > R_FT:
                        frame += 1
                        vorticity = (np.roll(u[:,:,0], -1, axis=1) - np.roll(u[:,:,0], 1, axis=1)) - (np.roll(u[:,:,1],-1,axis=0) - np.roll(u[:,:,1],1,axis=0))
                        vorticity[solids] = np.nan
                        plot(vorticity.T, frame=frame, save_dir=USE_DEVICE, img_type=IMG_TYPE, save=True)
                        last_saved = time.time() - start_time
            print("\n")

