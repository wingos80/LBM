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

from settings import *
from utils import *


def main():
    ## Initial Conditions
    # f = flow_taylorgreen()
    # f = flow_up()
    f = flow_right()
    solids = create_solids()

    if not RECORD:
        print("Rendering simulation")
        for t in tqdm(range(TIME)):
            f, u = update(f, solids)

            # plot every PLOT_EVERY seconds
            if t % PLOT_EVERY == 0:
                vorticity = get_vorticity(u, solids)
                plot(vorticity)
    elif RECORD:
        recorder = Recorder()
        while recorder.check_terminate():
            f, u = update(f, solids)

            # record simulation frames
            recorder.record(u, solids)

    print("\n")


if __name__ == "__main__":
    with CatchTime() as timer:
        main()
