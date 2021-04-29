import pandas as pd
import numpy as np
from tkinter import filedialog, Tk
from pathlib import Path
import matplotlib.pyplot as plt

root = Tk()
OUTPUT_DIR = Path('../output/')
filenames = filedialog.askopenfilenames(initialdir=OUTPUT_DIR)
root.destroy()

if len(filenames) == 1:
    data = pd.read_csv(Path(filenames[0]), comment='#', skipinitialspace=True)
    try:
        # CONFLOGS.
        plt.figure()
        plt.plot(data['t'] - data['t'][0], data['ni_ac'], label='ni_ac')
        plt.plot(data['t'] - data['t'][0], data['ni_los'], label='ni_los')
        plt.legend()
        plt.xlabel('Time [s]')
    except KeyError:
        pass
    try:
        # FLSTLOGS.
        print(f'Mean V: {np.mean(data["dist3D"] / data["flight_time"]):.2f}m/s')
    except KeyError:
        pass
    print(data.head())
else:
    data = {}
    plt.figure()

    for fpath in filenames:
        fname = fpath[fpath.rfind('/') + 1:]
        data[fname] = pd.read_csv(Path(fpath), comment='#', skipinitialspace=True)
        print(data[fname].head())
        try:
            # CONFLOGS.
            plt.plot(data[fname]['t'] - data[fname]['t'][0], data[fname]['ni_ac'], label=fname)
        except KeyError:
            pass
        try:
            # FLSTLOGS.
            print(f'Mean {fname}: {np.mean(data[fname]["dist3D"] / data[fname]["flight_time"]):.2f}m/s')
        except KeyError:
            pass
    plt.legend()

