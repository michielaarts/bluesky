import matplotlib.pyplot as plt
from pathlib import Path
import pickle as pkl
from tkinter import Tk, filedialog
from typing import Tuple
from utils.intersection_model import IntersectionModel
import re

RES_FOLDER = Path(r'../output/RESULT/')
PAPER_FOLDER = Path(r'C:\Users\michi\Dropbox\TU\Thesis\05_Paper')

COLORS = ('green', 'firebrick')
LABELS = ('Eastbound', 'Northbound')
MARKER = 'x'
ALPHA = 0.5
LINESTYLE = 'None'
LEGEND_ELEMENTS = [plt.Line2D([0], [0], linestyle='-', marker=MARKER, color=color) for color in COLORS]

S_H = 50.  # m
S_V = 25.  # ft
T_L = 20.  # s
SPEED = 10.  # m/s
BUILD_UP_DURATION = 15 * 60.  # s
EXPERIMENT_DURATION = 45 * 60.  # s
COOL_DOWN_DURATION = 15 * 60.  # s
DURATION = (BUILD_UP_DURATION, EXPERIMENT_DURATION, COOL_DOWN_DURATION)
MAX_VALUE = 30.
ACCURACY = 100

MEAN_FLIGHT_TIME = 2000. / SPEED

plt.rcParams.update({'font.size': 16,
                     'lines.markersize': 8})


def load_file() -> Tuple[dict, Tuple[float, float, float, float]]:
    tk_root = Tk()
    res_file = filedialog.askopenfilename(initialdir=RES_FOLDER, title='Select results to plot',
                                          filetypes=[('pickle', '*.pkl')])
    tk_root.destroy()

    with open(res_file, 'rb') as f:
        res = pkl.load(f)
    flow_ratio_string = re.findall(r'.*_(\d*)_NR.pkl', res_file)[0]
    fr = (float(flow_ratio_string[:2])/100., float(flow_ratio_string[2:4])/100.,
          float(flow_ratio_string[4:6])/100., float(flow_ratio_string[6:])/100.,)
    return res, fr


def load_analytical_model(ratio: Tuple[float, float, float, float]) -> IntersectionModel:
    ana_model = IntersectionModel(flow_ratio=ratio, max_value=MAX_VALUE, accuracy=ACCURACY,
                                  duration=DURATION, speed=SPEED, s_h=S_H, s_v=S_V, t_l=T_L)
    return ana_model


def process_delays(res: dict):
    wr_runs = []
    wr_ni = []
    flow_delay = ([], [])
    ratio = []
    # Extract NR flight times.
    for (run, result) in res.items():
        if run.startswith('NR'):
            eastbound = (result['flstlog']['hdg'] > 45) & (result['flstlog']['hdg'] < 135)
            ac_index = result['flstlog']['callsign'].isin(result['ac'])
            east_flight_time = result['flstlog']['flight_time'][eastbound & ac_index].mean()
            north_flight_time = result['flstlog']['flight_time'][~eastbound & ac_index].mean()
            break

    # Extract delays.
    for (run, result) in data.items():
        if run.startswith('WR'):
            wr_runs.append(run)
            wr_ni.append(result['conf']['ni_ac'])

            eastbound = (result['flstlog']['hdg'] > 45) & (result['flstlog']['hdg'] < 135)
            ac_index = result['flstlog']['callsign'].isin(result['ac'])

            east_delay = result['flstlog']['flight_time'][eastbound & ac_index].mean()
            north_delay = result['flstlog']['flight_time'][~eastbound & ac_index].mean()
            flow_delay[0].append(east_delay - east_flight_time)
            flow_delay[1].append(north_delay - north_flight_time)
            ratio.append((eastbound & ac_index).sum() / (~eastbound & ac_index).sum())

    return wr_ni, flow_delay, ratio


if __name__ == '__main__':
    data, flow_ratio = load_file()
    model = load_analytical_model(flow_ratio)
    n_inst_wr, flow_delays, exp_ratio = process_delays(data)

    # Plot delay.
    fig, ax = plt.subplots()
    for i in range(2):
        ax.plot(n_inst_wr, flow_delays[i],
                marker=MARKER, linestyle=LINESTYLE, alpha=ALPHA, color=COLORS[i], label=None)
    ax.plot(model.n_inst_wr, model.delays.loc['west', 'middle'],
            color=COLORS[0], label=None)
    ax.plot(model.n_inst_wr, model.delays.loc['south', 'middle'],
            color=COLORS[1], label=None)
    ax.legend(LEGEND_ELEMENTS, LABELS, loc='upper left')
    ax.set_xlabel('Number of instantaneous aircraft WR [-]')
    ax.set_ylabel('Mean delay per aircraft [s]')
    ax.set_xlim([-MAX_VALUE/20, MAX_VALUE])
    ax.set_ylim([-5/20, 5])
    # fig.savefig(PAPER_FOLDER / 'flow_delay.png', bbox_inches='tight')
    # fig.savefig(PAPER_FOLDER / 'flow_delay.eps', bbox_inches='tight')
