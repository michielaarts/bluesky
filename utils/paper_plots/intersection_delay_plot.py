import matplotlib.pyplot as plt
from pathlib import Path
import pickle as pkl
from tkinter import Tk, filedialog
from typing import Tuple
import pandas as pd
from utils.intersection_model import IntersectionModel
import re
import numpy as np
import scipy.optimize as opt

RES_FOLDER = Path(r'../../output/RESULT/')
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
MAX_VALUE = 32.5
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
    total_east_flight_time_nr = dict()
    total_north_flight_time_nr = dict()
    total_east_ac = dict()
    total_north_ac = dict()
    wr_runs = []
    nr_ni = []
    wr_ni = []
    flow_delay = ([], [])
    ratio = []
    # Extract NR flight times.
    for (run, result) in res.items():
        if run.startswith('NR'):
            eastbound = (result['flstlog']['hdg'] > 45) & (result['flstlog']['hdg'] < 135)
            ac_index = result['flstlog']['callsign'].isin(result['ac'])
            total_east_flight_time_nr[run] = result['flstlog']['flight_time'][eastbound & ac_index].sum()
            total_north_flight_time_nr[run] = result['flstlog']['flight_time'][~eastbound & ac_index].sum()
            total_east_ac[run] = sum(eastbound & ac_index)
            total_north_ac[run] = sum(~eastbound & ac_index)

    # Extract WR flight times and delays.
    for (run, result) in data.items():
        if run.startswith('WR'):
            nr_run = f'N{run[1:]}'

            wr_runs.append(run)
            wr_ni.append(result['conf']['ni_ac'])
            nr_ni.append(res[nr_run]['conf']['ni_ac'])

            eastbound = (result['flstlog']['hdg'] > 45) & (result['flstlog']['hdg'] < 135)
            ac_index = result['flstlog']['callsign'].isin(result['ac'])

            total_east_flight_time_wr = result['flstlog']['flight_time'][eastbound & ac_index].sum()
            total_north_flight_time_wr = result['flstlog']['flight_time'][~eastbound & ac_index].sum()
            flow_delay[0].append((total_east_flight_time_wr - total_east_flight_time_nr[nr_run])
                                 / total_east_ac[nr_run])
            flow_delay[1].append((total_north_flight_time_wr - total_north_flight_time_nr[nr_run])
                                 / total_north_ac[nr_run])
            ratio.append(total_east_ac[nr_run] / total_north_ac[nr_run])

    return nr_ni, wr_ni, flow_delay, ratio


def fit_k(exp, ana_model) -> float:
    return opt.fmin(lambda k: np.nansum(np.power(exp - ana_model.values * k, 2)),
                    x0=1, disp=False)[0]


if __name__ == '__main__':
    data, flow_ratio = load_file()
    model = load_analytical_model(flow_ratio)
    n_inst_nr, n_inst_wr, flow_delays, exp_ratio = process_delays(data)

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
    ax.set_ylim([-45/20, 45])

    fr_string = ''.join(str(round(flow * 100)).zfill(2) for flow in flow_ratio)
    fig.savefig(PAPER_FOLDER / f'flow_delay_{fr_string}.png', bbox_inches='tight')
    fig.savefig(PAPER_FOLDER / f'flow_delay_{fr_string}.eps', bbox_inches='tight')

    new_model = model.copy()
    new_model.n_inst = np.array(n_inst_nr)
    new_model.calculate_models()
    new_model.delays[new_model.delays >= 1E4] = np.nan
    k_east = fit_k(flow_delays[0], new_model.delays.loc['west', 'middle'])
    k_north = fit_k(flow_delays[1], new_model.delays.loc['south', 'middle'])
    k_df = pd.DataFrame({'k': [k_east, k_north]}, index=['east', 'north'])
    k_df['k_pct'] = k_df['k'].copy().apply(lambda k: (1 - abs((k - 1)/k)) * 100)
    k_df.to_csv(PAPER_FOLDER / f'flow_delay_{fr_string}_accuracy.csv')
    print(k_df.head())
