import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle as pkl
from tkinter import Tk, filedialog
from typing import Tuple, List
from utils.intersection_model import IntersectionModel
import pandas as pd
import scipy.optimize as opt

RES_FOLDER = Path(r'../../output/RESULT/')

COLORS = ('blue', 'firebrick')
LABELS = ('Turn', 'No turn')
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


def load_file() -> dict:
    tk_root = Tk()
    res_file = filedialog.askopenfilename(initialdir=RES_FOLDER, title='Select results to plot',
                                          filetypes=[('pickle', '*.pkl')])
    tk_root.destroy()

    with open(res_file, 'rb') as f:
        res = pkl.load(f)
    return res


def load_analytical_model(ratio: Tuple[float, float, float, float]) -> IntersectionModel:
    ana_model = IntersectionModel(flow_ratio=ratio, max_value=MAX_VALUE, accuracy=ACCURACY,
                                  duration=DURATION, speed=SPEED, s_h=S_H, s_v=S_V, t_l=T_L, turn_model=True)
    return ana_model


def load_models(data_dict: dict, ana_model: IntersectionModel) -> dict:
    all_ana_models = {}
    for (run, result) in data_dict.items():
        if run.startswith('WR'):
            flow_ratio = result['scn']['flow_ratio']
            if flow_ratio not in all_ana_models.keys():
                new_model = ana_model.copy()
                new_model.flow_ratio = flow_ratio
                new_model.calculate_models()
                all_ana_models[flow_ratio] = new_model
    return all_ana_models


def process_delays(data_dict: dict):
    turn_pct = []
    flow_ratio = []
    delay = []
    nr_conf = []
    wr_conf = []
    nr_ac = []
    wr_ac = []
    # Extract delays and pct turn.
    for (run, result) in data_dict.items():
        if run.startswith('WR'):
            turn_pct.append(result['scn']['flow_ratio'][2] / BASE_RATIO[0] * 100)
            flow_ratio.append(result['scn']['flow_ratio'])

            ac_index = result['flstlog']['callsign'].isin(result['ac'])
            nr_flight_time = data_dict[f'N{run[1:]}']['flstlog']['flight_time'][ac_index].mean()
            wr_flight_time = result['flstlog']['flight_time'][ac_index].mean()

            delay.append(wr_flight_time - nr_flight_time)
            nr_conf.append(data_dict[f'N{run[1:]}']['conf']['ntotal_conf'])
            wr_conf.append(result['conf']['ntotal_conf'])
            nr_ac.append(data_dict[f'N{run[1:]}']['conf']['ni_ac'])
            wr_ac.append(result['conf']['ni_ac'])

    parsed_df = pd.DataFrame({
        'turn_pct': turn_pct,
        'flow_ratio': flow_ratio,
        'delay_wr': delay,
        'c_total_nr': nr_conf,
        'c_total_wr': wr_conf,
        'n_inst_nr': nr_ac,
        'n_inst_wr': wr_ac
    })
    return parsed_df


def create_plots(save: bool, folder: Path) -> List[plt.Figure]:
    x = parsed_data['turn_pct']
    # Plot delay.
    fig, ax = plt.subplots()
    ax.plot(x, parsed_data['delay_wr'],
            marker=MARKER, linestyle=LINESTYLE, alpha=ALPHA, color=COLORS[0], label=None)
    ax.plot(x, [all_models[fr].delay_wr.iloc[0] for fr in parsed_data['flow_ratio']],
            color=COLORS[0], label=None)
    ax.plot([0, 100], [base_model.delay_wr.iloc[0]] * 2,
            color=COLORS[1], label=None)
    ax.legend(LEGEND_ELEMENTS, LABELS, loc='upper left')
    ax.set_xlabel('Percentage turning traffic [%]')
    ax.set_ylabel('Mean delay per aircraft [s]')
    ax.set_xlim([-5, 100])
    ax.set_ylim([-11 / 20, 11])
    if save:
        fig.savefig(folder / 'turn_assumption_model_delay.png', bbox_inches='tight')
        fig.savefig(folder / 'turn_assumption_model_delay.eps', bbox_inches='tight')

    # Plot conflicts.
    # NR.
    fig2, ax2 = plt.subplots()
    ax2.plot(x, parsed_data['c_total_nr'],
             marker=MARKER, linestyle=LINESTYLE, alpha=ALPHA, color=COLORS[0], label=None)
    ax2.plot(x, [all_models[fr].c_total_nr.iloc[0] for fr in parsed_data['flow_ratio']],
             color=COLORS[0], label=None)
    ax2.plot([0, 100], [base_model.c_total_nr.iloc[0]] * 2,
             color=COLORS[1], label=None)
    ax2.legend(LEGEND_ELEMENTS, LABELS, loc='upper left')
    ax2.set_xlabel('Percentage turning traffic [%]')
    ax2.set_ylabel('Total number of conflicts NR [-]')
    ax2.set_xlim([-5, 100])
    ax2.set_ylim([-250 / 20, 250])
    if save:
        fig2.savefig(folder / 'turn_assumption_model_conf_nr.png', bbox_inches='tight')
        fig2.savefig(folder / 'turn_assumption_model_conf_nr.eps', bbox_inches='tight')

    # WR
    fig3, ax3 = plt.subplots()
    ax3.plot(x, parsed_data['c_total_wr'],
             marker=MARKER, linestyle=LINESTYLE, alpha=ALPHA, color=COLORS[0], label=None)
    ax3.plot(x, [all_models[fr].c_total_wr.iloc[0] for fr in parsed_data['flow_ratio']],
             color=COLORS[0], label=None)
    ax3.plot([0, 100], [base_model.c_total_wr.iloc[0]] * 2,
             color=COLORS[1], label=None)
    ax3.legend(LEGEND_ELEMENTS, LABELS, loc='upper left')
    ax3.set_xlabel('Percentage turning traffic [%]')
    ax3.set_ylabel('Total number of conflicts WR [-]')
    ax3.set_xlim([-5, 100])
    ax3.set_ylim([-400 / 20, 400])
    if save:
        fig3.savefig(folder / 'turn_assumption_model_conf_wr.png', bbox_inches='tight')
        fig3.savefig(folder / 'turn_assumption_model_conf_wr.eps', bbox_inches='tight')
    return [fig, fig2, fig3]


def fit_k(exp, model) -> float:
    return opt.fmin(lambda k: np.nansum(np.power(exp - model.values * k, 2)),
                    x0=1, disp=False)[0]


def determine_k(save: bool, folder: Path) -> pd.Series:
    relevant_vars = ['c_total_nr', 'delay_wr', 'c_total_wr']

    model_df = parsed_data.copy()
    model_df[relevant_vars] = 0.
    for i in range(len(parsed_data)):
        k_model = all_models[parsed_data['flow_ratio'].iloc[i]].copy()
        k_model.n_inst = np.array([parsed_data['n_inst_nr'].iloc[i]])
        k_model.calculate_models()

        model_df.iloc[i, model_df.columns.get_loc('c_total_nr')] = k_model.c_total_nr.values
        model_df.iloc[i, model_df.columns.get_loc('delay_wr')] = k_model.delay_wr.values
        model_df.iloc[i, model_df.columns.get_loc('c_total_wr')] = k_model.c_total_wr.values

    k_dict = dict()
    for var in relevant_vars:
        k_dict[var] = fit_k(parsed_data[var], model_df[var])
    all_k = pd.Series(k_dict)
    if save:
        all_k.to_csv(folder / 'turn_accuracy.csv')
        print('Saved turn_accuracy.csv.')
    return all_k


def determine_k_pct(df: pd.Series, save: bool, folder: Path) -> pd.Series:
    pct = df.copy().apply(lambda k: (1 - abs((k - 1)/k)) * 100)
    if save:
        pct.to_csv(folder / 'turn_accuracy_pct.csv')
        print('Saved turn_accuracy_pct.csv')
    return pct


if __name__ == '__main__':
    BASE_RATIO = (0.6, 0.4, 0., 0.)
    SAT = 0.7
    SAVE = True
    PAPER_FOLDER = Path(r'C:\Users\michi\Dropbox\TU\Thesis\05_Paper')

    data = load_file()

    base_model = load_analytical_model(BASE_RATIO)
    base_n_inst_nr = SAT * MEAN_FLIGHT_TIME / (S_H * np.sqrt(2) / SPEED)
    base_model.n_inst = np.array([base_n_inst_nr])
    base_model.calculate_models()
    all_models = load_models(data, base_model)

    parsed_data = process_delays(data)

    figs = create_plots(save=SAVE, folder=PAPER_FOLDER)
    k_series = determine_k(save=SAVE, folder=PAPER_FOLDER)
    pct_series = determine_k_pct(k_series, save=SAVE, folder=PAPER_FOLDER)
