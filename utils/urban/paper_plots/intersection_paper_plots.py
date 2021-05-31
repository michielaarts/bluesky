import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tkinter import Tk, filedialog
from typing import List, Tuple
from utils.urban.intersection_model import IntersectionModel
import re
import scipy.optimize as opt
import numpy as np

RES_FOLDER = Path('../../../output/RESULT/')
COLORS = ('green', 'royalblue', 'orchid')
MARKER = 'x'
ALPHA = 0.5
LINESTYLE = 'None'

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

plt.rcParams.update({'font.size': 16})


def load_files() -> Tuple[List[pd.DataFrame], List[Tuple[float, float, float, float]]]:
    tk_root = Tk()
    res_files = filedialog.askopenfilenames(initialdir=RES_FOLDER, title='Select results to plot',
                                            filetypes=[('csv', '*.csv')])
    tk_root.destroy()

    dfs = []
    ratios = []
    for f in res_files:
        dfs.append(pd.read_csv(f'{f[:-4]}.csv', header=[0, 1]))
        flow_ratio_string = re.findall(r'.*_(\d*)_NR.csv', f)[0]
        flow_ratio = (float(flow_ratio_string[:2])/100., float(flow_ratio_string[2:4])/100.,
                      float(flow_ratio_string[4:6])/100., float(flow_ratio_string[6:])/100.,)
        ratios.append(flow_ratio)
    return dfs, ratios


def load_analytical_models(ratios: List[Tuple[float, float, float, float]]) -> List[IntersectionModel]:
    all_models = []
    for fr in ratios:
        model = IntersectionModel(flow_ratio=fr, max_value=MAX_VALUE, accuracy=ACCURACY,
                                  duration=DURATION, speed=SPEED, s_h=S_H, s_v=S_V, t_l=T_L, turn_model=True)
        all_models.append(model)
    return all_models


def create_plots(save: bool, folder: Path):
    legend_order = [1, 2, 0]
    legend_elements = [plt.Line2D([0], [0], linestyle='-', marker=MARKER, color=color) for color in COLORS]
    legend_entries = ['/'.join(str(fr) for fr in flow_ratio) for flow_ratio in flow_ratios]
    legend_loc = 'upper left'
    legend_elements = list(legend_elements[i] for i in legend_order)
    legend_entries = list(legend_entries[i] for i in legend_order)

    # NR conflict count.
    nr_conf_inst_fig, nr_conf_inst_ax = plt.subplots()
    for i in range(len(data)):
        nr_conf_inst_ax.plot(data[i]['NR', 'ni_ac'], data[i]['NR', 'ni_conf'], label=None,
                             linestyle=LINESTYLE, color=COLORS[i], marker=MARKER, alpha=ALPHA)
        nr_conf_inst_ax.plot(models[i].n_inst, models[i].c_inst_nr, label=None, color=COLORS[i])
    nr_conf_inst_ax.legend(legend_elements, legend_entries, loc=legend_loc)
    nr_conf_inst_ax.set_xlabel('Number of instantaneous aircraft NR [-]')
    nr_conf_inst_ax.set_ylabel('Number of instantaneous conflicts NR [-]')
    nr_conf_inst_ax.set_xlim([-MAX_VALUE / 20, MAX_VALUE])
    # nr_conf_inst_ax.set_ylim([-250/20, 250])

    nr_conf_fig, nr_conf_ax = plt.subplots()
    for i in range(len(data)):
        nr_conf_ax.plot(data[i]['NR', 'ni_ac'], data[i]['NR', 'ntotal_conf'], label=None,
                        linestyle=LINESTYLE, color=COLORS[i], marker=MARKER, alpha=ALPHA)
        nr_conf_ax.plot(models[i].n_inst, models[i].c_total_nr, label=None, color=COLORS[i])
    nr_conf_ax.legend(legend_elements, legend_entries, loc=legend_loc)
    nr_conf_ax.set_xlabel('Number of instantaneous aircraft NR [-]')
    nr_conf_ax.set_ylabel('Total number of conflicts NR [-]')
    nr_conf_ax.set_xlim([-MAX_VALUE / 20, MAX_VALUE])
    # nr_conf_ax.set_ylim([-250/20, 250])

    # NR intrusion count.
    nr_los_inst_fig, nr_los_inst_ax = plt.subplots()
    for i in range(len(data)):
        nr_los_inst_ax.plot(data[i]['NR', 'ni_ac'], data[i]['NR', 'ni_los'], label=None,
                            linestyle=LINESTYLE, color=COLORS[i], marker=MARKER, alpha=ALPHA)
        nr_los_inst_ax.plot(models[i].n_inst, models[i].los_inst_nr, label=None, color=COLORS[i])
    nr_los_inst_ax.legend(legend_elements, legend_entries, loc=legend_loc)
    nr_los_inst_ax.set_xlabel('Number of instantaneous aircraft NR [-]')
    nr_los_inst_ax.set_ylabel('Number of instantaneous intrusions NR [-]')
    nr_los_inst_ax.set_xlim([-MAX_VALUE / 20, MAX_VALUE])
    # nr_los_inst_ax.set_ylim([-1.75/20, 1.75])

    nr_los_fig, nr_los_ax = plt.subplots()
    for i in range(len(data)):
        nr_los_ax.plot(data[i]['NR', 'ni_ac'], data[i]['NR', 'ntotal_los'], label=None,
                       linestyle=LINESTYLE, color=COLORS[i], marker=MARKER, alpha=ALPHA)
        nr_los_ax.plot(models[i].n_inst, models[i].los_total_nr, label=None, color=COLORS[i])
    nr_los_ax.legend(legend_elements, legend_entries, loc=legend_loc)
    nr_los_ax.set_xlabel('Number of instantaneous aircraft NR [-]')
    nr_los_ax.set_ylabel('Total number of intrusions NR [-]')
    nr_los_ax.set_xlim([-MAX_VALUE / 20, MAX_VALUE])
    # nr_los_ax.set_ylim([-1.75/20, 1.75])

    # WR intrusion
    wr_los_inst_fig, wr_los_inst_ax = plt.subplots()
    for i in range(len(data)):
        wr_los_inst_ax.plot(data[i]['WR', 'ni_ac'], data[i]['WR', 'ni_los'], label=None,
                            linestyle=LINESTYLE, color=COLORS[i], marker=MARKER, alpha=ALPHA)
        wr_los_inst_ax.plot(models[i].n_inst, models[i].los_total_nr * 0., label=None, color=COLORS[i])
    wr_los_inst_ax.legend(legend_elements, legend_entries, loc=legend_loc)
    wr_los_inst_ax.set_xlabel('Number of instantaneous aircraft WR [-]')
    wr_los_inst_ax.set_ylabel('Number of instantaneous intrusions WR [-]')
    wr_los_inst_ax.set_xlim([-MAX_VALUE / 20, MAX_VALUE])
    # wr_los_inst_ax.set_ylim([-1.75/20, 1.75])

    wr_los_fig, wr_los_ax = plt.subplots()
    for i in range(len(data)):
        wr_los_ax.plot(data[i]['WR', 'ni_ac'], data[i]['WR', 'ntotal_los'], label=None,
                       linestyle=LINESTYLE, color=COLORS[i], marker=MARKER, alpha=ALPHA)
        wr_los_ax.plot(models[i].n_inst, models[i].los_total_nr * 0., label=None, color=COLORS[i])
    wr_los_ax.legend(legend_elements, legend_entries, loc=legend_loc)
    wr_los_ax.set_xlabel('Number of instantaneous aircraft WR [-]')
    wr_los_ax.set_ylabel('Total number of intrusions WR [-]')
    wr_los_ax.set_xlim([-MAX_VALUE / 20, MAX_VALUE])
    # wr_los_ax.set_ylim([-1.75/20, 1.75])

    # Delay.
    wr_delay_fig, wr_delay_ax = plt.subplots()
    for i in range(len(data)):
        wr_delay_ax.plot(data[i]['WR', 'ni_ac'], data[i]['WR', 'delay'],
                         label=None, linestyle=LINESTYLE, color=COLORS[i], marker=MARKER, alpha=ALPHA)
        wr_delay_ax.plot(models[i].n_inst_wr, models[i].delay_wr,
                         label=None, color=COLORS[i])
    wr_delay_ax.legend(legend_elements, legend_entries, loc=legend_loc)
    wr_delay_ax.set_xlabel('Number of instantaneous aircraft WR [-]')
    wr_delay_ax.set_ylabel('Mean intersection delay per vehicle [s]')
    wr_delay_ax.set_xlim([-MAX_VALUE / 20, MAX_VALUE])
    # wr_delay_ax.set_ylim([-6/20, 6])

    # Mean V.
    wr_v_fig, wr_v_ax = plt.subplots()
    for i in range(len(data)):
        wr_v_ax.plot(data[i]['WR', 'ni_ac'], data[i]['WR', 'mean_v'],
                     label=None, linestyle=LINESTYLE, color=COLORS[i], marker=MARKER, alpha=ALPHA)
        wr_v_ax.plot(models[i].n_inst_wr, models[i].mean_v_wr,
                     label=None, color=COLORS[i])
    wr_v_ax.legend(legend_elements, legend_entries, loc=legend_loc)
    wr_v_ax.set_xlabel('Number of instantaneous aircraft WR [-]')
    wr_v_ax.set_ylabel('Mean speed [m/s]')
    wr_v_ax.set_xlim([-MAX_VALUE / 20, MAX_VALUE])
    # wr_v_ax.set_ylim([9.5, 10.05])

    # WR conflict count.
    wr_conf_fig, wr_conf_ax = plt.subplots()
    for i in range(len(data)):
        wr_conf_ax.plot(data[i]['WR', 'ni_ac'], data[i]['WR', 'ntotal_conf'], label=None,
                        linestyle=LINESTYLE, color=COLORS[i], marker=MARKER, alpha=ALPHA)
        wr_conf_ax.plot(models[i].n_inst_wr, models[i].c_total_wr,
                        label=None, color=COLORS[i])
    wr_conf_ax.legend(legend_elements, legend_entries, loc=legend_loc)
    wr_conf_ax.set_xlabel('Number of instantaneous aircraft WR [-]')
    wr_conf_ax.set_ylabel('Total number of conflicts WR [-]')
    wr_conf_ax.set_xlim([-MAX_VALUE / 20, MAX_VALUE])
    # wr_conf_ax.set_ylim([-550/20, 550])

    # WR Ni.
    wr_ni_fig, wr_ni_ax = plt.subplots()
    for i in range(len(data)):
        wr_ni_ax.plot(data[i]['NR', 'ni_ac'], data[i]['WR', 'ni_ac'], label=None,
                      linestyle=LINESTYLE, color=COLORS[i], marker=MARKER, alpha=ALPHA)
        wr_ni_ax.plot(models[i].n_inst, models[i].n_inst_wr,
                      label=None, color=COLORS[i])
    wr_ni_ax.legend(legend_elements, legend_entries, loc=legend_loc)
    wr_ni_ax.set_xlabel('Number of instantaneous aircraft NR [-]')
    wr_ni_ax.set_ylabel('Number of instantaneous aircraft WR [-]')
    wr_ni_ax.set_xlim([-MAX_VALUE / 20, MAX_VALUE])
    # wr_ni_ax.set_ylim([-550/20, 550])

    # MFD.
    mfd_fig, mfd_ax = plt.subplots()
    for i in range(len(data)):
        mfd_ax.plot(data[i]['WR', 'ni_ac'], data[i]['WR', 'ni_ac'] * data[i]['WR', 'mean_v'],
                    label=None, linestyle=LINESTYLE, color=COLORS[i], marker=MARKER, alpha=ALPHA)
        mfd_ax.plot(models[i].n_inst_wr, models[i].flow_rate_wr,
                    label=None, color=COLORS[i])
    mfd_ax.legend(legend_elements, legend_entries, loc=legend_loc)
    mfd_ax.set_xlabel('Number of instantaneous aircraft WR [-]')
    mfd_ax.set_ylabel('Network flow rate [veh m / s]')
    mfd_ax.set_xlim([-MAX_VALUE / 20, MAX_VALUE])
    # mfd_ax.set_ylim([-550/20, 550])

    # DEP.
    dep_fig, dep_ax = plt.subplots()
    for i in range(len(data)):
        dep_ax.plot(data[i]['NR', 'ni_ac'], data[i]['WR', 'ntotal_conf'] / data[i]['NR', 'ntotal_conf'] - 1,
                    label=None, linestyle=LINESTYLE, color=COLORS[i], marker=MARKER, alpha=ALPHA)
        dep_ax.plot(models[i].n_inst, models[i].dep,
                    label=None, color=COLORS[i])
    dep_ax.legend(legend_elements, legend_entries, loc=legend_loc)
    dep_ax.set_xlabel('Number of instantaneous aircraft NR [-]')
    dep_ax.set_ylabel('Domino Effect Parameter [-]')
    dep_ax.set_xlim([-MAX_VALUE / 20, MAX_VALUE])
    # dep_ax.set_ylim([-550/20, 550])

    if save:
        # Save figures.
        nr_conf_inst_fig.savefig(folder / 'c_inst_nr.eps', bbox_inches='tight')
        nr_conf_inst_fig.savefig(folder / 'c_inst_nr.png', bbox_inches='tight')
        nr_conf_fig.savefig(folder / 'c_total_nr.eps', bbox_inches='tight')
        nr_conf_fig.savefig(folder / 'c_total_nr.png', bbox_inches='tight')
        nr_los_inst_fig.savefig(folder / 'los_inst_nr.eps', bbox_inches='tight')
        nr_los_inst_fig.savefig(folder / 'los_inst_nr.png', bbox_inches='tight')
        nr_los_fig.savefig(folder / 'los_total_nr.eps', bbox_inches='tight')
        nr_los_fig.savefig(folder / 'los_total_nr.png', bbox_inches='tight')
        wr_los_inst_fig.savefig(folder / 'los_inst_wr.eps', bbox_inches='tight')
        wr_los_inst_fig.savefig(folder / 'los_inst_wr.png', bbox_inches='tight')
        wr_los_fig.savefig(folder / 'los_total_wr.eps', bbox_inches='tight')
        wr_los_fig.savefig(folder / 'los_total_wr.png', bbox_inches='tight')
        wr_delay_fig.savefig(folder / 'delay_wr.eps', bbox_inches='tight')
        wr_delay_fig.savefig(folder / 'delay_wr.png', bbox_inches='tight')
        wr_conf_fig.savefig(folder / 'c_total_wr.eps', bbox_inches='tight')
        wr_conf_fig.savefig(folder / 'c_total_wr.png', bbox_inches='tight')
        wr_ni_fig.savefig(folder / 'n_inst_wr.eps', bbox_inches='tight')
        wr_ni_fig.savefig(folder / 'n_inst_wr.png', bbox_inches='tight')
        mfd_fig.savefig(folder / 'mfd.eps', bbox_inches='tight')
        mfd_fig.savefig(folder / 'mfd.png', bbox_inches='tight')
        dep_fig.savefig(folder / 'dep.eps', bbox_inches='tight')
        dep_fig.savefig(folder / 'dep.png', bbox_inches='tight')


def fit_k(exp, model) -> float:
    return opt.fmin(lambda k: np.nansum(np.power(exp - model.values * k, 2)),
                    x0=1, disp=False)[0]


def determine_k(save: bool, folder: Path) -> pd.DataFrame:
    legend_entries = ['/'.join(str(fr) for fr in flow_ratio) for flow_ratio in flow_ratios]
    all_k_dict = dict()

    for i in range(len(data)):
        k_model = models[i].copy()
        k_model.n_inst = data[i]['NR']['ni_ac'].to_numpy()
        k_model.calculate_models()

        k_dict = dict()
        k_dict['c_inst_nr'] = fit_k(data[i]['NR', 'ni_conf'], k_model.c_inst_nr)
        k_dict['c_total_nr'] = fit_k(data[i]['NR', 'ntotal_conf'], k_model.c_total_nr)
        k_dict['los_inst_nr'] = fit_k(data[i]['NR', 'ni_los'], k_model.los_inst_nr)
        k_dict['los_total_nr'] = fit_k(data[i]['NR', 'ntotal_los'], k_model.los_total_nr)
        k_dict['delay_wr'] = fit_k(data[i]['WR', 'delay'], k_model.delay_wr)
        k_dict['mean_v'] = np.nan
        k_dict['c_total_wr'] = fit_k(data[i]['WR', 'ntotal_conf'], k_model.c_total_wr)
        k_dict['n_inst_wr'] = fit_k(data[i]['WR', 'ni_ac'], k_model.n_inst_wr)
        k_dict['mfd'] = fit_k(data[i]['WR', 'ni_ac'] * data[i]['WR', 'mean_v'], k_model.flow_rate_wr)
        k_dict['dep'] = fit_k(data[i]['WR', 'ntotal_conf'] / data[i]['NR', 'ntotal_conf'] - 1, k_model.dep)

        all_k_dict[legend_entries[i]] = k_dict

    all_k_df = pd.DataFrame.from_dict(all_k_dict)
    if save:
        all_k_df.to_csv(folder / 'accuracy.csv')
        print('Saved accuracy.csv.')
    return all_k_df


def determine_k_pct(df: pd.DataFrame, save: bool, folder: Path) -> pd.DataFrame:
    pct = df.copy().apply(lambda k: (1 - abs((k - 1)/k)) * 100)
    if save:
        pct.to_csv(folder / 'accuracy_pct.csv')
        print('Saved accuracy_pct.csv')
    return pct


if __name__ == '__main__':
    SAVE = True
    PAPER_FOLDER = Path(r'C:\Users\michi\Dropbox\TU\Thesis\05_Paper')

    data, flow_ratios = load_files()
    models = load_analytical_models(flow_ratios)
    create_plots(save=SAVE, folder=PAPER_FOLDER)
    k_df = determine_k(save=SAVE, folder=PAPER_FOLDER)
    pct_df = determine_k_pct(k_df, save=SAVE, folder=PAPER_FOLDER)
