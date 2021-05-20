import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tkinter import Tk, filedialog
from typing import List, Tuple
from utils.network_model import NetworkModel
from plugins.urban import UrbanGrid
import pickle as pkl
import re
import numpy as np
import scipy.optimize as opt

RES_FOLDER = Path('../../output/RESULT/')
GRID_FOLDER = Path('../../scenario/URBAN/Data/')
COLORS = ['firebrick']
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
MAX_VALUE = 70.
ACCURACY = 100

plt.rcParams.update({'font.size': 16})


def load_files() -> Tuple[List[pd.DataFrame], List[UrbanGrid], List[str]]:
    tk_root = Tk()
    res_files = filedialog.askopenfilenames(initialdir=RES_FOLDER, title='Select results to plot',
                                            filetypes=[('csv', '*.csv')])
    tk_root.destroy()

    dfs = []
    grids = []
    prefixs = []
    for f in res_files:
        dfs.append(pd.read_csv(f'{f[:-4]}.csv', header=[0, 1]))
        prefix = re.findall('.*batch_(.*)_NR.csv', f)[0]
        with open(f'{GRID_FOLDER / prefix}_urban_grid.pkl', 'rb') as pkl_file:
            grids.append(pkl.load(pkl_file))
        prefixs.append(prefix)
    return dfs, grids, prefixs


def load_analytical_models(grids: List[UrbanGrid]) -> List[NetworkModel]:
    all_models = []
    for grid in grids:
        all_models.append(NetworkModel(urban_grid=grid, max_value=MAX_VALUE, accuracy=ACCURACY, duration=DURATION,
                                       speed=SPEED, s_h=S_H, s_v=S_V, t_l=T_L, turn_model=True))
    return all_models


def create_plots(save: bool, folder: Path) -> List[plt.Figure]:
    all_figures = []
    legend_elements = [plt.Line2D([0], [0], linestyle='-', marker='None', color=COLORS[0]),
                       plt.Line2D([0], [0], linestyle='None', marker=MARKER, color=COLORS[0])]
    legend_entries = ['Model', 'Experiment']
    legend_loc = 'upper left'

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
    all_figures.append(nr_conf_inst_fig)

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
    all_figures.append(nr_conf_fig)

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
    all_figures.append(nr_los_inst_fig)

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
    all_figures.append(nr_los_fig)

    # WR intrusion inst
    wr_los_inst_fig, wr_los_inst_ax = plt.subplots()
    for i in range(len(data)):
        wr_los_inst_ax.plot(data[i]['WR', 'ni_ac'], data[i]['WR', 'ni_los'], label=None,
                            linestyle=LINESTYLE, color=COLORS[i], marker=MARKER, alpha=ALPHA)
        wr_los_inst_ax.plot(models[i].n_inst, models[i].n_inst * 0., label=None, color=COLORS[i])
    wr_los_inst_ax.legend(legend_elements, legend_entries, loc=legend_loc)
    wr_los_inst_ax.set_xlabel('Number of instantaneous aircraft WR [-]')
    wr_los_inst_ax.set_ylabel('Number of instantaneous intrusions WR [-]')
    wr_los_inst_ax.set_xlim([-MAX_VALUE / 20, MAX_VALUE])
    # wr_los_inst_ax.set_ylim([-1.75/20, 1.75])
    all_figures.append(wr_los_inst_fig)

    # WR intrusion total
    wr_los_fig, wr_los_ax = plt.subplots()
    for i in range(len(data)):
        wr_los_ax.plot(data[i]['WR', 'ni_ac'], data[i]['WR', 'ntotal_los'], label=None,
                       linestyle=LINESTYLE, color=COLORS[i], marker=MARKER, alpha=ALPHA)
        wr_los_ax.plot(models[i].n_inst, models[i].n_inst * 0., label=None, color=COLORS[i])
    wr_los_ax.legend(legend_elements, legend_entries, loc=legend_loc)
    wr_los_ax.set_xlabel('Number of instantaneous aircraft WR [-]')
    wr_los_ax.set_ylabel('Total number of intrusions WR [-]')
    wr_los_ax.set_xlim([-MAX_VALUE / 20, MAX_VALUE])
    # wr_los_ax.set_ylim([-1.75/20, 1.75])
    all_figures.append(wr_los_fig)

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
    all_figures.append(wr_delay_fig)

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
    all_figures.append(wr_v_fig)

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
    all_figures.append(wr_conf_fig)

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
    all_figures.append(wr_ni_fig)

    # MFD.
    mfd_fig, mfd_ax = plt.subplots()
    for i in range(len(data)):
        mfd_ax.plot(data[i]['WR', 'ni_ac'], data[i]['WR', 'ni_ac'] * data[i]['WR', 'mean_v'],
                    label=None, linestyle=LINESTYLE, color=COLORS[i], marker=MARKER, alpha=ALPHA)
        mfd_ax.plot(models[i].n_inst_wr, models[i].flow_rate_wr,
                    label=None, color=COLORS[i])
    mfd_ax.legend(legend_elements, legend_entries, loc=legend_loc)
    mfd_ax.set_xlabel('Number of instantaneous aircraft WR [-]')
    mfd_ax.set_ylabel(r'Network flow rate [veh$\cdot$m / s]')
    mfd_ax.set_xlim([-MAX_VALUE / 20, MAX_VALUE])
    # mfd_ax.set_ylim([-550/20, 550])
    all_figures.append(mfd_fig)

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
    all_figures.append(dep_fig)

    # Conf LoS figure.
    nr_conf_los_fig, nr_conf_los_ax = plt.subplots()
    for i in range(len(data)):
        nr_conf_los_ax.plot(data[i]['NR', 'ni_ac'], data[i]['NR', 'ntotal_conf'],
                            label='Conflicts', linestyle='None', color='royalblue', marker=MARKER, alpha=ALPHA)
        nr_conf_los_ax.plot(data[i]['NR', 'ni_ac'], data[i]['NR', 'ntotal_los'],
                            label='Intrusions', linestyle='None', color='darkorange', marker=MARKER, alpha=ALPHA)
    nr_conf_los_ax.legend(loc=legend_loc)
    nr_conf_los_ax.set_xlabel('Number of instantaneous aircraft NR [-]')
    nr_conf_los_ax.set_ylabel('Total number of ... [-]')
    nr_conf_los_ax.set_xlim([-MAX_VALUE / 20, MAX_VALUE])
    # nr_conf_los_ax.set_ylim([-550/20, 550])
    all_figures.append(nr_conf_los_fig)

    # Save figures.
    if save:
        nr_conf_inst_fig.savefig(folder / 'grid_c_inst_nr.eps', bbox_inches='tight')
        nr_conf_inst_fig.savefig(folder / 'grid_c_inst_nr.png', bbox_inches='tight')
        nr_conf_fig.savefig(folder / 'grid_c_total_nr.eps', bbox_inches='tight')
        nr_conf_fig.savefig(folder / 'grid_c_total_nr.png', bbox_inches='tight')
        nr_los_inst_fig.savefig(folder / 'grid_los_inst_nr.eps', bbox_inches='tight')
        nr_los_inst_fig.savefig(folder / 'grid_los_inst_nr.png', bbox_inches='tight')
        nr_los_fig.savefig(folder / 'grid_los_total_nr.eps', bbox_inches='tight')
        nr_los_fig.savefig(folder / 'grid_los_total_nr.png', bbox_inches='tight')
        wr_los_inst_fig.savefig(folder / 'grid_los_inst_wr.eps', bbox_inches='tight')
        wr_los_inst_fig.savefig(folder / 'grid_los_inst_wr.png', bbox_inches='tight')
        wr_los_fig.savefig(folder / 'grid_los_total_wr.eps', bbox_inches='tight')
        wr_los_fig.savefig(folder / 'grid_los_total_wr.png', bbox_inches='tight')
        wr_delay_fig.savefig(folder / 'grid_delay_wr.eps', bbox_inches='tight')
        wr_delay_fig.savefig(folder / 'grid_delay_wr.png', bbox_inches='tight')
        wr_conf_fig.savefig(folder / 'grid_c_total_wr.eps', bbox_inches='tight')
        wr_conf_fig.savefig(folder / 'grid_c_total_wr.png', bbox_inches='tight')
        wr_ni_fig.savefig(folder / 'grid_n_inst_wr.eps', bbox_inches='tight')
        wr_ni_fig.savefig(folder / 'grid_n_inst_wr.png', bbox_inches='tight')
        mfd_fig.savefig(folder / 'grid_mfd.eps', bbox_inches='tight')
        mfd_fig.savefig(folder / 'grid_mfd.png', bbox_inches='tight')
        dep_fig.savefig(folder / 'grid_dep.eps', bbox_inches='tight')
        dep_fig.savefig(folder / 'grid_dep.png', bbox_inches='tight')
        nr_conf_los_fig.savefig(folder / 'grid_nr_conf_los.eps', bbox_inches='tight')
        nr_conf_los_fig.savefig(folder / 'grid_nr_conf_los.png', bbox_inches='tight')
    return all_figures


def fit_k(exp, model) -> float:
    return opt.fmin(lambda k: np.nansum(np.power(exp - model.values * k, 2)),
                    x0=1, disp=False)[0]


def determine_k(save: bool, folder: Path) -> pd.DataFrame:
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

        all_k_dict[prefixes[i]] = k_dict

    all_k_df = pd.DataFrame.from_dict(all_k_dict)
    if save:
        all_k_df.to_csv(folder / 'grid_accuracy.csv')
        print('Saved grid_accuracy.csv.')
    return all_k_df


def determine_k_pct(df: pd.DataFrame, save: bool, folder: Path) -> pd.DataFrame:
    pct = df.copy().apply(lambda k: (1 - abs((k - 1)/k)) * 100)
    if save:
        pct.to_csv(folder / 'grid_accuracy_pct.csv')
        print('Saved grid_accuracy_pct.csv')
    return pct


def determine_mean_los_duration():
    total_los = np.array([data[i]['WR', 'ntotal_los'] for i in range(len(data))])
    inst_los = np.array([data[i]['WR', 'ni_los'] for i in range(len(data))])
    logging_time = models[0].duration[1]
    mean_los_duration = opt.fmin(lambda t_los: np.nansum(np.power(inst_los * logging_time / t_los - total_los, 2)),
                                 x0=1, disp=False)[0]
    return mean_los_duration


if __name__ == '__main__':
    SAVE = True
    PAPER_FOLDER = Path(r'C:\Users\michi\Dropbox\TU\Thesis\05_Paper')

    data, urban_grids, prefixes = load_files()
    models = load_analytical_models(urban_grids)
    figs = create_plots(save=SAVE, folder=PAPER_FOLDER)
    k_df = determine_k(save=SAVE, folder=PAPER_FOLDER)
    pct_df = determine_k_pct(k_df, save=SAVE, folder=PAPER_FOLDER)
    mean_t_los = determine_mean_los_duration()
    print('Mean los duration: ', mean_t_los)
