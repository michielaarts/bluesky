import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tkinter import Tk, filedialog
from typing import List, Tuple
from intersection_model import IntersectionModel
import re

RES_FOLDER = Path('../output/RESULT/')
COLORS = ('green', 'royalblue', 'orchid')
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
MAX_VALUE = 40.
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
        dfs.append(pd.read_csv(f'{f[:-4]}.csv', header=[0,1]))
        flow_ratio_string = re.findall(r'.*_(\d*)_NR.csv', f)[0]
        flow_ratio = (float(flow_ratio_string[:2])/100., float(flow_ratio_string[2:4])/100.,
                      float(flow_ratio_string[4:6])/100., float(flow_ratio_string[6:])/100.,)
        ratios.append(flow_ratio)
    return dfs, ratios


def load_analytical_models(ratios: List[Tuple[float, float, float, float]]) -> List[IntersectionModel]:
    all_models = []
    for fr in ratios:
        model = IntersectionModel(flow_ratio=fr, max_value=MAX_VALUE, accuracy=ACCURACY,
                                  duration=DURATION, speed=SPEED, s_h=S_H, s_v=S_V, t_l=T_L)
        all_models.append(model)
    return all_models


if __name__ == '__main__':
    data, flow_ratios = load_files()
    models = load_analytical_models(flow_ratios)

    # NR conflict count.
    nr_conf_fig, nr_conf_ax = plt.subplots()
    for i in range(len(data)):
        nr_conf_ax.plot(data[i]['NR', 'ni_ac'], data[i]['NR', 'ntotal_conf'], label=None,
                        linestyle=LINESTYLE, color=COLORS[i], marker=MARKER, alpha=ALPHA)
        nr_conf_ax.plot(models[i].n_inst, models[i].c_total_nr, label=None, color=COLORS[i])
    nr_conf_ax.legend(LEGEND_ELEMENTS, ['/'.join(str(fr) for fr in flow_ratio) for flow_ratio in flow_ratios])
    nr_conf_ax.set_xlabel('Number of instantaneous aircraft NR [-]')
    nr_conf_ax.set_ylabel('Total number of conflicts NR [-]')
    nr_conf_ax.set_xlim([-MAX_VALUE/20, MAX_VALUE])
    nr_conf_ax.set_ylim([-300/20, 300])

    # Delay.
    wr_delay_fig, wr_delay_ax = plt.subplots()
    for i in range(len(data)):
        wr_delay_ax.plot(data[i]['WR', 'ni_ac'], data[i]['WR', 'flight_time'] - data[i]['NR', 'flight_time'],
                         label=None, linestyle=LINESTYLE, color=COLORS[i], marker=MARKER, alpha=ALPHA)
        wr_delay_ax.plot(models[i].n_inst_wr, models[i].delay_wr,
                         label=None, color=COLORS[i])
    wr_delay_ax.legend(LEGEND_ELEMENTS, ['/'.join(str(fr) for fr in flow_ratio) for flow_ratio in flow_ratios])
    wr_delay_ax.set_xlabel('Number of instantaneous aircraft WR [-]')
    wr_delay_ax.set_ylabel('Mean intersection delay per vehicle [s]')
    wr_delay_ax.set_xlim([-MAX_VALUE/20, MAX_VALUE])
    wr_delay_ax.set_ylim([-10/20, 10])

    # Mean V.
    wr_v_fig, wr_v_ax = plt.subplots()
    for i in range(len(data)):
        wr_v_ax.plot(data[i]['WR', 'ni_ac'], data[i]['WR', 'mean_v'],
                     label=None, linestyle=LINESTYLE, color=COLORS[i], marker=MARKER, alpha=ALPHA)
        wr_v_ax.plot(models[i].n_inst_wr, models[i].mean_v_wr,
                     label=None, color=COLORS[i])
    wr_v_ax.legend(LEGEND_ELEMENTS, ['/'.join(str(fr) for fr in flow_ratio) for flow_ratio in flow_ratios])
    wr_v_ax.set_xlabel('Number of instantaneous aircraft WR [-]')
    wr_v_ax.set_ylabel('Mean speed [m/s]')
    wr_v_ax.set_xlim([-MAX_VALUE / 20, MAX_VALUE])
    wr_v_ax.set_ylim([9.5, 10.05])

    # WR conflict count.
    wr_conf_fig, wr_conf_ax = plt.subplots()
    for i in range(len(data)):
        wr_conf_ax.plot(data[i]['WR', 'ni_ac'], data[i]['WR', 'ntotal_conf'], label=None,
                        linestyle=LINESTYLE, color=COLORS[i], marker=MARKER, alpha=ALPHA)
        wr_conf_ax.plot(models[i].n_inst_wr, models[i].c_total_wr,
                        label=None, color=COLORS[i])
    wr_conf_ax.legend(LEGEND_ELEMENTS, ['/'.join(str(fr) for fr in flow_ratio) for flow_ratio in flow_ratios])
    wr_conf_ax.set_xlabel('Number of instantaneous aircraft WR [-]')
    wr_conf_ax.set_ylabel('Total number of conflicts WR [-]')
    wr_conf_ax.set_xlim([-MAX_VALUE/20, MAX_VALUE])
    wr_conf_ax.set_ylim([-50, 1000])

