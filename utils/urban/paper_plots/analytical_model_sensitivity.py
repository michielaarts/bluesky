"""
This file assesses the sensitivity of the analytical grid network model for various airspace design parameters.

Created by Michiel Aarts, May 2021.
"""

import numpy as np
import pandas as pd
from utils.urban.network_model import NetworkModel
from utils.urban.urban_grid_network import UrbanGrid
from pathlib import Path
import matplotlib.pyplot as plt

# Base grid.
N_ROWS = 7
N_COLS = N_ROWS
GRID_HEIGHT = 200.  # m
GRID_WIDTH = GRID_HEIGHT
BASE_GRID = UrbanGrid(N_ROWS, N_COLS, GRID_WIDTH, GRID_HEIGHT)

# Base model.
S_H = 50.  # m
S_V = 25.  # ft
T_L = 20.  # s
SPEED = 10.
DURATION = (900., 2700., 900.)
BASE_MODEL = NetworkModel(BASE_GRID, max_value=250, accuracy=1000,
                          duration=DURATION, speed=SPEED, s_h=S_H, s_v=S_V, t_l=T_L, turn_model=True)

# Plot parameters.
plt.rcParams.update({'font.size': 16})


def max_capacity(model: NetworkModel) -> dict:
    """ Extracts the maximum capacity of a network model according to the assumption:
    once the mean queue length of an intersection flow is longer than the distance to
    the upstream intersection of that flow, the entire airspace becomes unstable.

    :param model: Analytical grid network model
    :return: Maximum capacity density (inst. no. of aircraft).
    """
    max_idx = np.nanargmax(model.n_inst_wr)
    return {'ni_nr': model.n_inst[max_idx],
            'ni_wr': model.n_inst_wr.iloc[max_idx],
            'ntotal': model.n_total[max_idx],
            'c_total_nr': model.c_total_nr.iloc[max_idx],
            'c_total_wr': model.c_total_wr.iloc[max_idx],
            'dep': model.dep.iloc[max_idx],
            'mean_d': model.delay_wr.iloc[max_idx]}


if __name__ == '__main__':
    all_capacities = dict()
    all_capacities['Base'] = max_capacity(BASE_MODEL)

    # Speed x2.
    v2_model = BASE_MODEL.copy()
    v2_model.speed *= 2.
    v2_model.calculate_models()
    all_capacities[r'$2 \cdot V_{NR}$'] = max_capacity(v2_model)

    # S_h x2.
    sh_model = BASE_MODEL.copy()
    sh_model.s_h *= 2.
    sh_model.calculate_models()
    all_capacities[r'$2 \cdot S_h$'] = max_capacity(sh_model)

    # Grid node distance x2.
    grid_grid = UrbanGrid(N_ROWS, N_COLS, GRID_WIDTH * 2., GRID_HEIGHT * 2.)
    grid_model = BASE_MODEL.copy()
    grid_model.urban_grid = grid_grid
    grid_model.calculate_models()
    all_capacities[r'$2 \cdot D$'] = max_capacity(grid_model)

    # Construct dataframe.
    capacity = pd.DataFrame.from_dict(all_capacities, orient='index')
    capacity.to_csv(Path('../../../output/RESULT/analytical_model_sensitivity.csv'))
    print(capacity)

    # Create bar plot.
    plt.figure()
    plt.barh(capacity.index, capacity['ni_wr'], color='slategrey')
    plt.gca().invert_yaxis()
    plt.xlabel('Maximum number of instantaneous aircraft WR [-]')
    plt.savefig(Path('../../../output/RESULT/analytical_model_sensitivity.eps'), bbox_inches='tight')
    plt.savefig(Path('../../../output/RESULT/analytical_model_sensitivity.png'), bbox_inches='tight')
