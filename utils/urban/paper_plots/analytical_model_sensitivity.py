"""
This file assesses the sensitivity of the analytical grid network model for various airspace design parameters.

Created by Michiel Aarts, May 2021.
"""

import numpy as np
from utils.urban.network_model import NetworkModel
from utils.urban.urban_grid_network import UrbanGrid

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


def max_capacity(model: NetworkModel) -> float:
    """ Extracts the maximum capacity of a network model according to the assumption:
    once the mean queue length of an intersection flow is longer than the distance to
    the upstream intersection of that flow, the entire airspace becomes unstable.

    :param model: Analytical grid network model
    :return: Maximum capacity density (inst. no. of aircraft).
    """
    return np.nanmax(model.n_inst_wr)


if __name__ == '__main__':
    all_capacities = dict()
    all_capacities['base'] = max_capacity(BASE_MODEL)

    # Speed x2.
    v2_model = BASE_MODEL.copy()
    v2_model.speed *= 2.
    v2_model.calculate_models()
    all_capacities['V_NR (x2)'] = max_capacity(v2_model)

    # S_h x2.
    sh_model = BASE_MODEL.copy()
    sh_model.s_h *= 2.
    sh_model.calculate_models()
    all_capacities['S_h (x2)'] = max_capacity(sh_model)

    # Grid node distance x2.
    grid_grid = UrbanGrid(N_ROWS, N_COLS, GRID_WIDTH * 2., GRID_HEIGHT * 2.)
    grid_model = BASE_MODEL.copy()
    grid_model.urban_grid = grid_grid
    grid_model.calculate_models()
    all_capacities['D (x2)'] = max_capacity(grid_model)

    print(all_capacities)
