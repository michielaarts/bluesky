"""
The analytical model class for an urban grid network.

Created by Michiel Aarts, March 2021
"""

import numpy as np
import pandas as pd
import pickle as pkl
from plugins.urban import UrbanGrid
from pathlib import Path


class AnalyticalModel:
    def __init__(self, urban_grid: UrbanGrid):
        self.urban_grid = urban_grid
        self.all_angles = np.array(range(9)) * 45
        self.corner_angles = np.array(range(4)) * 90 + 45

    def delay_model(self):
        pass


if __name__ == '__main__':
    pkl_file = Path(r'../scenario/URBAN/Data/test1_urban_grid.pkl')
    with open(pkl_file, 'rb') as f:
        grid = pkl.load(f)

    ana_model = AnalyticalModel(grid)
