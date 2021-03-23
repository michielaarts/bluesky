"""
The analytical model class for an urban grid network.

Created by Michiel Aarts, March 2021
"""
from typing import Tuple

import numpy as np
import pandas as pd
import pickle as pkl
from plugins.urban import UrbanGrid
from pathlib import Path
from scn_reader import plot_flow_rates
from bluesky.tools.aero import fpm

VS = 113. / fpm


class AnalyticalModel:
    def __init__(
            self,
            urban_grid: UrbanGrid, max_value: float,
            speed: float, s_h: float, s_v: float, t_l: float, vs: float = VS,
    ):
        self.urban_grid = urban_grid
        self.max_value = max_value
        self.speed = speed
        self.vs = vs
        self.s_h = s_h
        self.s_v = s_v
        self.t_l = t_l

        self.cruise_alt = 50.
        self.departure_alt = self.cruise_alt - self.s_v * 1.5

        # Sanity check.
        if self.urban_grid.grid_height != self.urban_grid.grid_width or \
                self.urban_grid.n_rows != self.urban_grid.n_cols:
            raise NotImplementedError('Analytical model can only be determined for an equal grid size')

        self.n_inst = np.linspace(10, self.max_value, 10)

        self.flow_rates = self.determine_flow_rates()
        self.departure_rate = self.determine_departure_rate()

        self.area = np.power(self.urban_grid.grid_height * 1000 * (self.urban_grid.n_rows - 1), 2)
        self.c_inst_nr = self.nr_model()

        self.delay_model()
        self.wr_model()

    def nr_model(self) -> np.ndarray:
        # Crossing flows.
        vrel = 2 * self.speed * np.sin(np.deg2rad(90) / 2)
        c_inst_nr_crossing = 4 * np.power(self.n_inst / 4, 2) * 2 * self.s_h * vrel * self.t_l / self.area

        # Self interaction with departing traffic. Same as crossing, but in xz-plane.
        vrel = self.vs
        alt_to_climb = self.cruise_alt - self.departure_alt
        time_to_climb = alt_to_climb / self.vs
        n_inst_departing = self.departure_rate * time_to_climb
        n_inst_cruise = self.n_inst - n_inst_departing
        area = alt_to_climb * np.sqrt(self.area)
        c_inst_nr_departing = n_inst_departing * n_inst_cruise * 2 * self.s_v * vrel * self.t_l / area
        return c_inst_nr_crossing + c_inst_nr_departing

    def determine_flow_rates(self) -> pd.DataFrame:
        flow_rates = pd.DataFrame(index=self.urban_grid.flow_df.index, columns=self.n_inst)
        for ni in self.n_inst:
            passage_rate = ni * self.speed / (self.urban_grid.grid_height * 1000)
            flow_rates[ni] = self.urban_grid.flow_df['flow_distribution'] * passage_rate
        return flow_rates

    def determine_departure_rate(self) -> np.ndarray:
        avg_route_duration = self.urban_grid.avg_route_length * 1000 / self.speed
        spawn_rate = self.n_inst / avg_route_duration
        return spawn_rate

    @staticmethod
    def general_delay():
        pass

    @staticmethod
    def stochastic_delay():
        pass

    def delay_model(self) -> None:
        for ni in self.n_inst:
            pass

    def wr_model(self) -> None:
        pass


if __name__ == '__main__':
    S_H = 50.  # m
    S_V = 25.  # ft
    T_L = 20.  # s
    SPEED = 10.

    pkl_file = Path(r'../scenario/URBAN/Data/test1_urban_grid.pkl')
    with open(pkl_file, 'rb') as f:
        grid = pkl.load(f)

    plot_flow_rates(grid.flow_df)

    ana_model = AnalyticalModel(grid, max_value=500, speed=SPEED, s_h=S_H, s_v=S_V, t_l=T_L)
