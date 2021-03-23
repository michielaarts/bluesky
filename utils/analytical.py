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
        self.t_X = self.s_h / self.speed  # Time to cross an intersection [s]

        # Sanity check.
        if self.urban_grid.grid_height != self.urban_grid.grid_width or \
                self.urban_grid.n_rows != self.urban_grid.n_cols:
            raise NotImplementedError('Analytical model can only be determined for an equal grid size')

        self.n_inst = np.linspace(10, self.max_value, 10)

        self.flow_rates = self.determine_flow_rates()
        self.departure_rate = self.determine_departure_rate()

        self.area = np.power(self.urban_grid.grid_height * 1000 * (self.urban_grid.n_rows - 1), 2)
        self.c_inst_nr = self.nr_model()

        self.delays = self.delay_model(self.flow_rates)

        self.wr_model()

    def nr_model(self) -> np.ndarray:
        # Crossing flows.
        vrel = 2 * self.speed * np.sin(np.deg2rad(90) / 2)
        c_inst_nr_crossing = 4 * np.power(self.n_inst / 4, 2) * 2 * self.s_h * vrel * self.t_l / self.area

        # Self interaction with departing traffic. Same as crossing, but in xz-plane.
        alt_to_climb = self.cruise_alt - self.departure_alt
        time_to_climb = alt_to_climb / self.vs
        n_inst_departing = self.departure_rate * time_to_climb
        n_inst_cruise = self.n_inst - n_inst_departing
        area = alt_to_climb * np.sqrt(self.area)
        c_inst_nr_departing = n_inst_departing * n_inst_cruise * 2 * self.s_v * self.vs * self.t_l / area

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

    def general_delay(self):
        pass

    @staticmethod
    def stochastic_delay():
        pass

    def delay_model(self, flow_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the delay per vehicle at each node for the provided flow dataframe.

        :param flow_df:
        :return: Delay dataframe
        """
        from_flows = flow_df.groupby(['from', 'via']).sum()
        delays = pd.DataFrame().reindex_like(from_flows)
        delays = delays.append(
            pd.DataFrame(index=[['departure'] * len(self.urban_grid.od_nodes), self.urban_grid.od_nodes],
                         columns=delays.columns))

        # Intersection delays.
        # Group by direction per intersection.
        for isct in self.urban_grid.isct_nodes:
            isct_flows = from_flows.iloc[
                from_flows.index.get_level_values('via') == isct
            ].copy()
            from_nodes = isct_flows.index.get_level_values('from').unique()
            if len(from_nodes) == 1:
                # Border node, no intersection. Skip.
                continue
            elif len(from_nodes) != 2:
                # Sanity check.
                raise ValueError(f'Intersection with {len(from_nodes)} directions found!\n', isct_flows)

            total_q = isct_flows.sum()
            total_y = total_q * self.t_X
            total_stochastic_delay = total_y * total_y / (2 * total_q * (1 - total_y))
            for (i, j) in [(1, 0), (0, 1)]:
                # General delay.
                q_g = isct_flows.iloc[i].squeeze()
                q_r = isct_flows.iloc[j].squeeze()
                lambda_u = 1 - self.t_X * q_r
                c_u = 1 / q_r
                y = q_g * self.t_X
                general_delay = c_u * np.power(1 - lambda_u, 2) / (2 * (1 - y))

                # Stochastic delay.
                stochastic_flow_delay = y * y / (2 * q_g * (1 - y))
                stochastic_delay = total_stochastic_delay - stochastic_flow_delay

                # Add to delays df.
                delays.loc[(from_nodes[i], isct)] = general_delay + stochastic_delay

        # Departure delays.
        departure_flow_per_node = self.departure_rate / len(self.urban_grid.od_nodes)
        for origin in self.urban_grid.od_nodes:
            origin_flows = from_flows.iloc[from_flows.index.get_level_values('via') == origin]
            from_nodes = [origin_flows.index.get_level_values('from')[0], 'departure']
            passing_flow = origin_flows.squeeze()

            total_q = passing_flow + departure_flow_per_node
            total_y = total_q * self.t_X
            total_stochastic_delay = total_y * total_y / (2 * total_q * (1 - total_y))
            flows = [passing_flow, departure_flow_per_node]
            for (i, j) in [(1, 0), (0, 1)]:
                q_g = flows[i]
                q_r = flows[j]

                # General delay.
                lambda_u = 1 - self.t_X * q_r
                c_u = 1 / q_r
                y = q_g * self.t_X
                general_delay = c_u * np.power(1 - lambda_u, 2) / (2 * (1 - y))

                # Stochastic delay.
                stochastic_flow_delay = y * y / (2 * q_g * (1 - y))
                stochastic_delay = total_stochastic_delay - stochastic_flow_delay

                delays.loc[(from_nodes[i], origin)] = general_delay + stochastic_delay

        return delays

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
