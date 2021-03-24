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

VS = 113. * fpm


class AnalyticalModel:
    def __init__(
            self,
            urban_grid: UrbanGrid, max_value: float, accuracy: int,
            speed: float, s_h: float, s_v: float, t_l: float, vs: float = VS,
    ):
        self.urban_grid = urban_grid
        self.max_value = max_value
        self.accuracy = accuracy
        self.speed = speed
        self.vs = vs
        self.s_h = s_h
        self.s_v = s_v
        self.t_l = t_l

        self.cruise_alt = 50.
        self.departure_alt = self.cruise_alt - self.s_v * 1.5
        self.t_X = self.s_h / self.speed  # Time to cross an intersection [s]
        self.avg_duration = self.urban_grid.avg_route_length / self.speed

        # Sanity check.
        if self.urban_grid.grid_height != self.urban_grid.grid_width or \
                self.urban_grid.n_rows != self.urban_grid.n_cols:
            raise NotImplementedError('Analytical model can only be determined for an equal grid size')

        self.n_inst = np.linspace(10, self.max_value, self.accuracy)

        self.flow_proportion = self.expand_flow_proportion()
        self.flow_rates = self.determine_flow_rates()
        self.departure_rate = self.determine_departure_rate()
        self.arrival_rate = self.departure_rate  # by definition, for a stable system.
        self.from_flow_rates = self.determine_from_flow_rates(self.flow_rates)

        self.c_inst_nr = self.nr_model()

        self.delays = self.delay_model(self.from_flow_rates)

        self.mean_v_wr, self.n_inst_wr, self.mean_duration_wr = self.wr_model()

    def nr_model(self) -> np.ndarray:
        # Crossing flows.
        vrel = 2 * self.speed * np.sin(np.deg2rad(90) / 2)
        c_inst_nr_crossing = 4 * np.power(self.n_inst / 4, 2) * 2 * self.s_h * vrel * self.t_l / self.urban_grid.area

        # Self interaction with departing traffic. Same as crossing, but in xz-plane.
        alt_to_climb = self.cruise_alt - self.departure_alt
        time_to_climb = alt_to_climb / self.vs
        n_inst_departing = self.departure_rate * time_to_climb
        n_inst_cruise = self.n_inst - n_inst_departing
        area = alt_to_climb * np.sqrt(self.urban_grid.area)
        c_inst_nr_departing = n_inst_departing * n_inst_cruise * 2 * self.s_v * self.vs * self.t_l / area

        return c_inst_nr_crossing + c_inst_nr_departing

    def determine_flow_rates(self) -> pd.DataFrame:
        flow_rates = pd.DataFrame(index=self.urban_grid.flow_df.index, columns=self.n_inst)
        for ni in self.n_inst:
            passage_rate = ni * self.speed / self.urban_grid.grid_height
            flow_rates[ni] = self.urban_grid.flow_df['flow_distribution'] * passage_rate
        return flow_rates

    def determine_departure_rate(self) -> np.ndarray:
        avg_route_duration = self.urban_grid.avg_route_length / self.speed
        spawn_rate = self.n_inst / avg_route_duration
        return spawn_rate

    def determine_from_flow_rates(self, flow_df) -> pd.DataFrame:
        from_flows = flow_df.groupby(['from', 'via']).sum()

        departure_flows = pd.DataFrame(
            index=[['departure'] * len(self.urban_grid.od_nodes), self.urban_grid.od_nodes],
            columns=from_flows.columns)
        departure_rate_per_node = self.departure_rate / len(self.urban_grid.od_nodes)
        data = np.ones(departure_flows.shape) * departure_rate_per_node
        departure_flows = pd.DataFrame(data, index=departure_flows.index, columns=departure_flows.columns)

        from_flows = from_flows.append(departure_flows)
        return from_flows

    def expand_flow_proportion(self) -> pd.Series:
        """
        Sum of flow proportion should be 1.

        :return:
        """
        from_proportion = self.urban_grid.flow_df['flow_distribution'].groupby(['from', 'via']).sum()
        departure_proportion = self.urban_grid.flow_df['origin_distribution'].mean()
        arrival_proportion = self.urban_grid.flow_df['destination_distribution'].mean()
        departure_flows = pd.Series(departure_proportion,
                                    index=[['departure'] * len(self.urban_grid.od_nodes), self.urban_grid.od_nodes])
        arrival_flows = pd.Series(arrival_proportion,
                                  index=[self.urban_grid.od_nodes, ['arrival'] * len(self.urban_grid.od_nodes)])
        return from_proportion.append(departure_flows).append(arrival_flows)

    def delay_model(self, flow_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the delay per vehicle at each node for the provided from_flow dataframe.

        :param flow_df: a from_flow dataframe (from e.g. determine_from_flow_rates())
        :return: Delay dataframe
        """
        delays = pd.DataFrame().reindex_like(flow_df)

        # Delay model.
        for node in self.urban_grid.all_nodes:
            node_flows = flow_df.iloc[
                flow_df.index.get_level_values('via') == node
            ].copy()
            from_nodes = node_flows.index.get_level_values('from').unique()
            if len(from_nodes) == 1:
                # Border node, no intersection. Skip.
                continue
            elif len(from_nodes) != 2:
                # Sanity check.
                raise ValueError(f'Intersection with {len(from_nodes)} directions found!\n', node_flows)

            total_q = node_flows.sum()
            total_y = total_q * self.t_X
            total_stochastic_delay = total_y * total_y / (2 * total_q * (1 - total_y))
            for (i, j) in [(1, 0), (0, 1)]:
                # General delay.
                q_g = node_flows.iloc[i].squeeze()
                q_r = node_flows.iloc[j].squeeze()
                lambda_u = 1 - self.t_X * q_r
                c_u = 1 / q_r
                y = q_g * self.t_X
                general_delay = c_u * np.power(1 - lambda_u, 2) / (2 * (1 - y))

                # Stochastic delay.
                stochastic_flow_delay = y * y / (2 * q_g * (1 - y))
                stochastic_delay = total_stochastic_delay - stochastic_flow_delay

                # Add to delays df.
                delays.loc[(from_nodes[i], node)] = general_delay + stochastic_delay

        return delays

    def wr_model(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Initiate arrays.
        ni = pd.DataFrame({0: self.n_inst}, index=self.n_inst)
        from_flows = [self.from_flow_rates]
        delays = [self.from_flow_rates * 0]
        mean_delay = pd.DataFrame().reindex_like(ni)
        mean_duration = pd.DataFrame(np.ones(self.n_inst.shape) * self.avg_duration, index=self.n_inst)
        mean_v = pd.DataFrame(np.ones(self.n_inst.shape) * self.speed, index=self.n_inst)

        # Iterate.
        num_iterations = 10
        for i in range(1, num_iterations + 1):
            # Calculate delays based on current flow rates.
            delays.append(self.delay_model(from_flows[i - 1]))
            for col in delays[i].columns:
                delays[i][col] = delays[i][col] * self.flow_proportion.loc[delays[i].index]
            mean_delay[i] = delays[i].sum() * ni[i - 1]

            # Mean velocity scales proportionally with delay.
            mean_duration[i] = mean_delay[i] + self.avg_duration
            mean_v[i] = self.speed * (self.avg_duration / mean_duration[i])
            # Negative or increasing mean velocities signify an unstable system.
            mean_v[i].where((mean_v[i] > 0) & (mean_v[i] <= mean_v[i - 1]), np.nan, inplace=True)

            # Inst. amount of vehicles scales proportionally with mean_velocity.
            ni[i] = self.n_inst * mean_v[0] / mean_v[i]
            from_flows.append(from_flows[0] / self.n_inst * ni[i])

        # Extract last column and replace first nan.
        last_col = max(mean_v.columns)
        mean_v_wr = mean_v[last_col].values
        n_inst_wr = ni[last_col].values
        mean_duration_wr = mean_duration[last_col].values

        nanidx = np.argwhere(np.isnan(mean_v_wr))
        if len(nanidx) > 0:
            mean_v_wr[nanidx[0]] = 0.
            n_inst_wr[nanidx[0]] = n_inst_wr[nanidx[0] - 1] * 5.
            mean_duration_wr[nanidx[0]] = mean_duration_wr[nanidx[0] - 1] * 5.

        return mean_v_wr, n_inst_wr, mean_duration_wr


if __name__ == '__main__':
    S_H = 50.  # m
    S_V = 25.  # ft
    T_L = 20.  # s
    SPEED = 10.

    pkl_file = Path(r'../scenario/URBAN/Data/test1_urban_grid.pkl')
    with open(pkl_file, 'rb') as f:
        grid = pkl.load(f)

    plot_flow_rates(grid.flow_df)

    ana_model = AnalyticalModel(grid, max_value=300, accuracy=10, speed=SPEED, s_h=S_H, s_v=S_V, t_l=T_L)
