"""
The analytical model class for an urban grid network.

Created by Michiel Aarts, March 2021
"""
from typing import Tuple
import numpy.polynomial.polynomial as np_poly
import scipy.optimize as opt
import numpy as np
import pandas as pd
import pickle as pkl
from plugins.urban import UrbanGrid
from pathlib import Path
from scn_reader import plot_flow_rates
from bluesky.tools.aero import fpm, ft

# VS depends on the steepness used in autopilot.py, adjust accordingly.
VS = 194. * fpm


class AnalyticalModel:
    def __init__(
            self,
            urban_grid: UrbanGrid, max_value: float, accuracy: int, duration: Tuple[float, float, float],
            speed: float, s_h: float, s_v: float, t_l: float, vs: float = VS,
    ):
        self.urban_grid = urban_grid
        self.max_value = max_value
        self.accuracy = accuracy
        self.duration = duration
        self.speed = speed
        self.vs = vs
        self.s_h = s_h
        self.s_v = s_v * ft  # m
        self.t_l = t_l

        self.cruise_alt = 50. * ft  # m
        self.departure_alt = 0. * ft  # m
        self.avg_duration = self.urban_grid.avg_route_length / self.speed

        # Sanity check.
        if self.urban_grid.grid_height != self.urban_grid.grid_width or \
                self.urban_grid.n_rows != self.urban_grid.n_cols:
            raise NotImplementedError('Analytical model can only be determined for an equal grid size')

        self.n_inst = np.linspace(10, self.max_value, self.accuracy)
        self.n_total = self.n_inst * self.duration[1] / self.avg_duration

        self.flow_proportion = self.expand_flow_proportion()
        self.flow_rates = self.determine_flow_rates()
        self.departure_rate = self.determine_departure_rate()
        self.arrival_rate = self.departure_rate  # by definition, for a stable system.
        self.from_flow_rates = self.determine_from_flow_rates(self.flow_rates)

        # n_inst departure / arrival aircraft.
        self.n_inst_da = (self.n_inst * self.flow_proportion.where(
            (self.flow_proportion.index.get_level_values('from') == 'departure')
            | (self.flow_proportion.index.get_level_values('via') == 'arrival')).sum())

        # NR Model
        self.c_inst_nr, self.los_inst_nr = self.nr_model()

        # Fitted variables
        self.c_inst_wr_fitted = None
        self.avg_conflict_duration_nr = None  # s
        self.avg_conflict_duration_wr = None  # s
        self.c_total_nr = None
        self.c_total_wr = None
        self.false_conflict_ratio = None
        self.resolve_ratio = None
        self.los_total_nr = None
        self.los_total_wr = None
        self.avg_los_duration_nr = None  # s
        self.avg_los_duration_wr = None  # s
        self.los_inst_nr_fitted = None
        self.los_inst_wr = None

        # WR Model
        self.delays = self.delay_model(self.from_flow_rates)

        self.mean_v_wr, self.n_inst_wr, self.mean_duration_wr = self.wr_model()

    def nr_model(self) -> Tuple[np.ndarray, np.ndarray]:
        print('Calculating analytical NR model...')
        # Conflicts --------
        # Crossing flows.
        vrel = 2 * self.speed * np.sin(np.deg2rad(90) / 2)
        c_inst_nr_crossing = (4 * np.power(self.n_inst / 4, 2) * 2 * self.s_h * vrel * self.t_l
                              / self.urban_grid.area)

        # Self interaction with departing traffic. Same as crossing, but in xz-plane.
        # TODO: Add depth (equivalent to altitude layers).
        alt_to_climb = self.cruise_alt - self.departure_alt
        time_to_climb = alt_to_climb / self.vs
        n_inst_departing = self.departure_rate * time_to_climb
        n_inst_cruise = self.n_inst - n_inst_departing
        xz_area = alt_to_climb * np.sqrt(self.urban_grid.area)
        # c_inst_nr_departing = n_inst_departing * n_inst_cruise * self.s_v * self.vs * self.t_l / xz_area
        c_inst_nr_departing = 0 * n_inst_departing

        c_inst_nr = c_inst_nr_crossing + c_inst_nr_departing

        # LoS ---------
        # los_inst_nr_crossing = (4 * np.power(self.n_inst / 4, 2) * np.pi * np.power(self.s_h, 2)
        #                         / self.urban_grid.area)
        los_inst_nr_crossing = (np.power(self.n_inst, 2) * np.pi * np.power(self.s_h, 2)
                                / self.urban_grid.area)


        # Self interaction with departing traffic.
        # TODO: Add depth (equivalent to altitude layers).
        # los_inst_nr_departing = n_inst_departing * n_inst_cruise * self.s_v * self.s_h / xz_area
        los_inst_nr_departing = 0 * n_inst_departing

        los_inst_nr = los_inst_nr_crossing + los_inst_nr_departing

        return c_inst_nr, los_inst_nr

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
            if node in self.urban_grid.od_nodes:
                # Time to merge an altitude layer [s].
                # TODO: does the s_v corner of the disc play a role here?
                t_x = self.s_h / self.speed
            else:
                # Time to cross an intersection [s]. Change sqrt(2) if angle between airways is not 90 degrees.
                t_x = self.s_h * np.sqrt(2) / self.speed
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
            total_y = total_q * t_x
            total_stochastic_delay = total_y * total_y / (2 * total_q * (1 - total_y))
            for (i, j) in [(1, 0), (0, 1)]:
                # General delay.
                q_g = node_flows.iloc[i].squeeze()
                q_r = node_flows.iloc[j].squeeze()
                lambda_u = 1 - t_x * q_r
                c_u = 1 / q_r
                y = q_g * t_x
                general_delay = c_u * np.power(1 - lambda_u, 2) / (2 * (1 - y))

                # Stochastic delay.
                stochastic_flow_delay = y * y / (2 * q_g * (1 - y))
                stochastic_delay = total_stochastic_delay - stochastic_flow_delay

                # Add to delays df.
                delays.loc[(from_nodes[i], node)] = general_delay + stochastic_delay

        delays[delays.isna()] = 0
        return delays

    def wr_model(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        print('Calculating analytical WR model...')
        # Initiate arrays.
        ni = pd.DataFrame({0: self.n_inst}, index=self.n_inst)
        mean_delay = pd.DataFrame().reindex_like(ni)
        mean_duration = pd.DataFrame(np.ones(self.n_inst.shape) * self.avg_duration, index=self.n_inst)
        mean_v = pd.DataFrame(np.ones(self.n_inst.shape) * self.speed, index=self.n_inst)

        # Calculate nr case (as check).
        max_ni_per_section = self.urban_grid.grid_height / self.s_h
        nr_duration_per_section = pd.DataFrame(self.urban_grid.grid_height / self.speed,
                                               index=self.delays.index, columns=self.delays.columns)
        nr_duration_per_section = nr_duration_per_section[
            nr_duration_per_section.index.get_level_values('from') != 'departure']
        nr_v_per_section = self.urban_grid.grid_height / nr_duration_per_section
        nr_separation_per_section = nr_v_per_section / self.from_flow_rates
        nr_ni_per_section = self.urban_grid.grid_height / nr_separation_per_section
        nr_ni = nr_ni_per_section.sum() + self.n_inst_da
        nr_mean_v = (nr_v_per_section * nr_ni_per_section.loc[nr_v_per_section.index]).sum() / nr_ni_per_section.sum()

        # Calculate wr case.
        wr_duration_per_section = self.delays.loc[nr_duration_per_section.index] + nr_duration_per_section
        wr_v_per_section = self.urban_grid.grid_height / wr_duration_per_section
        wr_separation_per_section = wr_v_per_section / self.from_flow_rates
        wr_ni_per_section = self.urban_grid.grid_height / wr_separation_per_section
        wr_ni = wr_ni_per_section.sum() + self.n_inst_da
        wr_mean_v = (wr_v_per_section * wr_ni_per_section.loc[wr_v_per_section.index]).sum() / wr_ni_per_section.sum()
        wr_mean_duration = self.urban_grid.avg_route_length / wr_mean_v

        # Filter unstable values.
        unstable_filter = (wr_ni_per_section > max_ni_per_section).any()
        wr_mean_v[unstable_filter] = 0
        wr_ni[unstable_filter] = 0
        wr_mean_duration[unstable_filter] = 0

        return wr_mean_v, wr_ni, wr_mean_duration

    def _fit_avg_duration(self, exp_inst: np.ndarray, exp_total: np.ndarray) -> float:
        """ Least squares fit of the avg. conflict or los duration """
        if exp_inst.shape != exp_total.shape:
            raise ValueError('Inputs must be of same size')
        t_avg = opt.fmin(lambda a: np.nanmean(np.power(exp_inst * self.duration[1] / a - exp_total, 2)),
                         x0=2., disp=False)[0]
        return t_avg

    @staticmethod
    def _fit_conflict_los_ratio(exp_c_total: np.ndarray, exp_los_total: np.ndarray) -> float:
        """ Least squares fit of the false conflict ratio """
        if exp_c_total.shape != exp_los_total.shape:
            raise ValueError('Inputs must be of same size')
        ratio = opt.fmin(lambda a: np.nanmean(np.power(exp_c_total * a - exp_los_total, 2)),
                         x0=0.5, disp=False)[0]
        return 1 - ratio

    def _fit_c_inst_wr(self, exp_n_inst_nr: np.ndarray, exp_c_inst_wr: np.ndarray) -> np.ndarray:
        """ Least squares, second degree polynomial fit of the instantaneous number of conflicts with resolution """
        res = opt.fmin(lambda a: np.nanmean(np.power(a * exp_n_inst_nr * exp_n_inst_nr - exp_c_inst_wr, 2)),
                       x0=1, disp=False)[0]
        return res * np.power(self.n_inst, 2)

    def fit_derivatives(self, data: dict):
        print('Fitting derivatives...')
        # Set all unstable data to nan
        for reso in data.keys():
            for key in data[reso].keys():
                if key != 'stable_filter':
                    data[reso][key] = np.where(data[reso]['stable_filter'], data[reso][key], np.nan)

        self.c_inst_wr_fitted = self._fit_c_inst_wr(data['NR']['ni_ac'], data['WR']['ni_conf'])
        self.avg_conflict_duration_nr = self._fit_avg_duration(data['NR']['ni_conf'], data['NR']['ntotal_conf'])
        self.avg_conflict_duration_wr = self._fit_avg_duration(data['WR']['ni_conf'], data['WR']['ntotal_conf'])
        self.avg_los_duration_nr = self._fit_avg_duration(data['NR']['ni_los'], data['NR']['ntotal_los'])
        self.avg_los_duration_wr = self._fit_avg_duration(data['WR']['ni_los'], data['WR']['ntotal_los'])
        self.false_conflict_ratio = self._fit_conflict_los_ratio(data['NR']['ntotal_conf'],
                                                                 data['NR']['ntotal_los'])
        self.resolve_ratio = self._fit_conflict_los_ratio(data['WR']['ntotal_conf'], data['WR']['ntotal_los'])

        self.c_total_nr = self.c_inst_nr * self.duration[1] / self.avg_conflict_duration_nr
        self.c_total_wr = self.c_inst_wr_fitted * self.duration[1] / self.avg_conflict_duration_wr
        self.los_total_nr = self.c_total_nr * (1 - self.false_conflict_ratio)
        self.los_total_wr = self.c_total_wr * (1 - self.resolve_ratio)
        self.los_inst_nr_fitted = self.los_total_nr * self.avg_los_duration_nr / self.duration[1]
        self.los_inst_wr = self.los_total_wr * self.avg_los_duration_wr / self.duration[1]


if __name__ == '__main__':
    S_H = 50.  # m
    S_V = 25.  # ft
    T_L = 20.  # s
    SPEED = 10.
    DURATION = (900., 2700., 900.)

    pkl_file = Path(r'../scenario/URBAN/Data/validation_urban_grid.pkl')
    with open(pkl_file, 'rb') as f:
        grid = pkl.load(f)

    plot_flow_rates(grid.flow_df)

    ana_model = AnalyticalModel(grid, max_value=300, accuracy=20,
                                duration=DURATION, speed=SPEED, s_h=S_H, s_v=S_V, t_l=T_L)
