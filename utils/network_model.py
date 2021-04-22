"""
The analytical model class for an urban grid network.

Created by Michiel Aarts, March 2021
"""
from analytical import AnalyticalModel
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import scipy.optimize as opt
from typing import Tuple
from plugins.urban import UrbanGrid
from pathlib import Path
from scn_reader import plot_flow_rates
from bluesky.tools.aero import fpm, ft

# Vertical speed depends on the steepness used in bluesky/traffic/autopilot.py, adjust accordingly.
VS = 194. * fpm


class NetworkModel(AnalyticalModel):
    def __init__(self, urban_grid: UrbanGrid, max_value: float, accuracy: int, duration: Tuple[float, float, float],
                 speed: float, s_h: float, s_v: float, t_l: float, vs: float = VS):
        """
        Class for the analytical conflict count and delay models.

        :param urban_grid: UrbanGrid
        :param max_value: maximum inst. no. of aircraft to model for
        :param accuracy: number of evaluations modelled between 0 and max_value
        :param duration: simulation duration (build-up, logging, cool-down) [s, s, s]
        :param speed: autopilot speed of aircraft [m/s]
        :param s_h: horizontal separation distance [m]
        :param s_v: vertical separation distance [ft]
        :param t_l: look-ahead time [s]
        :param vs: vertical speed of aircraft [m/s] (optional)
        """
        super().__init__()

        # Parse inputs.
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
        self.mean_route_length = self.urban_grid.mean_route_length
        self.mean_flight_time_nr = self.mean_route_length / self.speed


        # Sanity check.
        if self.urban_grid.grid_height != self.urban_grid.grid_width or \
                self.urban_grid.n_rows != self.urban_grid.n_cols:
            raise NotImplementedError('Analytical model can only be determined for an equal grid size')

        # Initiate arrays.
        self.n_inst = np.linspace(1, self.max_value, self.accuracy)

        # Determine flow proportions and rates.
        self.n_total = self.n_inst * self.duration[1] / self.mean_flight_time_nr
        self.flow_proportion = self.expand_flow_proportion()
        self.flow_rates = self.determine_flow_rates()  # veh / s
        self.departure_rate = self.n_inst / self.mean_flight_time_nr  # veh / s
        self.arrival_rate = self.departure_rate  # By definition, for a stable system
        self.from_flow_rates = self.determine_from_flow_rates(self.flow_rates)  # veh / s

        # Instantaneous no. of departure / arrival aircraft.
        self.n_inst_da = (self.n_inst * self.flow_proportion.where(
            (self.flow_proportion.index.get_level_values('from') == 'departure')
            | (self.flow_proportion.index.get_level_values('via') == 'arrival')).sum())

        # NR conflict count model.
        self.c_inst_nr, self.los_inst_nr = self.nr_model()

        # WR delay model.
        self.delays = self.delay_model(self.from_flow_rates)
        self.mean_v_wr, self.n_inst_wr, self.mean_flight_time_wr, self.flow_rate_wr = self.wr_model()

        # Fitted variables.
        self.c_inst_wr_fitted = None
        self.mean_conflict_duration_nr = None  # s
        self.mean_conflict_duration_wr = None  # s
        self.c_total_nr = None
        self.c_total_wr = None
        self.c_total_wr_fitted = None
        self.false_conflict_ratio = None
        self.resolve_ratio = None
        self.los_total_nr = None
        self.los_total_wr = None
        self.mean_los_duration_nr = None  # s
        self.mean_los_duration_wr = None  # s
        self.los_inst_nr_fitted = None
        self.los_inst_wr = None

    def nr_model(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        The conflict count model without conflict resolution.
        Calculates the instantaneous number of conflicts and losses of separation,
        based on the flows through the urban grid.

        :return: (Inst. no. of conflicts, Inst. no. of LoS)
        """
        print('Calculating analytical NR model...')
        # Self interaction (LoS only). TODO: Include departing traffic?
        nr_ni_per_section = self.urban_grid.grid_height / self.speed * self.from_flow_rates.copy()
        nr_li_si_per_section = 0. * nr_ni_per_section.copy()
        for n in range(2, 100):
            poisson_prob_per_section = nr_ni_per_section.pow(n) * np.exp(-nr_ni_per_section) / np.math.factorial(n)
            nr_li_si_per_section += (poisson_prob_per_section * n * (n - 1) * self.s_h
                                     / (self.urban_grid.grid_height + self.s_h))
            if poisson_prob_per_section.max().max() < 1E-5:
                # Probability has become insignificant. Break loop.
                print(f'NR model: Poisson probability loop broken after P(x={n})')
                break
        nr_li_self_interaction = nr_li_si_per_section.sum()

        # Crossing conflicts and LoS.
        # Define areas.
        los_area = np.pi * np.power(self.s_h, 2)
        conf_vrel = 2 * self.speed * np.sin(np.deg2rad(90) / 2)
        conf_area = 2 * self.s_h * conf_vrel * self.t_l
        border_isct_area = 2 * np.power(self.urban_grid.grid_height, 2)
        regular_isct_area = 4 * np.power(self.urban_grid.grid_height, 2)

        # Obtain headings per intersection.
        from_idx = nr_ni_per_section.index.get_level_values('from')
        via_idx = nr_ni_per_section.index.get_level_values('via')
        hdg = (self.urban_grid.flow_df['from_hdg']
               .groupby(['from', 'via']).mean()
               .reindex_like(nr_ni_per_section))

        # Loop over all intersections.
        nr_li_crossing = 0. * nr_li_self_interaction.copy()
        nr_ci_crossing = 0. * nr_li_self_interaction.copy()
        for isct in self.urban_grid.isct_nodes:
            isct_ni = nr_ni_per_section[(from_idx == isct) | (via_idx == isct)].copy()
            isct_hdg = hdg[(from_idx == isct) | (via_idx == isct)].copy()
            ni_per_hdg = (isct_ni.join(isct_hdg)
                          .groupby('from_hdg').sum())
            if len(ni_per_hdg) != 2:
                # Sanity check.
                raise NotImplementedError(f'Intersections with {len(ni_per_hdg)} headings not implemented.')
            if len(isct_ni) == 2:
                # Corner node, skip.
                continue
            elif len(isct_ni) == 3:
                # Border intersection node.
                nr_li_crossing += ni_per_hdg.iloc[0] * ni_per_hdg.iloc[1] * los_area / border_isct_area
                nr_ci_crossing += ni_per_hdg.iloc[0] * ni_per_hdg.iloc[1] * conf_area / border_isct_area
            elif len(isct_ni) == 4:
                # Regular intersection node.
                nr_li_crossing += ni_per_hdg.iloc[0] * ni_per_hdg.iloc[1] * los_area / regular_isct_area
                nr_ci_crossing += ni_per_hdg.iloc[0] * ni_per_hdg.iloc[1] * conf_area / regular_isct_area
            else:
                # Sanity check.
                raise NotImplementedError(f'Intersections with {len(isct_ni)} segments not implemented.')

        # Sum self interaction and crossing conflicts / LoS.
        nr_li = nr_li_crossing + nr_li_self_interaction
        # Self interaction with equal speeds must be a LoS. Therefore, ci_self_interaction = departing traffic only.
        nr_ci = nr_ci_crossing

        return nr_ci, nr_li

    def determine_flow_rates(self) -> pd.DataFrame:
        """
        Determines flow rates through the UrbanGrid for each n_inst.

        :return: Flow rates dataframe
        """
        flow_rates = pd.DataFrame(index=self.urban_grid.flow_df.index, columns=self.n_inst)
        for ni in self.n_inst:
            passage_rate = ni * self.speed / self.urban_grid.grid_height
            flow_rates[ni] = self.urban_grid.flow_df['flow_distribution'] * passage_rate
        return flow_rates

    def determine_from_flow_rates(self, flow_df) -> pd.DataFrame:
        """
        Groups flow rates based on from and via nodes, to obtain the 'from flow'-rates.

        :param flow_df: Flow rates dataframe.
        :return: From flow rates dataframe
        """
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
        Extracts the flow proportion from the UrbanGrid and adds the departure and arrival proportions.
        Sum of flow proportion should be 1.

        :return: Flow proportion dataframe
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

                # If intersection unstable, set delay very large.
                stochastic_delay[total_y >= 1] = 1E5

                # Add to delays df.
                delays.loc[(from_nodes[i], node)] = general_delay + stochastic_delay

        delays[delays.isna()] = 0
        return delays

    def wr_model(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        The delay model with conflict resolution.
        Calculates the mean velocity, instantaneous number of aircraft,
        and the mean flight time of the flows through an urban grid.
        If the delay model predicts an unstable system, the values are set to zero.

        :return: (Mean velocity, No. Inst. A/C, Mean flight time, Flow rate)
        """
        print('Calculating analytical WR model...')
        max_ni_per_section = self.urban_grid.grid_height / self.s_h

        nr_flight_time_per_section = self.urban_grid.grid_height / self.speed
        wr_flight_time_per_section = self.delays.copy() + nr_flight_time_per_section
        wr_flight_time_per_section = wr_flight_time_per_section[
            wr_flight_time_per_section.index.get_level_values('from') != 'departure'
        ]
        wr_v_per_section = self.urban_grid.grid_height / wr_flight_time_per_section
        wr_separation_per_section = wr_v_per_section / self.from_flow_rates
        wr_ni_per_section = self.urban_grid.grid_height / wr_separation_per_section
        wr_ni = wr_ni_per_section.sum() + self.n_inst_da
        wr_mean_v = (wr_v_per_section * wr_ni_per_section.loc[wr_v_per_section.index]).sum() / wr_ni_per_section.sum()
        wr_mean_flight_time = self.mean_route_length / wr_mean_v

        # Filter unstable values and set to zero.
        unstable_filter = (wr_ni_per_section > max_ni_per_section).any()
        wr_mean_v[unstable_filter] = 0
        wr_ni[unstable_filter] = 0
        wr_mean_flight_time[unstable_filter] = 0
        wr_flow_rate = wr_mean_v * wr_ni

        return wr_mean_v, wr_ni, wr_mean_flight_time, wr_flow_rate

    def _fit_mean_duration(self, exp_inst: np.ndarray, exp_total: np.ndarray) -> float:
        """ Least squares fit of the mean conflict or los duration """
        if exp_inst.shape != exp_total.shape:
            raise ValueError('Inputs must be of same size')
        t_mean = opt.fmin(lambda a: np.nanmean(np.power(exp_inst * self.duration[1] / a - exp_total, 2)),
                          x0=2., disp=False)[0]
        return t_mean

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

    def fit_derivatives(self, data: dict) -> None:
        """
        Fits all derivatives and functions of a given simulation experiment result.
        E.g.: mean conflict duration, false conflict percentage, etc.

        :param data: Data dictionary from utils/log_reader.py
        """
        print('Fitting derivatives...')
        # Set all unstable data to nan.
        for reso in data.keys():
            for key in data[reso].keys():
                if key != 'stable_filter':
                    data[reso][key] = np.where(data[reso]['stable_filter'], data[reso][key], np.nan)

        self.c_inst_wr_fitted = self._fit_c_inst_wr(data['NR']['ni_ac'], data['WR']['ni_conf'])
        self.mean_conflict_duration_nr = self._fit_mean_duration(data['NR']['ni_conf'], data['NR']['ntotal_conf'])
        self.mean_conflict_duration_wr = self._fit_mean_duration(data['WR']['ni_conf'], data['WR']['ntotal_conf'])
        self.mean_los_duration_nr = self._fit_mean_duration(data['NR']['ni_los'], data['NR']['ntotal_los'])
        self.mean_los_duration_wr = self._fit_mean_duration(data['WR']['ni_los'], data['WR']['ntotal_los'])
        self.false_conflict_ratio = self._fit_conflict_los_ratio(data['NR']['ntotal_conf'],
                                                                 data['NR']['ntotal_los'])
        self.resolve_ratio = self._fit_conflict_los_ratio(data['WR']['ntotal_conf'], data['WR']['ntotal_los'])

        self.c_total_nr = self.c_inst_nr * self.duration[1] / self.mean_conflict_duration_nr
        self.c_total_wr = self.wr_conflict_model()
        self.c_total_wr_fitted = self.c_inst_wr_fitted * self.duration[1] / self.mean_conflict_duration_wr
        self.los_total_nr = self.c_total_nr * (1 - self.false_conflict_ratio)
        self.los_total_wr = self.c_total_wr * (1 - self.resolve_ratio)
        self.los_inst_nr_fitted = self.los_total_nr * self.mean_los_duration_nr / self.duration[1]
        self.los_inst_wr = self.los_total_wr * self.mean_los_duration_wr / self.duration[1]

    def wr_camda_model(self):
        """ Based on inst. no. of aircraft and CAMDA model """
        nr_fit = opt.fmin(lambda a: np.nanmean(np.power(4 * a * np.power(self.n_inst / 4, 2) - self.c_inst_nr, 2)),
                          x0=1, disp=False)[0]
        c_total_wr_ni = 4 * nr_fit * np.power(self.n_inst_wr / 4, 2) * self.duration[1] / self.mean_conflict_duration_nr
        c_1t = c_total_wr_ni / (self.n_total * self.mean_flight_time_nr)
        c_1_wr = c_1t * self.mean_flight_time_wr
        c_total_wr = c_1_wr * self.n_total
        return c_total_wr

    def wr_conflict_model(self) -> np.ndarray:
        """ Based on local flow rates and delay """
        vehicle_delay_per_second = self.delays * self.from_flow_rates
        vd_via_idx = vehicle_delay_per_second.index.get_level_values('via')
        ff_via_idx = self.from_flow_rates.index.get_level_values('via')
        additional_conflicts_per_second = pd.Series(0, index=self.n_inst)
        for node in self.urban_grid.all_nodes:
            # Loop over all intersections.
            node_delays_per_second = vehicle_delay_per_second[vd_via_idx == node].copy()
            node_from_flows = self.from_flow_rates[ff_via_idx == node].copy()
            node_from_flows_reversed = node_from_flows.copy().iloc[-1::-1]
            if len(node_delays_per_second) == 1:
                # Corner intersection, skip.
                continue
            elif len(node_delays_per_second) == 2:
                # Regular intersection.
                additional_conflicts_per_second += (node_delays_per_second * node_from_flows_reversed.values).sum()
                additional_conflicts_per_second += (node_delays_per_second * node_from_flows.values).sum()
            else:
                # Sanity check.
                raise NotImplementedError(f'Intersections with {len(node_delays_per_second)} headings not implemented.')

        c_total_wr = self.c_total_nr + additional_conflicts_per_second * self.duration[1]
        return c_total_wr


def plot_mfd(model: AnalyticalModel) -> None:
    """
    Plots the MFD of the provided analytical model.
    Useful when creating a scenario, to see whether the
    instantaneous numbers of aircraft without resolution
    are within the stable range.
    """
    plt.figure()
    plt.plot(model.n_inst, model.flow_rate_wr)
    plt.xlabel('No. of inst. aircraft NR')
    plt.ylabel('Flow rate [veh m / s]')


if __name__ == '__main__':
    S_H = 50.  # m
    S_V = 25.  # ft
    T_L = 20.  # s
    SPEED = 10.
    DURATION = (900., 2700., 900.)

    pkl_file = Path(r'../scenario/URBAN/Data/medium_ql_urban_grid.pkl')
    with open(pkl_file, 'rb') as f:
        grid = pkl.load(f)

    plot_flow_rates(grid.flow_df)

    ana_model = AnalyticalModel(grid, max_value=300, accuracy=50,
                                duration=DURATION, speed=SPEED, s_h=S_H, s_v=S_V, t_l=T_L)

    plot_mfd(ana_model)
