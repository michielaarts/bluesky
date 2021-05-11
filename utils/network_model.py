"""
The analytical model class for an urban grid network.

Created by Michiel Aarts, March 2021
"""
import numpy as np
import pandas as pd
import pickle as pkl
import scipy.optimize as opt
from typing import Tuple
from pathlib import Path
from plugins.urban import UrbanGrid
from utils.analytical import AnalyticalModel
from utils.scn_reader import plot_flow_rates
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
            raise NotImplementedError('Analytical model can only be determined for a square grid')

        # Initiate arrays.
        self.n_inst = np.linspace(1, self.max_value, self.accuracy)

        # Calculate models.
        self.calculate_models()

    def calculate_models(self):
        # Determine flow proportions and rates.
        self.n_total = self.n_inst * self.duration[1] / self.mean_flight_time_nr
        self.flow_rates = self.determine_flow_rates()  # veh / s
        self.departure_rate = pd.Series(self.n_inst / self.mean_flight_time_nr, index=self.n_inst)  # veh / s
        self.departure_rate_per_node = self.departure_rate / len(self.urban_grid.od_nodes)
        self.arrival_rate = self.departure_rate  # By definition, for a stable system
        self.from_flow_rates = self.determine_from_flow_rates()  # veh / s
        self.extended_from_flow_rates = self.determine_extended_from_flow_rates()  # veh / s

        # NR conflict count model.
        self.c_inst_nr, self.los_inst_nr, self.c_total_nr, self.los_total_nr = self.nr_model()

        # WR delay model.
        self.delays = self.delay_model()
        self.mean_v_wr, self.n_inst_wr, self.mean_flight_time_wr, self.flow_rate_wr, self.delay_wr = self.wr_model()

        # WR conflict count model.
        self.c_total_wr = self.wr_conflict_model()
        self.dep = (self.c_total_wr / self.c_total_nr) - 1.

    def determine_flow_rates(self) -> pd.DataFrame:
        """
        Calculates (from, via, to) flow rates based on flow proportion.

        :return: Flow rates dataframe
        """
        flow_rates = pd.DataFrame(index=self.urban_grid.flow_df.index, columns=self.n_inst)
        for ni in self.n_inst:
            passage_rate = ni * self.speed / self.urban_grid.grid_height
            flow_rates[ni] = self.urban_grid.flow_df['flow_distribution'] * passage_rate
        return flow_rates

    def determine_from_flow_rates(self) -> pd.DataFrame:
        """
        Determines the flow rates in each section of the grid.

        :return: From flow rates dataframe
        """
        from_flow_rates = pd.DataFrame(index=self.urban_grid.from_flow_df.index, columns=self.n_inst)
        for ni in self.n_inst:
            passage_rate = ni * self.speed / self.urban_grid.grid_height
            from_flow_rates[ni] = self.urban_grid.from_flow_df['flow_distribution'] * passage_rate
        return from_flow_rates

    def determine_extended_from_flow_rates(self) -> pd.DataFrame:
        """
        Appends the departure rates to the from flow df.

        :return: extended from flows dataframe
        """
        departure_index = pd.MultiIndex.from_frame(pd.DataFrame({'from': ['departure'] * len(self.urban_grid.od_nodes),
                                                                 'via': self.urban_grid.od_nodes}))
        departure_df = pd.DataFrame(index=departure_index, columns=self.n_inst)
        for i in range(len(departure_index)):
            departure_df.iloc[i] = self.departure_rate_per_node
        ff_df = self.from_flow_rates.append(departure_df)
        return ff_df

    def nr_model(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        The conflict count model without conflict resolution.
        Calculates the instantaneous number of conflicts and losses of separation,
        based on the flows through the urban grid.

        :return: (Inst. no. of conflicts, Inst. no. of LoS)
        """
        print('Calculating analytical NR model...')
        nr_ni_per_section = self.urban_grid.grid_height / self.speed * self.from_flow_rates.copy()

        # Self interaction (LoS only).
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

        # Obtain index arrays.
        from_idx = nr_ni_per_section.index.get_level_values('from')
        via_idx = nr_ni_per_section.index.get_level_values('via')

        # Loop over all intersections.
        nr_li_crossing = 0. * nr_li_self_interaction.copy()
        nr_ci_crossing = 0. * nr_li_self_interaction.copy()
        for isct in self.urban_grid.isct_nodes:
            isct_ni = nr_ni_per_section[(from_idx == isct) | (via_idx == isct)]
            upstream_ni = nr_ni_per_section[via_idx == isct]
            if len(isct_ni) == 2:
                # Corner node, skip.
                continue
            elif len(isct_ni) == 3:
                # Border intersection node.
                if len(upstream_ni) == 2:
                    # 2 flows merging into 1.
                    # In no. of conflicts equivalent to a regular intersection node.
                    combinations = 2 * upstream_ni.iloc[0] * upstream_ni.iloc[1]
                    nr_li_crossing += combinations * los_area / border_isct_area
                    nr_ci_crossing += combinations * conf_area / border_isct_area
                # Else: 1 flow separating into 2 flows: Only turning conflicts.
            elif len(isct_ni) == 4:
                # Regular intersection node.
                combinations = 4 * upstream_ni.iloc[0] * upstream_ni.iloc[1]
                nr_li_crossing += combinations * los_area / regular_isct_area
                nr_ci_crossing += combinations * conf_area / regular_isct_area
            else:
                # Sanity check.
                raise NotImplementedError(f'Intersections with {len(isct_ni)} segments not implemented.')

        # Sum self interaction and crossing conflicts / LoS.
        nr_li = nr_li_crossing + nr_li_self_interaction
        # Self interaction with equal speeds must be a LoS.
        nr_ci = nr_ci_crossing
        # Calculate total using mean conflict duration of lookahead time.
        nr_ctotal = nr_ci * self.duration[1] / self.t_l
        # Each conflict should result in a LoS.
        nr_lostotal = nr_ctotal

        return nr_ci, nr_li, nr_ctotal, nr_lostotal

    def delay_model(self) -> pd.DataFrame:
        """
        Calculates the delay per vehicle at each node for the extended from_flow dataframe.

        Delay model has two parts:
        1) departure separation (within flow).
        2) intersection separation (between flows).

        :return: Delay dataframe
        """
        delays = pd.DataFrame().reindex_like(self.extended_from_flow_rates)
        delays[:] = 0.

        # Obtain via index array.
        via_idx = self.extended_from_flow_rates.index.get_level_values('via')

        # Delay model.
        for node in self.urban_grid.all_nodes:
            node_flows = self.extended_from_flow_rates.iloc[via_idx == node]
            from_nodes = node_flows.index.get_level_values('from').unique()
            if node in self.urban_grid.od_nodes:
                # Departure separation: time to merge into an airway [s].
                t_x = self.s_h / self.speed
                # Append departure flow rate.
                node_flows = node_flows.append(self.departure_rate_per_node, ignore_index=True)
                from_nodes = [from_nodes[0], 'departure']
            else:
                # Intersection separation: Time to cross an intersection [s].
                # Change sqrt(2) if angle between airways is not 90 degrees.
                t_x = self.s_h * np.sqrt(2) / self.speed
                if len(from_nodes) == 1:
                    # Border node, no intersection. No delay.
                    continue
                elif len(from_nodes) != 2:
                    # Sanity check.
                    raise ValueError(f'Intersection with {len(from_nodes)} directions found!\n', node_flows)
            # Loop through both upstream flows.
            total_q = node_flows.sum()
            total_y = total_q * t_x
            for (i, j) in [(1, 0), (0, 1)]:
                # General delay.
                q_g = node_flows.iloc[i].squeeze()
                q_r = node_flows.iloc[j].squeeze()
                lambda_u = 1 - t_x * q_r
                c_u = 1 / q_r
                y = q_g * t_x
                general_delay = c_u * np.power(1 - lambda_u, 2) / (2 * (1 - y))

                # Stochastic delay.
                stochastic_y = q_g * general_delay
                stochastic_delay = stochastic_y * stochastic_y / (2 * q_g * (1 - stochastic_y))

                # If intersection unstable, set delay very large.
                general_delay[total_y >= 1] = 1E5

                # Add to delays df.
                delays.loc[(from_nodes[i], node)] += general_delay + stochastic_delay

        return delays

    def wr_model(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        The delay model with conflict resolution.
        Calculates the mean velocity, instantaneous number of aircraft,
        and the mean flight time of the flows through an urban grid.
        If the delay model predicts an unstable system, the values are set to zero.

        :return: (Mean velocity, No. Inst. A/C, Mean flight time, Flow rate, Mean delay)
        """
        print('Calculating analytical WR model...')
        max_ni_per_section = self.urban_grid.grid_height / self.s_h

        section_index = self.delays.index.get_level_values('from') != 'departure'
        section_delays = self.delays.loc[section_index].copy()
        departure_delays = self.delays.loc[~section_index].copy()

        # Section delays.
        nr_flight_time_per_section = self.urban_grid.grid_height / self.speed
        wr_flight_time_per_section = section_delays + nr_flight_time_per_section
        wr_v_per_section = self.urban_grid.grid_height / wr_flight_time_per_section
        wr_ni_per_section = self.extended_from_flow_rates.loc[section_index] * wr_flight_time_per_section

        # Departure delays.
        wr_ni_per_departure = self.extended_from_flow_rates.loc[~section_index] * departure_delays

        # Sum sections and departure inst. no. of aircraft.
        wr_ni = wr_ni_per_section.sum() + wr_ni_per_departure.sum()
        wr_mean_v = (wr_v_per_section * wr_ni_per_section).sum() / wr_ni  # Departing aircraft have mean_v = 0.
        wr_mean_flight_time = self.mean_route_length / wr_mean_v

        # Filter unstable values and set to zero.
        unstable_filter = (wr_ni_per_section > max_ni_per_section).any()
        wr_mean_v[unstable_filter] = np.nan
        wr_ni[unstable_filter] = np.nan
        wr_mean_flight_time[unstable_filter] = np.nan
        wr_flow_rate = wr_mean_v * wr_ni

        wr_delay = wr_mean_flight_time - self.mean_flight_time_nr

        return wr_mean_v, wr_ni, wr_mean_flight_time, wr_flow_rate, wr_delay

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
        vehicle_delay_per_second = self.delays * self.extended_from_flow_rates
        vd_via_idx = vehicle_delay_per_second.index.get_level_values('via')
        ff_via_idx = self.extended_from_flow_rates.index.get_level_values('via')
        additional_conflicts_per_second = pd.Series(0, index=self.n_inst)
        for node in self.urban_grid.all_nodes:
            # Loop over all nodes.
            node_delays_per_second = vehicle_delay_per_second[vd_via_idx == node].copy()
            node_from_flows = self.extended_from_flow_rates[ff_via_idx == node].copy()
            node_from_flows_reversed = node_from_flows.copy().iloc[-1::-1]
            if len(node_delays_per_second) == 1:
                # Border node with no delay, skip.
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


if __name__ == '__main__':
    S_H = 50.  # m
    S_V = 25.  # ft
    T_L = 20.  # s
    SPEED = 10.
    DURATION = (900., 2700., 900.)

    pkl_file = Path(r'../scenario/URBAN/Data/final_grid_urban_grid.pkl')
    with open(pkl_file, 'rb') as f:
        grid = pkl.load(f)

    plot_flow_rates(grid.flow_df)

    ana_model = NetworkModel(grid, max_value=100, accuracy=50,
                             duration=DURATION, speed=SPEED, s_h=S_H, s_v=S_V, t_l=T_L)

    ana_model.plot_mfd()
