"""
The analytical model class for a single orthogonal intersection.

Created by Michiel Aarts, April 2021
"""
from utils.urban.analytical import AnalyticalModel
import numpy as np
import pandas as pd
from typing import Tuple
from bluesky.tools.aero import ft


class IntersectionModel(AnalyticalModel):
    def __init__(
            self,
            flow_ratio: Tuple[float, float, float, float], max_value: float,
            accuracy: int, duration: Tuple[float, float, float],
            speed: float, s_h: float, s_v: float, t_l: float, turn_model: bool = False
    ):
        """
        Class for the analytical conflict count and delay models for a single intersection.

        :param flow_ratio: Green ratio between four flows -
         Tuple[eastbound, northbound, east-north, north-east]
        :param max_value: maximum inst. no. of aircraft to model for
        :param accuracy: number of evaluations modelled between 0 and max_value
        :param duration: simulation duration (build-up, logging, cool-down) [s, s, s]
        :param speed: autopilot speed of aircraft [m/s]
        :param s_h: horizontal separation distance [m]
        :param s_v: vertical separation distance [ft]
        :param t_l: look-ahead time [s]
        :param turn_model: Whether or not to include the turn model [bool]
        """
        super().__init__()

        # Parse inputs.
        self.flow_ratio = flow_ratio
        self.max_value = max_value
        self.accuracy = accuracy
        self.duration = duration
        self.speed = speed
        self.s_h = s_h
        self.s_v = s_v * ft  # m
        self.t_l = t_l
        self.turn_model = turn_model

        self.cruise_alt = 0. * ft  # m
        self.departure_alt = 0. * ft  # m
        self.mean_route_length = 2000.  # m
        self.section_length = self.mean_route_length / 2  # m.
        self.mean_flight_time_nr = self.mean_route_length / self.speed

        # Turn variables.
        self.n_total_flow = 0
        self.c_total_turn = 0

        # Initiate arrays.
        self.n_inst = np.linspace(1, self.max_value, self.accuracy)

        # Calculate models.
        self.calculate_models()

    def calculate_models(self):
        assert isinstance(self.n_inst, np.ndarray), 'calculate_models() only works with n_inst as numpy array'

        # Determine flow proportions and rates.
        self.n_total = self.n_inst * self.duration[1] / self.mean_flight_time_nr
        self.flow_rates = self.determine_flow_rates()  # veh / s
        self.departure_rate = self.n_inst / self.mean_flight_time_nr  # veh / s
        self.arrival_rate = self.departure_rate  # By definition, for a stable system
        self.from_flow_rates, self.from_flow_hdg = self.determine_from_flow_rates(self.flow_rates)  # veh / s

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
        Determines flow rates through the UrbanGrid for each n_inst.

        :return: Flow rates dataframe
        """
        flow_rates = pd.DataFrame(index=self.flow_ratio, columns=self.n_inst)
        for ni in self.n_inst:
            passage_rate = ni * self.speed / self.mean_route_length
            flow_rates[ni] = pd.Series(self.flow_ratio, index=self.flow_ratio, dtype=float) * passage_rate
        return flow_rates

    def determine_from_flow_rates(self, flow_df) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Groups flow rates based on from and via nodes, to obtain the 'from flow'-rates and headings.

        :param flow_df: Flow rates dataframe.
        :return: From flow rates dataframe
        """
        sections = pd.MultiIndex.from_frame(
            pd.DataFrame({'from': ['west', 'south', 'middle', 'middle'],
                          'to': ['middle', 'middle', 'east', 'north']}))

        from_flows = pd.DataFrame(index=sections, columns=self.n_inst, dtype=float)
        from_flows.loc['west', 'middle'] = flow_df.iloc[0] + flow_df.iloc[2]
        from_flows.loc['south', 'middle'] = flow_df.iloc[1] + flow_df.iloc[3]
        from_flows.loc['middle', 'east'] = flow_df.iloc[0] + flow_df.iloc[3]
        from_flows.loc['middle', 'north'] = flow_df.iloc[1] + flow_df.iloc[2]

        hdg = pd.Series([90, 0, 90, 0], index=sections, name='hdg', dtype=float)
        return from_flows, hdg

    def nr_model(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        The conflict count model without conflict resolution.
        Calculates the instantaneous number of conflicts and losses of separation,
        based on the flows through the intersection.

        :return: (Inst. no. of conflicts, Inst. no. of LoS, Total no. of conflicts, Total no. of LoS)
        """
        print('Calculating analytical NR model...')
        nr_ni_per_section = self.from_flow_rates.copy() * (self.section_length / self.speed)

        # Self interaction (LoS only).
        # Self interaction not present anymore with departure separation.
        # nr_li_si_per_section = 0. * nr_ni_per_section.copy()
        # for n in range(2, 100):
        #     poisson_prob_per_section = nr_ni_per_section.pow(n) * np.exp(-nr_ni_per_section) / np.math.factorial(n)
        #     nr_li_si_per_section += (poisson_prob_per_section * n * (n - 1) * self.s_h
        #                              / (self.section_length + self.s_h))
        #     if poisson_prob_per_section.max().max() < 1E-5:
        #         # Probability has become insignificant. Break loop.
        #         print(f'NR model: Poisson probability loop broken after P(x={n})')
        #         break
        # nr_li_self_interaction = nr_li_si_per_section.sum()

        # Crossing conflicts and LoS.
        # Define areas.
        los_area = np.pi * np.power(self.s_h, 2)
        conf_vrel = 2 * self.speed * np.sin(np.deg2rad(90) / 2)
        conf_area = 2 * self.s_h * conf_vrel * self.t_l
        isct_area = np.power(2 * self.section_length, 2)

        upstream_nr_ni = nr_ni_per_section.loc[nr_ni_per_section.index.get_level_values('to') == 'middle']
        if len(upstream_nr_ni) == 2:
            # Regular intersection node.
            nr_li_crossing = 4 * upstream_nr_ni.iloc[0] * upstream_nr_ni.iloc[1] * los_area / isct_area
            nr_ci_crossing = 4 * upstream_nr_ni.iloc[0] * upstream_nr_ni.iloc[1] * conf_area / isct_area
        else:
            # Sanity check.
            raise NotImplementedError(f'Intersections with {len(nr_ni_per_section)} segments not implemented.')

        if self.turn_model:
            # Turning traffic.
            if self.flow_ratio[0] + self.flow_ratio[2] != 0:
                # Prevent div / 0 errors.
                p_east = [self.flow_ratio[2] / (self.flow_ratio[0] + self.flow_ratio[2])]
            else:
                p_east = [0]
            if self.flow_ratio[1] + self.flow_ratio[3] != 0:
                # Prevent div / 0 errors.
                p_north = [self.flow_ratio[3] / (self.flow_ratio[1] + self.flow_ratio[3])]
            else:
                p_north = [0]
            p_turn = np.array([p_east, p_north])
            q_total = np.array([self.from_flow_rates.loc['west', 'middle'],
                                self.from_flow_rates.loc['south', 'middle']])
            lambda_d = q_total / self.speed
            x = self.s_h * np.sqrt(2)
            p_dist = 1 - (1 - self.s_h * lambda_d) * np.exp(-lambda_d * (x - self.s_h))
            n_total_flow = q_total * self.duration[1]
            c_total_turn = np.sum(p_turn * p_dist * n_total_flow, axis=0)
            self.n_total_flow = n_total_flow
            self.c_total_turn = c_total_turn
        else:
            c_total_turn = 0

        # Sum crossing conflicts / LoS.
        nr_li = nr_li_crossing  # + nr_li_self_interaction
        # Self interaction with equal speeds must be a LoS. Therefore, ci_self_interaction = departing traffic only.
        nr_ci = nr_ci_crossing
        nr_ctotal = nr_ci * self.duration[1] / self.t_l + c_total_turn
        # Each conflict should result in a LoS.
        nr_lostotal = nr_ctotal

        return nr_ci, nr_li, nr_ctotal, nr_lostotal

    def delay_model(self) -> pd.DataFrame:
        """
        Calculates the delay per vehicle at each node for the provided from_flow dataframe.

        Delay model has two parts:
        1) departure separation (within flow).
        2) intersection separation (between flows).
        Departure separation is performed in the scenario generator.

        :return: Delay dataframe
        """
        delays = pd.DataFrame().reindex_like(self.from_flow_rates)

        # Intersection separation.
        t_parallel = self.s_h / self.speed
        t_x = t_parallel * np.sqrt(2)  # Time to cross an intersection [s].
        approach_flows = self.from_flow_rates.loc[self.from_flow_rates.index.get_level_values('to') == 'middle'].copy()
        from_nodes = approach_flows.index.get_level_values('from')
        for (i, j) in [(1, 0), (0, 1)]:
            q_g = approach_flows.iloc[i].squeeze()
            q_r = approach_flows.iloc[j].squeeze()
            lambda_u = 1 - t_x * q_r
            c_u = 1 / q_r
            x = q_g * t_parallel / lambda_u
            general_delay = c_u * np.power(1 - lambda_u, 2) / (2 * (1 - lambda_u * x))
            stochastic_delay = np.power(x, 2) / (2 * q_g * (1 - x))

            # If intersection unstable, set delay very large.
            unstable_filter = (lambda_u * x >= 1) | (x >= 1)
            if isinstance(general_delay, pd.Series):
                general_delay[unstable_filter] = 1E5
                stochastic_delay[unstable_filter] = 1E5
            else:
                if unstable_filter:
                    general_delay = 1E5
                    stochastic_delay = 1E5

            # Add to delays df.
            delays.loc[(from_nodes[i], 'middle')] = general_delay + stochastic_delay

        delays[delays.isna()] = 0.
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
        isct_flow_rates = self.from_flow_rates.loc[self.from_flow_rates.index.get_level_values('to') == 'middle'].sum()
        wr_delay = (self.delays * self.from_flow_rates.loc[self.delays.index]).sum() / isct_flow_rates
        wr_delay[wr_delay >= 1E5] = np.nan

        max_ni_per_section = self.section_length / self.s_h

        nr_flight_time_per_section = self.section_length / self.speed
        wr_flight_time_per_section = self.delays.copy() + nr_flight_time_per_section
        wr_v_per_section = self.section_length / wr_flight_time_per_section
        wr_ni_per_section = self.from_flow_rates.loc[self.delays.index] * wr_flight_time_per_section
        wr_ni = wr_ni_per_section.sum()
        wr_mean_v = (wr_v_per_section * wr_ni_per_section.loc[wr_v_per_section.index]).sum() / wr_ni_per_section.sum()
        wr_mean_flight_time = self.mean_route_length / wr_mean_v

        # Filter unstable values and set to NaN.
        unstable_filter = (wr_ni_per_section > max_ni_per_section).any()
        wr_mean_v[unstable_filter] = np.nan
        wr_ni[unstable_filter] = np.nan
        wr_mean_flight_time[unstable_filter] = np.nan
        wr_flow_rate = wr_mean_v * wr_ni

        return wr_mean_v, wr_ni, wr_mean_flight_time, wr_flow_rate, wr_delay

    def wr_conflict_model(self) -> np.ndarray:
        """ Based on local flow rates and delay """
        vehicle_delay_per_second = self.delays * self.from_flow_rates
        vd_to_idx = vehicle_delay_per_second.index.get_level_values('to')
        ff_to_idx = self.from_flow_rates.index.get_level_values('to')
        additional_conflicts_per_second = pd.Series(0, index=self.n_inst)
        isct_delays_per_second = vehicle_delay_per_second[vd_to_idx == 'middle'].copy()
        isct_from_flows = self.from_flow_rates[ff_to_idx == 'middle'].copy()
        isct_from_flows_reversed = isct_from_flows.copy().iloc[-1::-1]
        if len(isct_delays_per_second) == 2:
            # Regular intersection.
            additional_conflicts_per_second += (isct_delays_per_second * isct_from_flows_reversed.values).sum()
            additional_conflicts_per_second += (isct_delays_per_second * isct_from_flows.values).sum()
        else:
            # Sanity check.
            raise NotImplementedError(f'Intersections with {len(isct_delays_per_second)} headings not implemented.')

        c_total_wr = self.c_total_nr + additional_conflicts_per_second * self.duration[1]
        c_total_wr[np.isnan(self.n_inst_wr)] = np.nan
        return c_total_wr


if __name__ == '__main__':
    S_H = 50.  # m
    S_V = 25.  # ft
    T_L = 20.  # s
    SPEED = 10.
    DURATION = (900., 2700., 900.)
    FLOW_RATIO = (0.54, 0.36, 0.06, 0.04)

    ana_model = IntersectionModel(FLOW_RATIO, max_value=50, accuracy=50,
                                  duration=DURATION, speed=SPEED, s_h=S_H, s_v=S_V, t_l=T_L, turn_model=False)

    ana_model.plot_mfd()
