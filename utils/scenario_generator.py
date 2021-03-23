"""
Scenario generator file for an orthogonal grid.

Created by Michiel Aarts, March 2021
"""
import time
from plugins.urban import UrbanGrid
import numpy as np
import scipy.stats as stats
import random
from pathlib import Path
from bluesky.tools.aero import nm, kts
import pickle as pkl
from typing import Tuple
from scn_reader import plot_flow_rates


class ScenarioGenerator:
    def __init__(
        self, n_rows: int, n_cols: int, grid_width: float, grid_height: float
    ) -> None:
        """
        Initiates the scenario generator class, with an urban grid of plugins/urban.py.

        :param n_rows: Number of rows (should be a multiple of 4, minus 1)
        :param n_cols: Number of columns (should be a multiple of 4, minus 1)
        :param grid_width: Distance between two longitudinal nodes [km]
        :param grid_height: Distance between two lateral nodes [km]

        :return: None
        """

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.grid_width = grid_width
        self.grid_height = grid_height

        # Initiate urban_grid
        self.urban_grid = UrbanGrid(self.n_rows, self.n_cols, self.grid_width, self.grid_height)
        self.grid_area = self.n_rows * self.grid_width + self.n_cols * self.grid_height  # km2
        self.avg_route_length = self.urban_grid.avg_route_length * 1000  # m

    def create_scenario(self, n_inst: np.ndarray, repetitions: int,
                        speed: np.ndarray, duration: Tuple[float, float, float],
                        h_sep: np.ndarray, v_sep: np.ndarray, t_lookahead: np.ndarray, ac_type: str = 'M600'
                        ) -> list:
        """
        Creates a scenario for each value of n_inst, or for a single n_inst.
        All other inputs are either an array of len(n_inst) or a float.

        :param n_inst: Desired inst. no. of aircraft [km^-2] - optionally an array, determines size
        :param repetitions: Number of repetitions of n_inst [-]
        :param speed: Undisturbed speed of aircraft [m/s] - optionally an array
        :param duration: Scenario duration [s] - Tuple[build-up, experiment, cool-down]
        :param h_sep: horizontal separation [m] - optionally an array
        :param v_sep: vertical separation [ft] - optionally an array
        :param t_lookahead: look-ahead time [s] - optionally an array
        :param ac_type: Type of aircraft

        :return: list of scenario dicts
        """
        # Parse inputs.
        if isinstance(n_inst, float):
            n_inst = np.array([n_inst])
        if isinstance(speed, float):
            speed = np.array([speed] * len(n_inst))
        if isinstance(h_sep, float):
            h_sep = np.array([h_sep] * len(n_inst))
        if isinstance(v_sep, float):
            v_sep = np.array([v_sep] * len(n_inst))
        if isinstance(t_lookahead, float):
            t_lookahead = np.array([t_lookahead] * len(n_inst))

        if len(speed) != len(n_inst):
            raise ValueError('Length of speed array does not match density array')
        if len(duration) != 3:
            raise ValueError('Duration must be a tuple of (build-up, experiment, cool-down)')
        if len(h_sep) != len(n_inst):
            raise ValueError('Length of horizontal separation array does not match density array')
        if len(v_sep) != len(n_inst):
            raise ValueError('Length of vertical separation array does not match density array')
        if len(t_lookahead) != len(n_inst):
            raise ValueError('Length of look-ahead time array does not match density array')

        T = sum(duration)

        # Loop through densities.
        all_scen = []
        for (n_i, V, s_h, s_v, t_l) in zip(n_inst, speed, h_sep, v_sep, t_lookahead):
            for rep in range(repetitions):
                print(f'Calculating scenario for N_inst={n_i}, Rep={rep}...')

                # Determine departure times.
                avg_route_duration = self.avg_route_length / V
                spawn_rate = n_i / avg_route_duration
                spawn_interval = 1 / spawn_rate
                n_total = round(T * spawn_rate)
                # Exponential distribution.
                # departure_times = np.cumsum(stats.expon(scale=spawn_interval).rvs(n_total))
                # Exponential is more realistic, but makes comparison with analytical model less clear.
                # Uniform interval.
                departure_times = np.array(range(n_total)) * spawn_interval
                # Add noise.
                departure_times = departure_times + np.random.uniform(0, spawn_interval * 0.99, size=n_total)

                # Calculate origin-destination combinations and routes.
                od_nodes = self.urban_grid.od_nodes
                prev_origin = None
                prev_destination = None
                all_ac = []
                for ac_id in range(len(departure_times)):
                    origin = random.choice(od_nodes)
                    while origin == prev_origin:
                        origin = random.choice(od_nodes)

                    destination = random.choice(od_nodes)
                    while destination == origin or destination == prev_destination:
                        destination = random.choice(od_nodes)

                    path, path_length, _, _ = self.urban_grid.calculate_shortest_path(origin, destination)

                    ac_dict = {'id': ac_id, 'departure_time': departure_times[ac_id],
                               'origin': origin, 'destination': destination,
                               'path': path, 'path_length': path_length,
                               'ac_type': ac_type}
                    all_ac.append(ac_dict)
                    prev_origin = origin
                    prev_destination = destination

                scen_dict = {'n_inst': n_i, 'rep': rep, 'n_total': n_total,
                             'speed': V, 'duration': duration,
                             's_h': s_h, 's_v': s_v, 't_l': t_l,
                             'scenario': all_ac, 'grid_nodes': self.urban_grid.nodes}
                all_scen.append(scen_dict)
        return all_scen

    @staticmethod
    def tim2txt(t):
        """Convert time to timestring: HH:MM:SS.hh"""
        return time.strftime("%H:%M:%S.", time.gmtime(t)) + f'{(t % 1) * 100:.0f}'.zfill(2)

    def write_scenario(
        self, all_scen: list, asas: str = 'on',
        prefix: str = 'test_scen',
        scn_path: Path = Path('../scenario/URBAN/'),
        pkl_path: Path = Path('../scenario/URBAN/Data/')
    ) -> None:
        """
        Writes one or more scenarios to *.scn files.

        :param all_scen: list with scenarios generated by create_scenario()
        :param asas: toggle for statebased asas (default: statebased)
        :param prefix: Prefix to scenario names
        :param scn_path: path to scenario folder
        :param pkl_path: path to pickle folder

        :return: None
        """

        if not scn_path.is_dir():
            scn_path.mkdir(parents=True, exist_ok=True)
        if not pkl_path.is_dir():
            pkl_path.mkdir(parents=True, exist_ok=True)

        all_scn_files = []
        for scn in all_scen:
            n_inst = scn['n_inst']
            rep = scn['rep']
            spd = scn['speed']
            tas = spd / kts
            duration = scn['duration']

            # Let aircraft climb slightly to cruise altitude, to prevent LoS at creation.
            cruise_alt = 50.  # ft
            departure_alt = cruise_alt - scn['s_v'] * 1.5  # ft

            # Save data to .pkl file.
            pkl_file = f'{prefix}_N{n_inst:.0f}_R{rep:.0f}.pkl'
            with open(pkl_path / pkl_file, 'wb') as f:
                pkl.dump(scn, f, protocol=pkl.HIGHEST_PROTOCOL)
                print(f'Written {pkl_path / pkl_file}')

            # Save scenario to .scn file.
            scn_file = f'{prefix}_N{n_inst:.0f}_R{rep:.0f}.scn'
            all_scn_files.append(scn_file)
            with open(scn_path / scn_file, 'w') as f:
                f.write('# ########################################### #\n')
                f.write(f'# Scenario: {scn_file}\n')
                f.write(f'# Instantaneous number of aircraft: {n_inst:.0f}\n')
                f.write(f'# Repetition no.: {rep:.0f}\n')
                f.write(f'# Speed: {spd:.1f}m/s\n')
                f.write(f'# Duration (build-up, experiment, cool-down): {duration}s\n')
                f.write(f'# Horizontal separation: {scn["s_h"]:.1f}m\n')
                f.write(f'# Vertical separation: {scn["s_v"]:.1f}ft\n')
                f.write(f'# Look-ahead time: {scn["t_l"]:.1f}s\n')
                f.write(f'# Mean route length: {self.avg_route_length:.1f}m\n')
                f.write('# ########################################### #\n\n')

                # Load urban grid
                f.writelines(self.urban_grid.city_grid_scenario(self.tim2txt(0)))

                # Pan screen to center node, zoom and hold simulation
                f.write('# Pan screen to center node\n')
                f.write(f'{self.tim2txt(0)}>PAN {self.urban_grid.center_node}\n')
                f.write(f'{self.tim2txt(0)}>ZOOM 20\n\n')

                # Set ASAS and RESO
                f.write('# Set ASAS variables\n')
                f.write(f'{self.tim2txt(0)}>ASAS {asas}\n')
                f.write(f'{self.tim2txt(0)}>ZONER {scn["s_h"] / nm}\n')
                f.write(f'{self.tim2txt(0)}>ZONEDH {scn["s_v"] / 2}\n')
                f.write(f'{self.tim2txt(0)}>DTLOOK {scn["t_l"]}\n\n')

                # Load experiment area and logger.
                f.write('# Initiate area and logger\n')
                f.write(f'{self.tim2txt(0)}>LOGPREFIX {scn_file[:-4]}\n')
                bbox = [self.urban_grid.min_lat - 1, self.urban_grid.min_lon - 1,
                        self.urban_grid.max_lat + 1, self.urban_grid.max_lon + 1]
                f.write(f'{self.tim2txt(0)}>AREA {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n\n')

                # Create aircraft.
                f.write('# Create aircraft.\n')
                max_id = scn['scenario'][-1]['id']
                len_ac_id = len(str(max_id))
                for ac in scn['scenario']:
                    time_string = self.tim2txt(ac['departure_time'])
                    callsign = f'UAV{str(ac["id"]).zfill(len_ac_id)}'

                    # Determine departure heading.
                    origin = ac['origin']
                    first_node = ac['path'][1]
                    hdg = self.urban_grid.edges[origin][first_node]['hdg']

                    # Write to .scn file.
                    f.write(f'# Creating aircraft no. {ac["id"]}\n')
                    f.write(f'{time_string}>CRE {callsign} {ac["ac_type"]} '
                            f'{origin} {hdg} {departure_alt} {tas}\n')
                    f.write(f'{time_string}>ORIG {callsign} {origin}\n')
                    f.write(f'{time_string}>DEST {callsign} {ac["destination"]}\n')
                    for wpt in ac["path"][1:-1]:
                        f.write(f'{time_string}>ADDWPT {callsign} {wpt} {cruise_alt} {tas}\n')
                    f.write(f'{time_string}>LNAV {callsign} ON\n')
                    f.write(f'{time_string}>VNAV {callsign} ON\n')
                    f.write('\n')
                print(f'Written {scn_path / scn_file}')

        if len(all_scen) > 1:
            # Write batch file
            filename_nr = f'batch_{prefix}_NR.scn'
            filename_wr = f'batch_{prefix}_WR.scn'
            safety_factor = 1.5

            for scn_file in [filename_nr, filename_wr]:
                with open(scn_path / scn_file, 'w') as f:
                    f.write('# ########################################### #\n')
                    f.write(f'# Batch file {scn_file} for files:\n')
                    for fname in all_scn_files:
                        f.write(f'# File: {fname}\n')
                    f.write(f'# Instantaneous number of aircraft: '
                            f'{", ".join(str(scn["n_inst"]).format(".0f") for scn in all_scen)}\n')
                    f.write(f'# Number of repetitions: {all_scen[0]["rep"]:.0f}\n')
                    f.write(f'# Speed: '
                            f'{", ".join(str(scn["speed"]).format(".1f") for scn in all_scen)} m/s\n')
                    f.write(f'# Duration: '
                            f'{", ".join(str(scn["duration"]).format(".0f") for scn in all_scen)} s\n')
                    f.write(f'# Horizontal separation: '
                            f'{", ".join(str(scn["s_h"]).format(".1f") for scn in all_scen)} m\n')
                    f.write(f'# Vertical separation: '
                            f'{", ".join(str(scn["s_v"]).format(".1f") for scn in all_scen)} ft\n')
                    f.write(f'# Look-ahead time: '
                            f'{", ".join(str(scn["t_l"]).format(".1f") for scn in all_scen)} s\n')
                    f.write(f'# Mean route length: {self.avg_route_length:.1f}m\n')
                    f.write('# ########################################### #\n\n')

                    for i in range(len(all_scen)):
                        t = sum(all_scen[i]['duration']) * safety_factor
                        f.write(f'{self.tim2txt(i * t)}>PCALL URBAN/{all_scn_files[i]} REL\n')
                        if 'WR' in scn_file:
                            f.write(f'{self.tim2txt(i * t)}>RESO SPEEDBASED\n')
                        f.write(f'{self.tim2txt(i * t)}>OP\n')
                        f.write(f'{self.tim2txt(i * t)}>FF {t}\n')
                        f.write(f'{self.tim2txt((i + 1) * t)}>CLOSELOG\n\n')
                    f.write(f'{self.tim2txt((i + 1) * t)}>HOLD\n\n')

                print(f'Written {scn_path / scn_file}')

    def save_urban_grid(self, prefix: str = '', pkl_path: Path = Path('../scenario/URBAN/Data/')) -> None:
        """ Save urban grid to .pkl file. """
        # Ensure flow df is evaluated.
        # Note: this may take >5 mins.
        _ = self.urban_grid.flow_df

        # Save to pickle.
        pkl_file = f'{prefix}_urban_grid.pkl'
        with open(pkl_path / pkl_file, 'wb') as f:
            pkl.dump(self.urban_grid, f, protocol=pkl.HIGHEST_PROTOCOL)
            print(f'Written urban grid to {pkl_path / pkl_file}')


if __name__ == '__main__':
    N_INST = np.array([10, 50, 100, 250])
    REPETITIONS = 2
    SPEED = 10.
    BUILD_UP_DURATION = 900.
    EXPERIMENT_DURATION = 2700.
    COOL_DOWN_DURATION = 900.
    DURATION = (BUILD_UP_DURATION, EXPERIMENT_DURATION, COOL_DOWN_DURATION)
    PREFIX = 'test1'

    N_ROWS = 19
    N_COLS = N_ROWS
    HORIZONTAL_SEPARATION_KM = 0.2  # km
    VERTICAL_SEPARATION_KM = 0.2  # km
    S_H = 50.  # m
    S_V = 25.  # ft
    T_L = 20.  # s

    scen_gen = ScenarioGenerator(N_ROWS, N_COLS, HORIZONTAL_SEPARATION_KM, VERTICAL_SEPARATION_KM)
    all_scenarios = scen_gen.create_scenario(N_INST, REPETITIONS, SPEED, DURATION, S_H, S_V, T_L)
    scen_gen.write_scenario(all_scenarios, prefix=PREFIX)
    scen_gen.save_urban_grid(prefix=PREFIX)

    # Plot flow rates of first scenario for validation.
    print('Creating flow rates plot for first scenario...')
    plot_flow_rates(scen_gen.urban_grid.flow_df)
