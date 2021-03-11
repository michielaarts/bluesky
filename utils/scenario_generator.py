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
        """

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.grid_width = grid_width
        self.grid_height = grid_height

        # Initiate urban_grid
        self.urban_grid = UrbanGrid(self.n_rows, self.n_cols, self.grid_width, self.grid_height)
        self.grid_area = self.n_rows * self.grid_width + self.n_cols * self.grid_height  # km2
        self.avg_route_length = self.urban_grid.avg_route_length * 1000  # m

    def create_scenario(self, n_inst: np.ndarray, speed: np.ndarray, duration: np.ndarray,
                        h_sep: np.ndarray, v_sep: np.ndarray, t_lookahead: np.ndarray, ac_type: str = 'M600'
                        ) -> list:
        """
        Creates a scenario for each value of n_inst, or for a single n_inst.
        All other inputs are either an array of len(n_inst) or a float.

        :param n_inst: Desired inst. no. of aircraft [km^-2] - optionally an array, determines size
        :param speed: Undisturbed speed of aircraft [m/s] - optionally an array
        :param duration: Scenario duration [s] - optionally an array
        :param h_sep: horizontal separation [m] - optionally an array
        :param v_sep: vertical separation [ft] - optionally an array
        :param t_lookahead: look-ahead time [s] - optionally an array
        :param ac_type: Type of aircraft
        """
        # Parse inputs.
        if isinstance(n_inst, float):
            n_inst = np.array([n_inst])
        if isinstance(speed, float):
            speed = np.array([speed] * len(n_inst))
        if isinstance(duration, float):
            duration = np.array([duration] * len(n_inst))
        if isinstance(h_sep, float):
            h_sep = np.array([h_sep] * len(n_inst))
        if isinstance(v_sep, float):
            v_sep = np.array([v_sep] * len(n_inst))
        if isinstance(t_lookahead, float):
            t_lookahead = np.array([t_lookahead] * len(n_inst))

        if len(speed) != len(n_inst):
            raise ValueError('Length of speed array does not match density array')
        if len(duration) != len(n_inst):
            raise ValueError('Length of duration array does not match density array')
        if len(h_sep) != len(n_inst):
            raise ValueError('Length of horizontal separation array does not match density array')
        if len(v_sep) != len(n_inst):
            raise ValueError('Length of vertical separation array does not match density array')
        if len(t_lookahead) != len(n_inst):
            raise ValueError('Length of look-ahead time array does not match density array')

        # Loop through densities.
        all_scen = []
        for (n_i, V, T, s_h, s_v, t_l) in zip(n_inst, speed, duration, h_sep, v_sep, t_lookahead):
            print(n_i, V, T, s_h, s_v, t_l)

            # Determine departure times.
            avg_route_duration = self.avg_route_length / V
            spawn_rate = n_i / avg_route_duration
            n_total = round(T * spawn_rate)
            departure_times = np.cumsum(stats.expon(scale=1 / spawn_rate).rvs(n_total))

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

            scen_dict = {'n_inst': n_i, 'n_total': n_total,
                         'speed': V, 'duration': T,
                         's_h': s_h, 's_v': s_v, 't_l': t_l,
                         'scenario': all_ac}
            all_scen.append(scen_dict)
        return all_scen

    @staticmethod
    def tim2txt(t):
        """Convert time to timestring: HH:MM:SS.hh"""
        return time.strftime("%H:%M:%S.", time.gmtime(t)) + f'{(t % 1) * 100:.0f}'

    def write_scenario(
        self, all_scen: list, asas: str = 'on', reso: str = 'off',
        prefix: str = 'test_scen', filepath: Path = Path('../scenario/URBAN/')
    ) -> None:
        """
        Writes one or more scenarios to *.scn files.

        :param all_scen: list with scenarios generated by create_scenario()
        :param asas: toggle for statebased asas (default: statebased)
        :param reso: select resolution method (default: off)
        :param prefix: Prefix to scenario names
        :param filepath: path to scenario folder
        :return: None
        """

        if not filepath.is_dir():
            filepath.mkdir(parents=True, exist_ok=True)

        for scen in all_scen:
            n_inst = scen['n_inst']
            spd = scen['speed']
            tas = spd / kts
            duration = scen['duration']

            # Let aircraft climb slightly to cruise altitude, to prevent LoS at creation.
            cruise_alt = 50.  # ft
            departure_alt = cruise_alt - scen['s_v'] * 1.1  # ft

            filename = f'{prefix}_N{n_inst:.0f}_V{spd:.0f}_T{duration:.0f}.scn'
            with open(filepath / filename, 'w') as f:
                f.write('# ########################################### #\n')
                f.write(f'# Scenario: {filename}\n')
                f.write(f'# Instantaneous number of aircraft: {n_inst:.0f}\n')
                f.write(f'# Speed: {spd:.1f}m/s\n')
                f.write(f'# Duration: {duration:.1f}s\n')
                f.write(f'# Horizontal separation: {scen["s_h"]:.1f}m\n')
                f.write(f'# Vertical separation: {scen["s_v"]:.1f}ft\n')
                f.write(f'# Look-ahead time: {scen["t_l"]:.1f}s\n')
                f.write(f'# Mean route length: {self.avg_route_length:.1f}m\n')
                f.write('# ########################################### #\n')

                # Load urban plugin
                f.write(f'{self.tim2txt(0)}>PLUGIN URBAN\n\n')

                # Set ASAS and RESO
                f.write(f'{self.tim2txt(0)}>ASAS {asas}\n')
                f.write(f'{self.tim2txt(0)}>RESO {reso}\n')
                f.write(f'{self.tim2txt(0)}>ZONER {scen["s_h"] / nm}\n')
                f.write(f'{self.tim2txt(0)}>ZONEDH {scen["s_v"] / 2}\n')
                f.write(f'{self.tim2txt(0)}>DTLOOK {scen["t_l"]}\n\n')

                # Pan screen to center node, zoom and hold simulation
                f.write(f'{self.tim2txt(0)}>PAN {self.urban_grid.center_node}\n')
                f.write(f'{self.tim2txt(0)}>ZOOM 20\n')
                f.write(f'{self.tim2txt(0)}>HOLD\n\n')

                # Load experiment area and logger.
                bbox = [self.urban_grid.min_lat - 1, self.urban_grid.min_lon - 1,
                        self.urban_grid.max_lat + 1, self.urban_grid.max_lon + 1]
                f.write(f'{self.tim2txt(0)}>AREA {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n\n')

                max_id = scen['scenario'][-1]['id']
                len_ac_id = len(str(max_id))
                for ac in scen['scenario']:
                    time_string = self.tim2txt(ac['departure_time'])
                    callsign = f'UAV{str(ac["id"]).zfill(len_ac_id)}'

                    # Determine departure heading.
                    origin = ac['origin']
                    first_node = ac['path'][1]
                    hdg = self.urban_grid.edges[origin][first_node]['hdg']

                    # Write to .scn file.
                    f.write(f'\n# Creating aircraft no. {ac["id"]}\n')
                    f.write(f'{time_string}>CRE {callsign} {ac["ac_type"]} '
                            f'{origin} {hdg} {departure_alt} {tas}\n')
                    f.write(f'{time_string}>ORIG {callsign} {origin}\n')
                    f.write(f'{time_string}>DEST {callsign} {ac["destination"]}\n')
                    for wpt in ac["path"][1:-1]:
                        f.write(f'{time_string}>ADDWPT {callsign} {wpt} {cruise_alt} {tas}\n')
                    f.write(f'{time_string}>LNAV {callsign} ON\n')
                    f.write(f'{time_string}>VNAV {callsign} ON\n')
                print(f'Written {filepath / filename}')


if __name__ == '__main__':
    N_INST = np.array([10., 100., 200.])
    SPEED = 10.
    DURATION = 1800.

    N_ROWS = 19
    N_COLS = N_ROWS
    HORIZONTAL_SEPARATION_KM = 0.2  # km
    VERTICAL_SEPARATION_KM = 0.2  # km
    S_H = 50.  # m
    S_V = 25.  # ft
    T_L = 10.  # s

    scen_gen = ScenarioGenerator(N_ROWS, N_COLS, HORIZONTAL_SEPARATION_KM, VERTICAL_SEPARATION_KM)
    all_scenarios = scen_gen.create_scenario(N_INST, SPEED, DURATION, S_H, S_V, T_L)
    scen_gen.write_scenario(all_scenarios, reso='off')
