"""
Scenario generator file for a single orthogonal intersection.
Intersection has 4 flows: East- and northbound, east-north turn, and north-east turn.

Created by Michiel Aarts, April 2021
"""
import numpy as np
import scipy.stats as stats
from pathlib import Path
from bluesky.tools.aero import nm, kts
import pickle as pkl
from typing import Tuple
from plugins.urban import UrbanGrid
from intersection_model import IntersectionModel

# Let aircraft climb slightly to cruise altitude, to prevent LoS at creation.
DEPARTURE_ALTITUDE = 0.  # ft
CRUISE_ALTITUDE = 0.  # ft
APPROACH_DISTANCE = 1000.  # m
SATURATION = np.linspace(0.05, 0.95, 10)  # [0.15, 0.35, 0.55, 0.75, 0.95]

# Use exponential distribution for departure separation. If False: uniform + noise.
EXPONENTIAL = False


class ScenarioGenerator:
    def __init__(self):
        # Initiate lat & lon of route points.
        self.west = (UrbanGrid.m_to_lat(APPROACH_DISTANCE), 0.)
        self.south = (0., UrbanGrid.m_to_lon(APPROACH_DISTANCE))
        self.middle = (self.west[0], self.south[1])
        self.east = (self.west[0], self.south[1] * 2)
        self.north = (self.west[0] * 2, self.south[1])

        # Initiate routes.
        self.routes = [
            (self.west, self.middle, self.east),
            (self.south, self.middle, self.north),
            (self.west, self.middle, self.north),
            (self.south, self.middle, self.east)
        ]
        self.hdg = [90., 0., 90., 0.]

        self.model = None
        self.flow_ratio = None

    def create_scenario(self, flow_ratio: Tuple[float, float, float, float], repetitions: int,
                        speed: float, duration: Tuple[float, float, float],
                        s_h: float, s_v: float, t_lookahead: float,
                        t_reaction: float, ac_type: str = 'M600'
                        ) -> list:
        """
        Creates a scenario for each value of n_inst, or for a single n_inst.
        All other inputs are either an array of len(n_inst) or a float.

        :param flow_ratio: Desired green ratio between four flows -
         Tuple[eastbound, northbound, east-north, north-east]
        :param repetitions: Number of repetitions of n_inst [-]
        :param speed: Undisturbed speed of aircraft [m/s]
        :param duration: Scenario duration [s] - Tuple[build-up, experiment, cool-down]
        :param s_h: horizontal separation [m]
        :param s_v: vertical separation [ft]
        :param t_lookahead: look-ahead time [s]
        :param t_reaction: reaction time [s]
        :param ac_type: Type of aircraft

        :return: list of scenario dicts
        """
        # Sanity check.
        if sum(flow_ratio) != 1:
            raise ValueError('Sum of flow_ratio should be 1')

        self.flow_ratio = flow_ratio

        # Extract analytical model.
        self.model = IntersectionModel(flow_ratio=self.flow_ratio, max_value=100, accuracy=100,
                                       duration=duration, speed=speed, s_h=s_h, s_v=s_v, t_l=t_lookahead)

        max_ni_nr = max(self.model.n_inst[~self.model.n_inst_wr.isna()])
        mean_flight_time = 2 * APPROACH_DISTANCE / speed
        max_intersection_flow_rate = max_ni_nr / mean_flight_time

        # Prelim. calculations.
        departure_sep = s_h / speed * 1.01

        # Loop through saturations.
        all_scen = []
        for sat in SATURATION:
            # For turn ratio simulations.
            # flow_ratio = (self.flow_ratio[0] * (1 - sat), self.flow_ratio[1] * (1 - sat),
            #               self.flow_ratio[0] * sat, self.flow_ratio[1] * sat)
            # sat = 0.7

            for rep in range(repetitions):
                start_id = 0
                ac_id = 0
                n_total = 0
                all_ac = []
                print(f'Calculating scenario for saturation={sat}, Rep={rep}...')
                for i in range(len(flow_ratio)):
                    if flow_ratio[i] == 0:
                        # No aircraft in this flow.
                        continue

                    # Determine departure times.
                    flow_rate = flow_ratio[i] * sat * max_intersection_flow_rate
                    spawn_interval = 1 / flow_rate
                    n_total_flow = round(sum(duration) * flow_rate)
                    n_total += n_total_flow
                    if EXPONENTIAL:
                        # Exponential distribution.
                        departure_times = np.cumsum(stats.expon(scale=spawn_interval).rvs(n_total_flow))
                    else:
                        # Uniform distribution.
                        departure_times = np.array(range(n_total_flow)) * spawn_interval
                        # Add noise.
                        departure_times = departure_times + np.random.uniform(0, spawn_interval * 0.99,
                                                                              size=n_total_flow)

                    # Create aircraft.
                    for ac_id in range(len(departure_times)):
                        ac_dict = {'id': ac_id + start_id, 'departure_time': departure_times[ac_id],
                                   'origin': self.routes[i][0], 'destination': self.routes[i][-1],
                                   'path': self.routes[i], 'path_length': 2 * APPROACH_DISTANCE,
                                   'hdg': self.hdg[i], 'ac_type': ac_type}
                        all_ac.append(ac_dict)
                    start_id += ac_id + 1

                # Sanity check on all id's.
                all_id = np.array([ac['id'] for ac in all_ac])
                _, id_counts = np.unique(all_id, return_counts=True)
                if np.any(id_counts > 1):
                    raise RuntimeError('Something went wrong in determining ACIDs')

                # Sort all ac on departure time.
                departure_sort = np.argsort([ac['departure_time'] for ac in all_ac])
                sorted_ac = np.array(all_ac)[departure_sort]

                # Separate aircraft from same origin prior to departure.
                for origin in [self.west, self.south]:
                    prior_dep_time = -1E9
                    for j in range(len(sorted_ac)):
                        if sorted_ac[j]['origin'] == origin:
                            if sorted_ac[j]['departure_time'] - prior_dep_time < departure_sep:
                                # Increase separation if too close to prior aircraft.
                                sorted_ac[j]['departure_time'] = prior_dep_time + departure_sep
                            prior_dep_time = sorted_ac[j]['departure_time']

                scen_dict = {'flow_ratio': flow_ratio, 'sat': sat, 'rep': rep,
                             'n_total': n_total, 'speed': speed, 'duration': duration,
                             's_h': s_h, 's_v': s_v, 't_l': t_lookahead, 't_r': t_reaction,
                             'scenario': sorted_ac, 'type': 'intersection'}
                all_scen.append(scen_dict)
        return all_scen

    @staticmethod
    def tim2txt(t):
        """ Convert time to timestring: HH:MM:SS.hh """
        hours = int(t / 3600.)
        minutes = int((t % 3600) / 60.)
        full_seconds = int(t % 60)
        hundredths = int((t % 1) * 100)
        return f'{hours:02d}:{minutes:02d}:{full_seconds:02d}.{hundredths:02d}'

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
            sat = scn['sat']
            rep = scn['rep']
            spd = scn['speed']
            tas = spd / kts
            duration = scn['duration']

            # Filename.
            flow_ratio_string = ''.join(str(round(flow * 100)).zfill(2) for flow in scn['flow_ratio'])
            filename = f'{prefix}_{flow_ratio_string}_S{sat * 100:.0f}_R{rep:.0f}'

            # Save data to .pkl file.
            pkl_file = f'{filename}.pkl'
            with open(pkl_path / pkl_file, 'wb') as f:
                pkl.dump(scn, f, protocol=pkl.HIGHEST_PROTOCOL)
                print(f'Written {pkl_path / pkl_file}')

            # Save scenario to .scn file.
            scn_file = f'{filename}.scn'
            all_scn_files.append(scn_file)
            with open(scn_path / scn_file, 'w') as f:
                f.write('# ########################################### #\n')
                f.write(f'# Scenario: {scn_file}\n')
                f.write(f'# Flow ratio: {scn["flow_ratio"]}\n')
                f.write(f'# Saturation: {sat:.0f}\n')
                f.write(f'# Repetition no.: {rep:.0f}\n')
                f.write(f'# Speed: {spd:.1f}m/s\n')
                f.write(f'# Duration (build-up, experiment, cool-down): {duration}s\n')
                f.write(f'# Horizontal separation: {scn["s_h"]:.1f}m\n')
                f.write(f'# Vertical separation: {scn["s_v"]:.1f}ft\n')
                f.write(f'# Look-ahead time: {scn["t_l"]:.1f}s\n')
                f.write(f'# Mean route length: {2 * APPROACH_DISTANCE:.1f}m\n')
                f.write(f'# NOTE: this scenario requires plugins: SPEEDBASED, URBAN_AREA\n')
                f.write('# ########################################### #\n\n')

                # Load middle node
                f.write('# Load middle node\n')
                f.write(f'{self.tim2txt(0)}>DEFWPT ISCT {self.middle[0]} {self.middle[1]} \n\n')

                # Pan screen to center node, zoom and hold simulation
                f.write('# Pan screen to center node\n')
                f.write(f'{self.tim2txt(0)}>PAN ISCT\n')
                f.write(f'{self.tim2txt(0)}>SWRAD SYM\n')
                f.write(f'{self.tim2txt(0)}>ZOOM 75\n\n')

                # Set ASAS and RESO
                f.write('# Set ASAS variables\n')
                f.write(f'{self.tim2txt(0)}>ASAS {asas}\n')
                f.write(f'{self.tim2txt(0)}>ZONER {scn["s_h"] / nm}\n')
                f.write(f'{self.tim2txt(0)}>ZONEDH {scn["s_v"] / 2}\n')
                f.write(f'{self.tim2txt(0)}>DTLOOK {scn["t_l"]}\n')
                f.write(f'{self.tim2txt(0)}>REACTTIME {scn["t_r"]}\n\n')

                # Load experiment area and logger.
                f.write('# Initiate area and logger\n')
                f.write(f'{self.tim2txt(0)}>LOGPREFIX {scn_file[:-4]}\n')
                bbox = [self.west[0] - 1, self.south[1] - 1,
                        self.east[0] + 1, self.north[1] + 1]
                f.write(f'{self.tim2txt(0)}>AREA {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n\n')

                # Set dt of fms and asas to 0
                f.write('# Set dt of FMS and ASAS to 0\n')
                f.write(f'{self.tim2txt(0)}>DT FMS 0.001\n')
                f.write(f'{self.tim2txt(0)}>DT ASAS 0.001\n\n')
                f.write(f'{self.tim2txt(0)}>DT AREA 0.001\n\n')

                # Create aircraft.
                f.write('# Create aircraft.\n')
                max_id = max([ac['id'] for ac in scn['scenario']])
                len_ac_id = len(str(max_id))
                for ac in scn['scenario']:
                    time_string = self.tim2txt(ac['departure_time'])
                    callsign = f'UAV{str(ac["id"]).zfill(len_ac_id)}'

                    # Write to .scn file.
                    f.write(f'# Creating aircraft no. {ac["id"]}\n')
                    f.write(f'{time_string}>CRE {callsign} {ac["ac_type"]} '
                            f'{ac["origin"][0]} {ac["origin"][1]} {ac["hdg"]} {DEPARTURE_ALTITUDE} {tas}\n')
                    f.write(f'{time_string}>ORIG {callsign} {ac["origin"][0]} {ac["origin"][1]}\n')
                    f.write(f'{time_string}>DEST {callsign} {ac["destination"][0]} {ac["destination"][1]}\n')
                    for wpt in ac["path"][1:-1]:
                        f.write(f'{time_string}>ADDWPT {callsign} {wpt[0]} {wpt[1]} {CRUISE_ALTITUDE} {tas}\n')
                    f.write(f'{time_string}>LNAV {callsign} ON\n')
                    f.write(f'{time_string}>VNAV {callsign} ON\n')
                    f.write('\n')
                print(f'Written {scn_path / scn_file}')

        if len(all_scen) > 1:
            # Write batch file
            flow_ratio_string = ''.join(str(round(flow * 100)).zfill(2) for flow in self.flow_ratio)
            filename_nr = f'batch_{prefix}_{flow_ratio_string}_NR.scn'
            filename_wr = f'batch_{prefix}_{flow_ratio_string}_WR.scn'
            safety_factor = 1.5
            max_rep_nr = max(scn['rep'] for scn in all_scen) + 1

            for scn_file in [filename_nr, filename_wr]:
                with open(scn_path / scn_file, 'w') as f:
                    f.write('# ########################################### #\n')
                    f.write(f'# Batch file {scn_file} for files:\n')
                    for fname in all_scn_files:
                        f.write(f'# File: {fname}\n')
                    f.write(f'# Flow ratio: {", ".join(str(scn["flow_ratio"]) for scn in all_scen)}\n')
                    f.write(f'# Saturation: '
                            f'{", ".join(str(scn["sat"]).format(".0f") for scn in all_scen)}\n')
                    f.write(f'# Number of repetitions: {max_rep_nr}\n')
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
                    f.write(f'# Mean route length: {2 * APPROACH_DISTANCE:.1f}m\n')
                    f.write(f'# NOTE: this scenario requires plugins: SPEEDBASED, URBAN_AREA\n')
                    f.write('# ########################################### #\n\n')

                    for i in range(len(all_scen)):
                        t = sum(all_scen[i]['duration']) * safety_factor
                        f.write(f'{self.tim2txt(i * t)}>PCALL URBAN/{all_scn_files[i]} REL\n')
                        if 'WR' in scn_file:
                            f.write(f'{self.tim2txt(i * t)}>RESO SPEEDBASED\n')
                        f.write(f'{self.tim2txt(i * t)}>OP\n')
                        f.write(f'{self.tim2txt(i * t)}>FF {t}\n')
                        f.write(f'{self.tim2txt((i + 1) * t)}>CLOSELOG\n\n')
                    f.write(f'{self.tim2txt((i + 1) * t + 10)}>QUIT\n\n')

                print(f'Written {scn_path / scn_file}')


if __name__ == '__main__':
    # FLOW_RATIO = (EASTBOUND, NORTHBOUND, EAST-NORTH TURN, NORTH-EAST TURN).
    FLOW_RATIO = (0.6, 0.4, 0.0, 0.0)  # Sum should be 1.
    REPETITIONS = 10

    BUILD_UP_DURATION = 15 * 60.  # s
    EXPERIMENT_DURATION = 45 * 60.  # s
    COOL_DOWN_DURATION = 15 * 60.  # s
    DURATION = (BUILD_UP_DURATION, EXPERIMENT_DURATION, COOL_DOWN_DURATION)
    PREFIX = 'turn_exp'

    SPEED = 10.  # m/s

    S_H = 50.  # m
    S_V = 25.  # ft
    T_L = 20.  # s
    REACTIONTIME = 5.  # s

    scen_gen = ScenarioGenerator()
    all_scenarios = scen_gen.create_scenario(FLOW_RATIO, REPETITIONS, SPEED, DURATION, S_H, S_V, T_L, REACTIONTIME)
    scen_gen.write_scenario(all_scenarios, prefix=PREFIX)
