"""
Urban environment orthogonal grid plugin.
Can also be used by a scenario_generator to create scenarios, without including this plugin in bluesky.

Created by Michiel Aarts, March 2021
"""
import numpy as np
from typing import List, Tuple
import pandas as pd
from bluesky import navdb
from bluesky.tools.aero import nm
from bluesky.core import Entity
import random
from bluesky.tools.geo import kwikqdrdist

N_ROWS = 19
N_COLS = N_ROWS
GRID_HEIGHT = 200  # m
GRID_WIDTH = GRID_HEIGHT
S_h = 100 / nm
t_l = 60.


### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    """ Plugin initialisation function. """
    UrbanGrid(N_ROWS, N_COLS, GRID_WIDTH, GRID_HEIGHT, load_grid=True)

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name': 'URBAN',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type': 'sim',
    }

    # init_plugin() should always return a configuration dict.
    return config


class UrbanGrid(Entity):
    def __init__(self, n_rows: int, n_cols: int, grid_width: float, grid_height: float, load_grid: bool = False):
        """
        Instantiates a counterclockwise orthogonal grid at location (0, 0).

        :param n_rows: Number of rows (should be a multiple of 4, minus 1)
        :param n_cols: Number of columns (should be a multiple of 4, minus 1)
        :param grid_width: Distance in meters between two longitudinal nodes
        :param grid_height: Distance in meters between two lateral nodes
        :param load_grid: Loads the grid as waypoints in BlueSky (default: false)
        """
        super().__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.load_grid = load_grid

        self.name_length = max([len(str(self.n_rows)), len(str(self.n_cols))])
        self.row_sep = self.m_to_lon(self.grid_width)
        self.col_sep = self.m_to_lat(self.grid_height)

        # Variables for the nodes and edges.
        self.nodes = {}
        self.edges = {}
        self.all_nodes = []
        self.od_nodes = []
        self.isct_nodes = []
        self.center_node = None

        # Variables for the bounding box.
        self.min_lat = None
        self.min_lon = None
        self.max_lat = None
        self.max_lon = None
        self.area = (self.n_rows - 1) * self.grid_height * (self.n_cols - 1) * self.grid_width

        self.route_accuracy = 1E4
        self._paths = None
        self._pathlengths = None
        self._calculated_mean_route_length = None
        self._flow_df = None
        self._from_flow_df = None

        self.create_nodes()
        self.load_edges()

        if self.load_grid:
            self.load_city_grid()

    # def reset(self):
        # # TODO Apparently plugin reset is called before the navdb is reset, thus this does not yet work.
        # super().reset()
        # if self.load_grid:
        #     self.load_city_grid()

    @staticmethod
    def m_to_lat(v_sep: float) -> float:
        return v_sep / 110574.

    @staticmethod
    def m_to_lon(h_sep: float, lat: float = 0) -> float:
        return h_sep / (111320. * np.cos(np.deg2rad(lat)))

    def create_nodes(self) -> None:
        """
        Creates a grid of nodes.

        Outer path should be a counterclockwise circle of E->N->W->S, starting from (0, 0).
        Smallest version:
        SW - W - NW
        |        |
        S    .   N
        |        |
        SE - E - NE

        :return: None
        """
        # Sanity check on inputs.
        if (self.n_rows + 1) % 4 != 0:
            raise ValueError(f'Amount of rows should be a multiple of 4, minus 1\nCurrent value: {self.n_rows}')
        if (self.n_cols + 1) % 4 != 0:
            raise ValueError(f'Amount of columns should be a multiple of 4, minus 1\nCurrent value: {self.n_cols}')

        self.all_nodes = []
        for row in np.arange(self.n_rows):
            for col in np.arange(self.n_cols):
                if row % 2 == 0 or col % 2 == 0:
                    if row % 4 == 0:
                        ew = 'E'
                    elif row % 2 == 0:
                        ew = 'W'
                    else:
                        ew = ''
                    if col % 4 == 0:
                        ns = 'S'
                    elif col % 2 == 0:
                        ns = 'N'
                    else:
                        ns = ''
                    row_id = str(row).zfill(self.name_length)
                    col_id = str(col).zfill(self.name_length)
                    node = f'CG{row_id}{col_id}'
                    lat = self.row_sep * row
                    lon = self.col_sep * col
                    self.nodes[node] = {'lat': lat, 'lon': lon,
                                        'dir': ns + ew,
                                        'row_id': row_id, 'col_id': col_id}
                    self.all_nodes.append(node)

        self.center_node = self.all_nodes[round(len(self.all_nodes) / 2)]

        first_node = self.all_nodes[0]
        last_node = self.all_nodes[-1]
        self.min_lat = self.nodes[first_node]['lat']
        self.min_lon = self.nodes[first_node]['lon']
        self.max_lat = self.nodes[last_node]['lat']
        self.max_lon = self.nodes[last_node]['lon']

        # Origin-Destination nodes are all nodes halfway two intersections,
        # i.e.: all nodes with only one direction.
        for node in self.all_nodes:
            if len(self.nodes[node]['dir']) == 1:
                self.od_nodes.append(node)
            else:
                self.isct_nodes.append(node)

    def load_edges(self) -> None:
        """
        Determines the unidirectional edges between all nodes,
        to be used in the shortest path algorithm.

        :return: None
        """
        for node in self.nodes.keys():
            self.edges[node] = {}
            row_id = self.nodes[node]['row_id']
            col_id = self.nodes[node]['col_id']
            row = int(row_id)
            col = int(col_id)
            for direction in self.nodes[node]['dir']:
                if direction == 'N':
                    hdg = 0.
                    length = self.grid_height
                    target_row_id = str(row + 1).zfill(self.name_length)
                    target = f'CG{target_row_id}{col_id}'
                elif direction == 'E':
                    hdg = 90.
                    length = self.grid_width
                    target_col_id = str(col + 1).zfill(self.name_length)
                    target = f'CG{row_id}{target_col_id}'
                elif direction == 'S':
                    hdg = 180.
                    length = self.grid_height
                    target_row_id = str(row - 1).zfill(self.name_length)
                    target = f'CG{target_row_id}{col_id}'
                elif direction == 'W':
                    hdg = 270.
                    length = self.grid_width
                    target_col_id = str(col - 1).zfill(self.name_length)
                    target = f'CG{row_id}{target_col_id}'
                else:
                    raise Exception(f'Something went wrong, dir={direction}')
                # Check if node exists
                if target in self.nodes.keys():
                    self.edges[node][target] = {'length': length, 'hdg': hdg}

    def load_city_grid(self) -> None:
        """
        Loads the city grid as waypoints directly in the bluesky navdb.

        :return: None
        """
        for node in self.nodes.keys():
            navdb.defwpt(name=node, lat=self.nodes[node]['lat'], lon=self.nodes[node]['lon'], wptype='citygrid')

    def city_grid_scenario(self, timestamp: str = '00:00:00.00') -> List[str]:
        """
        Loads the city grid as waypoints as stack commands.

        :return: list of commands to set up the grid in a scenario file
        """
        commands = ['# Load the city grid\n']
        for node in self.nodes.keys():
            commands.append(f'{timestamp}>DEFWPT {node} {self.nodes[node]["lat"]} {self.nodes[node]["lon"]}\n')
        commands.append('\n')
        return commands

    def calculate_shortest_path(
        self, origin: str, destination: str, prio: str = 'turns'
    ) -> Tuple[List[str], float, float, int]:
        """
        Gets the shortest path with the least amount of corners between origin and destination nodes

        :return: (Path: list[node_id], Path length [m], Path altitude variation [m], Number of turns)
        """
        # Sanity check.
        if origin == destination:
            raise ValueError('Origin == Destination, cannot calculate shortest path')

        # Shortest paths is a dict of nodes, whose value is a dict of the shortest path weights to that node.
        shortest_paths = {origin: {'prev_node': None, 'prev_hdg': None, 'dist': 0, 'turns': 0, 'alt_var': 0}}
        current = origin
        visited = set()

        while current != destination:
            visited.add(current)
            destinations = self.edges[current]
            weight_to_current = shortest_paths[current]

            for next_id in destinations:
                distance_to_next_node = self.edges[current][next_id]['length']
                hdg_to_next_node = self.edges[current][next_id]['hdg']

                # Add new node weights.
                w_distance = distance_to_next_node + weight_to_current['dist']
                w_alt_var = weight_to_current['alt_var']
                w_turns = weight_to_current['turns']
                w_hdg = weight_to_current['prev_hdg']

                # Check for turns.
                if w_hdg is None:
                    # Departing from origin.
                    w_hdg = hdg_to_next_node
                elif w_hdg != hdg_to_next_node:
                    # Making a turn.
                    w_alt_var += 0
                    w_turns += 1
                    w_hdg = hdg_to_next_node

                next_dict = {'prev_node': current, 'prev_hdg': w_hdg,
                             'dist': w_distance, 'turns': w_turns, 'alt_var': w_alt_var}
                # Is this a new node?
                if next_id not in shortest_paths:
                    shortest_paths[next_id] = next_dict
                # If we already have this node, is this a better way to get there?
                else:
                    cur_best_weight = shortest_paths[next_id]
                    if prio == 'turns':
                        # First prioritize turns, then path length
                        if cur_best_weight['turns'] < w_turns:
                            # Keep current node.
                            pass
                        elif cur_best_weight['turns'] == w_turns:
                            # Check for path length.
                            if cur_best_weight['dist'] > w_distance:
                                # Keep new node.
                                shortest_paths[next_id] = next_dict
                            else:
                                # Distance to next is longer or equal, keep current node.
                                pass
                        else:
                            # Keep new node.
                            shortest_paths[next_id] = next_dict
                    else:
                        raise NotImplementedError(f'Shortest path prio {prio} not yet implemented')

            next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}

            # Next node is the destination with the highest priority.
            if prio == 'turns':
                min_turns = np.inf
                min_turns_nodes = {}
                for node in next_destinations.keys():
                    # Check for turns.
                    if next_destinations[node]['turns'] < min_turns:
                        # Destination with least amount of turns.
                        min_turns = next_destinations[node]['turns']
                        min_turns_nodes = {node: next_destinations[node]}
                    elif next_destinations[node]['turns'] == min_turns:
                        # Destination with equally least amount of turns.
                        min_turns_nodes[node] = next_destinations[node]
                min_dist = np.inf
                for node in min_turns_nodes.keys():
                    # Check for distance.
                    if min_turns_nodes[node]['dist'] < min_dist:
                        min_dist = min_turns_nodes[node]['dist']
                        current = node
            else:
                raise NotImplementedError(f'Shortest path prio {prio} not yet implemented')

            # Sanity check
            if current in visited:
                raise ValueError('Something went wrong')

        # Work back through destinations in shortest path.
        path = []
        pathlength = shortest_paths[current]['dist']
        path_turns = shortest_paths[current]['turns']
        path_alt_variatons = shortest_paths[current]['alt_var']
        while current is not None:
            path.append(current)
            next_node = shortest_paths[current]['prev_node']
            current = next_node
        # Reverse path.
        path = path[::-1]

        return path, pathlength, path_alt_variatons, path_turns

    def _evaluate_routes(self) -> None:
        """
        Evaluates 10k routes for mean route length and the flow_df.

        :return:
        """
        num_iterations = int(self.route_accuracy)
        paths = np.zeros(num_iterations, dtype=object)
        pathlengths = np.zeros(num_iterations, dtype=float)
        for i in range(num_iterations):
            origin = random.choice(self.od_nodes)
            destination = origin
            while destination == origin:
                destination = random.choice(self.od_nodes)
            paths[i], pathlengths[i], _, _ = self.calculate_shortest_path(origin, destination)
        self._paths = paths
        self._pathlengths = pathlengths
        self._calculated_mean_route_length = float(np.mean(self._pathlengths))

    @property
    def mean_route_length(self) -> float:
        """
        Getter for the mean route length through the grid.

        :return: Mean route length [m]
        """
        if self._calculated_mean_route_length is None:
            print('Urban.py>>>Calculating mean route length. This may take some time.')
            self._evaluate_routes()
            print('Urban.py>>>Mean route length obtained.')
        return self._calculated_mean_route_length

    def _create_flow_df(self) -> None:
        """
        Creates a flow dataframe, to use as a base for flow rate calculations.
        It expresses the mean expected total passages of a node per single aircraft.
        To convert to a flow rate, the airspeed and n_inst are required.

        :return: None
        """
        all_angles = np.array(range(5)) * 90
        routing_dict = {}

        # Get paths.
        if self._paths is None:
            self._evaluate_routes()

        # Loop through paths.
        found_paths = set()
        for path in self._paths:
            for i in range(len(path) - 2):
                # Does not include origin and destination passages of nodes in via.
                frm, via, to = path[i:i + 3]
                if (frm, via, to) in found_paths:
                    routing_dict[(frm, via, to)]['num'] += 1
                else:
                    # Check if corner, i.e. bearing of from and to is approx 45 / 135 / 225 / 315 deg.
                    found_paths.add((frm, via, to))
                    frm_node = self.nodes[frm]
                    via_node = self.nodes[via]
                    to_node = self.nodes[to]
                    frm_qdr, _ = kwikqdrdist(frm_node['lat'], frm_node['lon'],
                                             via_node['lat'], via_node['lon'])
                    to_qdr, _ = kwikqdrdist(via_node['lat'], via_node['lon'],
                                            to_node['lat'], to_node['lon'])
                    frm_hdg = all_angles[np.isclose(frm_qdr, all_angles, atol=5)][0]
                    to_hdg = all_angles[np.isclose(to_qdr, all_angles, atol=5)][0]
                    if frm_hdg == 360:
                        frm_hdg = 0.
                    if to_hdg == 360:
                        to_hdg = 0.
                    if frm_hdg != to_hdg:
                        corner = True
                    else:
                        corner = False
                    routing_dict[(frm, via, to)] = {'num': 1, 'corner': corner, 'from_hdg': frm_hdg, 'to_hdg': to_hdg,
                                                    'lat': via_node['lat'], 'lon': via_node['lon']}

        # Convert to df.
        routing_df = pd.DataFrame.from_dict(routing_dict, orient='index')
        routing_df.index = routing_df.index.rename(['from', 'via', 'to'])

        # Normalize to obtain distribution.
        routing_df['flow_distribution'] = routing_df['num'] / routing_df['num'].sum()

        self._flow_df = routing_df

    @property
    def flow_df(self) -> pd.DataFrame:
        """
        Getter for the flow distribution dataframe.

        :return: the flow distribution dataframe
        """
        if self._flow_df is None:
            print('Urban.py>>>Calculating flow dataframe.')
            self._create_flow_df()
            print('Urban.py>>>Flow dataframe obtained.')
        return self._flow_df

    def _create_from_flow_df(self) -> None:
        """
        Creates a from_flow dataframe.
        It expresses the proportion of the passages of a section compared to other sections.
        Sum of proportion = 1.
        To convert to a flow rate, the airspeed and n_inst are required.

        :return: None
        """
        all_angles = np.array(range(5)) * 90
        ff_dict = {}

        # Get paths.
        if self._paths is None:
            self._evaluate_routes()

        # Loop through paths.
        found_sections = set()
        for path in self._paths:
            for i in range(len(path) - 1):
                # Does not include origin and destination passages of nodes in via.
                frm, via = path[i:i + 2]
                if (frm, via) in found_sections:
                    ff_dict[(frm, via)]['num'] += 1
                else:
                    # Check if corner, i.e. bearing of from and to is approx 45 / 135 / 225 / 315 deg.
                    found_sections.add((frm, via))
                    frm_node = self.nodes[frm]
                    via_node = self.nodes[via]
                    qdr, _ = kwikqdrdist(frm_node['lat'], frm_node['lon'],
                                         via_node['lat'], via_node['lon'])
                    hdg = all_angles[np.isclose(qdr, all_angles, atol=5)][0]
                    if hdg == 360:
                        hdg = 0.

                    ff_dict[(frm, via)] = {'num': 1, 'hdg': hdg}

        # Convert to dataframe.
        ff_df = pd.DataFrame.from_dict(ff_dict, orient='index')
        ff_df.index = ff_df.index.rename(['from', 'via'])

        # Normalize to obtain distribution.
        ff_df['flow_distribution'] = ff_df['num'] / ff_df['num'].sum()

        self._from_flow_df = ff_df

    @property
    def from_flow_df(self) -> pd.DataFrame:
        """
        Getter for the from_flow distribution dataframe.

        :return: the from_flow distribution dataframe
        """
        if self._from_flow_df is None:
            print('Urban.py>>>Calculating from_flow dataframe.')
            self._create_from_flow_df()
            print('Urban.py>>>From flow dataframe obtained.')
        return self._from_flow_df
