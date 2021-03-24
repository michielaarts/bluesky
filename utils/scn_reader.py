"""
Plots the flow rates of a scenario pickle.

Created by Michiel Aarts, March 2021
"""

import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
from tkinter import Tk, filedialog
from pathlib import Path
import pandas as pd
from bluesky.tools.geo import kwikqdrdist
from typing import Tuple


def create_routing_df(scn: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create routing dataframe.

    Note: may take a long time for densities > 100 inst. no. of ac.

    :param scn: scenario dict from scenario_generator.py
    :return: (Flows per node, Routing dataframe)
    """
    all_angles = np.array(range(9)) * 45
    corner_angles = np.array(range(4)) * 90 + 45
    index_df = pd.MultiIndex.from_frame(pd.DataFrame({'from': [], 'via': [], 'to': []}))
    routing_df = pd.DataFrame({'num': [], 'corner': [], 'hdg': [], 'lat': [], 'lon': []}, index=index_df)
    origins = {}
    destinations = {}
    for ac in scn['scenario']:
        if scn['duration'][0] < ac['departure_time'] < sum(scn['duration'][:2]):
            # Extract path per aircraft.
            for i in range(len(ac['path']) - 2):
                # Does not include origin and destination passages of nodes.
                frm, via, to = ac['path'][i:i+3]
                if routing_df.index.isin([(frm, via, to)]).any():
                    routing_df.loc[(frm, via, to)]['num'] += 1
                else:
                    # Check if corner, i.e. bearing of from and to is approx 45 / 135 / 225 / 315 deg.
                    frm_node = scn['grid_nodes'][frm]
                    via_node = scn['grid_nodes'][via]
                    to_node = scn['grid_nodes'][to]
                    qdr, _ = kwikqdrdist(frm_node['lat'], frm_node['lon'],
                                         to_node['lat'], to_node['lon'])
                    hdg = all_angles[np.isclose(qdr, all_angles, atol=5)][0]
                    if hdg == 360:
                        hdg = 0.
                    if hdg in corner_angles:
                        corner = True
                    else:
                        corner = False
                    routing_df.loc[(frm, via, to)] = {'num': 1, 'corner': corner, 'hdg': hdg,
                                                      'lat': via_node['lat'], 'lon': via_node['lon']}
            # Add origins and destinations per aircraft.
            if ac['origin'] in origins.keys():
                origins[ac['origin']] += 1
            else:
                origins[ac['origin']] = 1
            if ac['destination'] in destinations.keys():
                destinations[ac['destination']] += 1
            else:
                destinations[ac['destination']] = 1

    # Flow rate is approx. the number of passes of that node divided by the logging time.
    routing_df['flow_distribution'] = routing_df['num'] / scn['duration'][1]
    for key in origins.keys():
        origins[key] = origins[key] / scn['duration'][1]
    for key in destinations.keys():
        destinations[key] = destinations[key] / scn['duration'][1]

    # Pivot routing dataframe and add departing traffic.
    flow_df = (routing_df.groupby(['via', 'hdg'])['flow_distribution']
               .agg('sum').unstack('hdg')
               .merge(pd.Series(origins, name='origins'), how='left', left_index=True, right_index=True)
               .merge(pd.Series(destinations, name='destinations'), how='left', left_index=True, right_index=True))

    return flow_df, routing_df


def plot_flow_rates(routing_df: pd.DataFrame) -> None:
    """
    Plots the flow proportion of a routing_df.

    :param routing_df: dataframe from create_routing_df
    :return: None
    """
    flow_rates = (routing_df.copy()
                  .groupby('via').agg({'flow_distribution': 'sum', 'lat': 'mean', 'lon': 'mean'})
                  .pivot('lat', 'lon', 'flow_distribution'))

    plt.figure()
    plt.imshow(flow_rates, extent=[routing_df['lon'].min(), routing_df['lon'].max(),
                                   routing_df['lat'].min(), routing_df['lat'].max()])
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    plt.colorbar(label='Flow proportion [-]')


if __name__ == '__main__':
    # Select file.
    root = Tk()
    SCN_DATA_DIR = Path('../scenario/URBAN/Data/')
    filename = filedialog.askopenfilename(initialdir=SCN_DATA_DIR, filetypes=[('Scenario pickle', '*.pkl')])
    root.destroy()

    with open(Path(filename), 'rb') as f:
        scenario = pkl.load(f)

    flows, routing = create_routing_df(scenario)

    plot_flow_rates(routing)

    # # Save figure.
    # FIGURES_DIR = Path(r'C:\Users\michi\Dropbox\TU\Thesis\04_Prelim\Figures')
    # root = Tk()
    # savefile = filedialog.asksaveasfilename(initialdir=FIGURES_DIR,
    #                                         filetypes=[("Vector image", "*.svg"), ("EPS", "*.eps"), ("Any", "*")])
    # root.destroy()
    # plt.savefig(savefile)

    # Plot northbound flow rates.
    north_df = routing.copy()
    north_df.loc[north_df['hdg'] != 0, 'flow_rate'] = 0
    north_flow_rates = (north_df.groupby('via').agg({'flow_rate': 'sum', 'lat': 'mean', 'lon': 'mean'})
                        .pivot('lat', 'lon', 'flow_rate'))

    plt.figure()
    plt.imshow(north_flow_rates, extent=[north_df['lon'].min(), north_df['lon'].max(),
                                         north_df['lat'].min(), north_df['lat'].max()])
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    plt.colorbar(label='Flow rate [veh/s]')
    plt.title('Northbound flow rates [veh/s]')
