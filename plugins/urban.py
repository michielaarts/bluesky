""" Urban environment grid tester """
import numpy as np
from bluesky import navdb, stack  # traf, core, settings, sim, scr, tools
from bluesky.tools.aero import nm


### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    """ Plugin initialisation function. """
    stack.stack('ECHO Loading city grid...')
    N_ROWS = 20
    N_COLS = N_ROWS
    HORIZONTAL_SEPARATION_KM = 1
    VERTICAL_SEPARATION_KM = 1
    S_h = 100 / nm
    t_l = 60.

    load_city_grid(N_ROWS, N_COLS, km_to_lon(HORIZONTAL_SEPARATION_KM, 0), km_to_lat(VERTICAL_SEPARATION_KM))

    # Turn on Speed-based conflict resolution
    stack.stack('ASAS ON')
    stack.stack('RESO SpeedBased')
    stack.stack(f'ZONER {S_h}')
    stack.stack(f'DTLOOK {t_l}')

    # Create dummy aircrafts
    stack.stack('CRE UAV001 M600 CG0002 90 100 15')
    stack.stack('CRE UAV002 M600 CG0000 90 100 30')
    stack.stack('CRE UAV003 M600 CG0202 180 100 29')
    stack.stack('HOLD')
    stack.stack('ADDWPT UAV001 CG0004 100 30')
    stack.stack('ADDWPT UAV001 CG0404 100 30')
    stack.stack('UAV001 DEST CG0409')
    stack.stack('ADDWPT UAV002 CG0005 100 30')
    stack.stack('ADDWPT UAV002 CG0006 100 25')
    stack.stack('ADDWPT UAV003 CG0002 100 29')
    stack.stack('ADDWPT UAV003 CG0004 100 29')
    stack.stack('UAV003 DEST CG0008')
    stack.stack('POS UAV001')
    stack.stack('PAN UAV001')
    stack.stack('ZOOM 20')
    stack.stack('TRAIL ON')

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name': 'URBAN',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type': 'sim',
    }

    # init_plugin() should always return a configuration dict.
    return config


def km_to_lat(v_sep):
    return v_sep / 110.574


def km_to_lon(h_sep, lat):
    return h_sep / (111.320 * np.cos(np.deg2rad(lat)))


def load_city_grid(n_rows: int, n_cols: int, row_sep: float, col_sep: float):
    name_length = max([len(str(n_rows)), len(str(n_cols))])
    for row in np.arange(n_rows):
        for col in np.arange(n_cols):
            if row % 2 == 0 or col % 2 == 0:
                navdb.defwpt(name=f'CG{str(row).zfill(name_length)}{str(col).zfill(name_length)}',
                             lat=row_sep * row,
                             lon=col_sep * col,
                             wptype='citygrid')

