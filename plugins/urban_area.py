"""
BlueSky urban area plugin. Adaptation of area.py.
This plugin defines an experiment area to log aircraft and
deletes aircraft that reach their destination. Statistics on these flights can be
logged with the FLSTLOG logger, while the CONFLOG logger logs the conflicts.

Created by Michiel Aarts, March 2021
"""
import numpy as np
from bluesky import traf, sim
from bluesky.tools import datalog, areafilter
from bluesky.core import Entity, timed_function
from bluesky.tools.aero import ft, fpm
import datetime

# Log parameters for the flight statistics log
flst_header = \
    '#######################################################\n' + \
    'FLST LOG\n' + \
    'Flight Statistics\n' + \
    '#######################################################\n\n' + \
    'Parameters [Units]:\n' + \
    'Deletion Time [s], ' + \
    'Call sign [-], ' + \
    'Spawn Time [s], ' + \
    'Flight time [s], ' + \
    'Actual Distance 2D [m], ' + \
    'Actual Distance 3D [m], ' + \
    'Work Done [MJ], ' + \
    'Latitude [deg], ' + \
    'Longitude [deg], ' + \
    'Altitude [ft], ' + \
    'TAS [m/s], ' + \
    'Vertical Speed [fpm], ' + \
    'Heading [deg], ' + \
    'ASAS Active [bool], ' + \
    'Pilot ALT [ft], ' + \
    'Pilot SPD (TAS) [m/s], ' + \
    'Pilot HDG [deg], ' + \
    'Pilot VS [fpm]' + \
    'CR [bool]' + '\n'

flst_vars = 'arrival_time, callsign, departure_time, flight_time, dist2D, dist3D, work_done, lat, lon, alt, tas,' \
           'vs, hdg, asas, apalt, aptas, aphdg, apvs, cr\n'

conf_header = \
    '#######################################################\n' + \
    '# CONF LOG\n' + \
    '# Conflict Statistics\n' + \
    '#######################################################\n\n' + \
    'Parameters [Units]:\n' + \
    'Simulation time [s], ' + \
    'Inst. number of aircraft [-], ' + \
    'Inst. number of conflicts [-], ' + \
    'Inst. number of losses of separation [-], ' + \
    'Total number of aircraft [-], ' + \
    'Total number of conflicts [-], ' + \
    'Total number of losses of separation [-], ' + \
    'CR [bool]\n'

conf_vars = 't, ni_ac, ni_conf, ni_los, ntotal_ac, ntotal_conf, ntotal_los, cr\n'

# Global data
area = None


### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    # Additional initialization code
    global area
    area = Area()

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name': 'URBAN_AREA',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type': 'sim'
    }

    stackfunctions = {
        'AREA': [
            'AREA Shapename/OFF or AREA lat,lon,lat,lon,[top,bottom]',
            '[float/txt,float,float,float,alt,alt]',
            area.set_area,
            'Define experiment area'
        ],
        'CLOSELOG': [
            'CLOSELOG',
            '',
            area.close_log,
            'Close experiment logs'
        ],
        'LOGPREFIX': [
            'LOGPREFIX name',
            '[txt]',
            area.set_log_prefix,
            'Set the log prefix'
        ]
    }
    # init_plugin() should always return these two dicts.
    return config, stackfunctions


class Area(Entity):
    """ Traffic area: delete traffic when they reach their destination (i.e. when they switch off LNAV). """

    def __init__(self):
        super().__init__()
        # Parameters of area
        self.active = False
        self.exp_area = ''
        self.prevconfpairs = set()
        self.ntotal_conf = 0
        self.prevlospairs = set()
        self.ntotal_los = 0
        self.ntotal_ac = 0
        self.log_prefix = ''

        self.flst_log = datalog.crelog('FLSTLOG', None, flst_header)
        self.conf_log = datalog.crelog('CONFLOG', None, conf_header)

        with self.settrafarrays():
            self.inside_exp = np.array([], dtype=np.bool)  # In experiment area or not.
            self.distance2D = np.array([])
            self.distance3D = np.array([])
            self.dstart2D = np.array([])
            self.dstart3D = np.array([])
            self.workstart = np.array([])
            self.entrytime = np.array([])
            self.create_time = np.array([])

    def reset(self):
        """ Reset area state when simulation is reset. """
        super().reset()
        self.active = False
        self.exp_area = ''
        self.prevconfpairs = set()
        self.ntotal_conf = 0
        self.prevlospairs = set()
        self.ntotal_los = 0
        self.ntotal_ac = 0
        self.log_prefix = ''

    def create(self, n=1):
        """ Create is called when new aircraft are created. """
        super().create(n)
        self.inside_exp[-n:] = False
        self.distance2D[-n:] = 0.
        self.distance3D[-n:] = 0.
        self.dstart2D[-n:] = None
        self.dstart3D[-n:] = None
        self.workstart[-n:] = None
        self.entrytime[-n:] = None
        self.create_time[-n:] = sim.simt
        self.ntotal_ac += n

    @timed_function(name='AREA', dt=1.0)
    def update(self, dt):
        """ Log all desired data if AREA is active. """
        if self.active:
            total_spd = np.sqrt(traf.gs * traf.gs + traf.vs * traf.vs)
            self.distance2D += dt * traf.gs
            self.distance3D += dt * total_spd

            # Check whether all aircraft are currently inside the experiment area.
            inside_exp = areafilter.checkInside(self.exp_area, traf.lat, traf.lon, traf.alt)
            if not np.all(inside_exp):
                raise RuntimeError('An aircraft escaped the experiment area!')

            # Check for arrived flights.
            # Upon reaching destination, autopilot switches off the LNAV.
            arrived = ~traf.swlnav

            # Count new conflicts and losses of separation.
            # Store statistics for all new conflict pairs, i.e.:
            # Conflict pairs detected in the current timestep that were not yet
            # present in the previous timestep.
            confpairs_new = list(traf.cd.confpairs_unique - self.prevconfpairs)
            lospairs_new = list(traf.cd.lospairs_unique - self.prevlospairs)

            # Ignore conflicts and losses for descending or arrived aircraft.
            ignore_confpair = set()
            ignore_lospair = set()
            for pair in traf.cd.confpairs_unique:
                for ac in pair:
                    if traf.vs[traf.id.index(ac)] < -1E-4 or arrived[traf.id.index(ac)]:
                        ignore_confpair.add(pair)
            for pair in traf.cd.lospairs_unique:
                for ac in pair:
                    if traf.vs[traf.id.index(ac)] < -1E-4 or arrived[traf.id.index(ac)]:
                        ignore_lospair.add(pair)
            [confpairs_new.remove(pair) for pair in ignore_confpair if pair in confpairs_new]
            [lospairs_new.remove(pair) for pair in ignore_lospair if pair in lospairs_new]

            # if lospairs_new:
            #     print('LoS found:', lospairs_new)
            #     print()

            self.ntotal_conf += len(confpairs_new)
            self.ntotal_los += len(lospairs_new)

            self.conf_log.log(traf.ntraf, len(traf.cd.confpairs_unique) - len(ignore_confpair),
                              len(traf.cd.lospairs_unique) - len(ignore_lospair),
                              self.ntotal_ac, self.ntotal_conf, self.ntotal_los, bool(traf.cr.do_cr))

            # Log distance values upon entry of experiment area (includes spawning aircraft).
            newentries = np.logical_not(self.inside_exp) * inside_exp
            self.dstart2D[newentries] = self.distance2D[newentries]
            self.dstart3D[newentries] = self.distance3D[newentries]
            self.workstart[newentries] = traf.work[newentries]
            self.entrytime[newentries] = sim.simt

            # Update values for next loop.
            self.inside_exp = inside_exp
            self.prevconfpairs = set(traf.cd.confpairs_unique)
            self.prevlospairs = set(traf.cd.lospairs_unique)
            [self.prevconfpairs.remove(pair) for pair in ignore_confpair]
            [self.prevlospairs.remove(pair) for pair in ignore_lospair]

            # Log flight statistics when reaching destination and delete aircraft.
            del_idx = np.flatnonzero(arrived)
            self.flst_log.log(
                np.array(traf.id)[del_idx],
                self.create_time[del_idx],
                sim.simt - self.entrytime[del_idx],
                (self.distance2D[del_idx] - self.dstart2D[del_idx]),
                (self.distance3D[del_idx] - self.dstart3D[del_idx]),
                (traf.work[del_idx] - self.workstart[del_idx]) * 1e-6,
                traf.lat[del_idx],
                traf.lon[del_idx],
                traf.alt[del_idx] / ft,
                traf.tas[del_idx],
                traf.vs[del_idx] / fpm,
                traf.hdg[del_idx],
                traf.cr.active[del_idx],
                traf.aporasas.alt[del_idx] / ft,
                traf.aporasas.tas[del_idx],
                traf.aporasas.hdg[del_idx],
                traf.aporasas.vs[del_idx] / fpm,
                traf.cr.do_cr
            )
            traf.delete(del_idx)

    def set_area(self, *args):
        """ Set Experiment Area. Aircraft leaving this experiment area raise an error.
        Input can be existing shape name, or a box with optional altitude constraints. """
        # Set both exp_area and del_area to the same size.
        curname = self.exp_area
        msgname = 'Experiment area'
        # If no args, print current area state.
        if not args:
            return True, f'{msgname} is currently ON (name={curname})' if self.active else \
                f'{msgname} is currently OFF'

        # If the first argument is a string, it is an area name.
        if isinstance(args[0], str) and len(args) == 1:
            if areafilter.hasArea(args[0]):
                # Switch on area, set it to the shape name.
                self.exp_area = args[0]
                self.active = True

                # Initiate the loggers.
                self.flst_log.start(prefix=self.log_prefix)
                self.conf_log.start(prefix=self.log_prefix)
                self.flst_log.writeline(flst_vars)
                self.conf_log.writeline(conf_vars)

                return True, f'{msgname} is set to {args[0]}'
            elif args[0][:2] == 'OF':
                # Switch off the area and reset the logger.
                self.close_log()
                return True, f'{msgname} is switched OFF\nLogs are closed'
            elif args[0][:2] == 'ON':
                if not curname:
                    return False, 'No area defined.'
                else:
                    self.active = True
                    return True, f'{msgname} switched ON (name={curname})'
            else:
                # Shape name is unknown.
                return False, 'Shapename unknown. ' + \
                       'Please create shapename first or shapename is misspelled!'
        # If first argument is a float, make a box with the arguments.
        if isinstance(args[0], (float, int)) and 4 <= len(args) <= 6:
            self.active = True
            self.exp_area = 'EXPAREA'
            areafilter.defineArea('EXPAREA', 'BOX', args[:4], *args[4:])

            # Initiate the loggers.
            self.flst_log.start(prefix=self.log_prefix)
            self.conf_log.start(prefix=self.log_prefix)
            self.flst_log.writeline(flst_vars)
            self.conf_log.writeline(conf_vars)

            return True, f'{msgname} is ON. Area name is: {self.exp_area}'
        else:
            return False, 'Incorrect arguments\n' + \
                   'AREA Shapename/OFF or\n Area lat,lon,lat,lon,[top,bottom]'

    def close_log(self):
        """
        Close the logfiles and reset area.
        A new area / experiment can be defined afterwards.
        This helps if QUIT does not let the logs finish properly.
        """
        if traf.ntraf > 0:
            print(f'Warning! {self.ntraf} aircraft remaining while closing the logs!')
        datalog.reset()
        self.reset()
        return True, 'Logs are closed\nExperiment area set to None'

    def set_log_prefix(self, name):
        """
        Add an extra prefix to logs.
        Useful when using pcall in batch scenarios.
        Set before defining area.

        Also prints to the terminal.
        That way, the batch scenario can be tracked in detached mode.
        """
        self.log_prefix = name
        print(f'{datetime.datetime.now().strftime("%H:%M:%S")}>Evaluating scenario {name}')
        return True, f'Log prefix set to {name}'
