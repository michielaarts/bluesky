"""
BlueSky urban area plugin. Adaptation of area.py.
This plugin can use an area definition to log aircraft and
delete aircraft that reach their destination. Statistics on these flights can be
logged with the FLSTLOG logger.

Created by Michiel Aarts, March 2021
"""
import numpy as np
from bluesky import traf, sim
from bluesky.tools import datalog, areafilter
from bluesky.core import Entity, timed_function
from bluesky.tools.aero import ft, fpm

# Log parameters for the flight statistics log
flstheader = \
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
    'Origin Lat [deg], ' + \
    'Origin Lon [deg], ' + \
    'Destination Lat [deg], ' + \
    'Destination Lon [deg], ' + \
    'ASAS Active [bool], ' + \
    'Pilot ALT [ft], ' + \
    'Pilot SPD (TAS) [m/s], ' + \
    'Pilot HDG [deg], ' + \
    'Pilot VS [fpm]' + '\n'

flstvars = 't, arrival_time, callsign, departure_time, flight_time, dist2D, dist3D, work_done, lat, lon, alt, tas,' \
           ' vs, hdg, origin_lat, origin_lon, destination_lat, destination_lon, asas, apalt, aptas, aphdg, apvs\n'

confheader = \
    '#######################################################\n' + \
    '# CONF LOG\n' + \
    '# Conflict Statistics\n' + \
    '#######################################################\n\n' + \
    'Parameters [Units]:\n' + \
    'Simulation time [s], ' + \
    'Total number of conflicts in exp area [-], ' + \
    'Total number of losses of separation in exp area [-]\n'

confvars = 't, nconf, nlos\n'

# Global data
area = None


### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    # Addtional initilisation code
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
        'TAXI': [
            'TAXI ON/OFF [alt] : OFF auto deletes traffic below 1500 ft',
            'onoff[,alt]',
            area.set_taxi,
            'Switch on/off ground/low altitude mode, prevents auto-delete at 1500 ft'
        ]
    }
    # init_plugin() should always return these two dicts.
    return config, stackfunctions


class Area(Entity):
    """ Traffic area: delete traffic when they reach their destination """

    def __init__(self):
        super().__init__()
        # Parameters of area
        self.active = False
        self.exp_area = ''
        self.swtaxi = True  # Default ON: Doesn't do anything. See comments of set_taxi function below.
        self.swtaxialt = 1500.  # Default alt for TAXI OFF
        self.prevconfpairs = set()
        self.all_conf = 0
        self.prevlospairs = set()
        self.all_los = 0

        self.flst = datalog.crelog('FLSTLOG', None, flstheader)
        self.conflog = datalog.crelog('CONFLOG', None, confheader)

        with self.settrafarrays():
            self.inside_exp = np.array([], dtype=np.bool)  # In experiment area or not
            self.oldalt = np.array([])
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
        self.swtaxi = True
        self.swtaxialt = 1500.
        self.all_conf = 0

    def create(self, n=1):
        """ Create is called when new aircraft are created. """
        super().create(n)
        self.oldalt[-n:] = traf.alt[-n:]
        self.inside_exp[-n:] = False
        self.create_time[-n:] = sim.simt

    @timed_function(name='AREA', dt=1.0)
    def update(self, dt):
        """ Update flight efficiency metrics
            2D and 3D distance [m], and work done (force*distance) [J] """
        if self.active:
            total_spd = np.sqrt(traf.gs * traf.gs + traf.vs * traf.vs)
            self.distance2D += dt * traf.gs
            self.distance3D += dt * total_spd

            # Find out which aircraft are currently inside the experiment area.
            inside_exp = areafilter.checkInside(self.exp_area, traf.lat, traf.lon, traf.alt)
            if not np.all(inside_exp):
                raise RuntimeError('An aircraft escaped the experiment area!')

            # Count new conflicts and losses of separation.
            # Store statistics for all new conflict pairs, i.e.:
            # Conflict pairs detected in the current timestep that were not yet
            # present in the previous timestep.
            confpairs_new = list(traf.cd.confpairs_unique - self.prevconfpairs)
            lospairs_new = list(traf.cd.lospairs_unique - self.prevlospairs)
            if confpairs_new or lospairs_new:
                # If necessary: select conflict geometry parameters for new conflicts
                # idxdict = dict((v, i) for i, v in enumerate(traf.cd.confpairs))
                # idxnew = [idxdict.get(i) for i in confpairs_new]
                # dcpa_new = np.asarray(traf.cd.dcpa)[idxnew]
                # tcpa_new = np.asarray(traf.cd.tcpa)[idxnew]
                # tLOS_new = np.asarray(traf.cd.tLOS)[idxnew]
                # qdr_new = np.asarray(traf.cd.qdr)[idxnew]
                # dist_new = np.asarray(traf.cd.dist)[idxnew]

                self.all_conf += len(confpairs_new)
                self.all_los += len(lospairs_new)

                self.conflog.log(self.all_conf, self.all_los)

            self.prevconfpairs = set(traf.cd.confpairs)
            self.prevlospairs = set(traf.cd.lospairs)

            # Register distance values upon entry of experiment area (includes spawning aircraft).
            newentries = np.logical_not(self.inside_exp) * inside_exp
            self.dstart2D[newentries] = self.distance2D[newentries]
            self.dstart3D[newentries] = self.distance3D[newentries]
            self.workstart[newentries] = traf.work[newentries]
            self.entrytime[newentries] = sim.simt

            # Update inside_exp
            self.inside_exp = inside_exp

            # Log flight statistics when reaching destination.
            # Upon reaching destination, autopilot switches off the lnav.
            arrived = ~traf.swlnav
            if np.any(arrived):
                self.flst.log(
                    np.array(traf.id)[arrived],
                    self.create_time[arrived],
                    sim.simt - self.entrytime[arrived],
                    (self.distance2D[arrived] - self.dstart2D[arrived]),
                    (self.distance3D[arrived] - self.dstart3D[arrived]),
                    (traf.work[arrived] - self.workstart[arrived]) * 1e-6,
                    traf.lat[arrived],
                    traf.lon[arrived],
                    traf.alt[arrived] / ft,
                    traf.tas[arrived],
                    traf.vs[arrived] / fpm,
                    traf.hdg[arrived],
                    traf.cr.active[arrived],
                    traf.aporasas.alt[arrived] / ft,
                    traf.aporasas.tas[arrived],
                    traf.aporasas.vs[arrived] / fpm,
                    traf.aporasas.hdg[arrived])

            # Delete all arrived aircraft.
            delidx = np.flatnonzero(arrived)
            if len(delidx) > 0:
                traf.delete(delidx)

        # Autodelete for descending with swTaxi.
        if not self.swtaxi:
            delidxalt = np.where((self.oldalt >= self.swtaxialt)
                                 * (traf.alt < self.swtaxialt))[0]
            self.oldalt = traf.alt
            if len(delidxalt) > 0:
                traf.delete(list(delidxalt))

    def set_area(self, *args):
        """ Set Experiment Area. Aircraft leaving this experiment area raise an error.
        Input can be existing shape name, or a box with optional altitude constraints."""
        # Set both exp_area and del_area to the same size.
        curname = self.exp_area
        msgname = 'Experiment area'
        # if all args are empty, then print out the current area status
        if not args:
            return True, f'{msgname} is currently ON (name={curname})' if self.active else \
                f'{msgname} is currently OFF'

        # If the first argument is a string, it is an area name.
        if isinstance(args[0], str) and len(args) == 1:
            if areafilter.hasArea(args[0]):
                # switch on Area, set it to the shape name.
                self.exp_area = args[0]
                self.active = True

                # Initiate the loggers.
                self.flst.start()
                self.conflog.start()
                self.flst.writeline(flstvars)
                self.conflog.writeline(confvars)

                return True, f'{msgname} is set to {args[0]}'
            elif args[0][:2] == 'OF':
                # Switch off the area and reset the logger.
                self.active = False
                return True, f'{msgname} is switched OFF'
            elif args[0][:2] == 'ON':
                if not self.name:
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
            self.flst.start()
            self.conflog.start()
            self.flst.writeline(flstvars)
            self.conflog.writeline(confvars)

            return True, f'{msgname} is ON. Area name is: {self.exp_area}'
        else:
            return False, 'Incorrect arguments\n' + \
                   'AREA Shapename/OFF or\n Area lat,lon,lat,lon,[top,bottom]'

    def set_taxi(self, flag, alt=1500 * ft):
        """ Taxi ON/OFF to autodelete below a certain altitude if taxi is off """
        self.swtaxi = flag  # True =  taxi allowed, False = autodelete below swtaxialt
        self.swtaxialt = alt
