""" Urban environment grid """
from random import randint
import numpy as np
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import navdb, core, stack, traf, settings, sim, scr, tools
from bluesky.core import Entity
from bluesky.traffic.asas import ConflictResolution
from bluesky.tools.aero import nm


### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    """ Plugin initialisation function. """
    # Instantiate our example entity
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
    # stack.stack('PRIORULES ON FF2')
    # stack.stack('RMETHH SPD')
    stack.stack(f'ZONER {S_h}')
    stack.stack(f'DTLOOK {t_l}')

    # MVP Speed based accelerates instead of decelerating..


    # Create dummy aircrafts
    stack.stack('CRE UAV001 M600 CG0002 90 100 15')
    stack.stack('CRE UAV002 M600 CG0000 90 100 30')
    stack.stack('HOLD')
    stack.stack('ADDWPT UAV001 CG0004')
    stack.stack('ADDWPT UAV001 CG0404')
    stack.stack('UAV001 DEST CG0409')
    stack.stack('POS UAV001')
    stack.stack('PAN UAV001')
    stack.stack('ZOOM 9')
    stack.stack('TRAIL ON')



    # Use entity class to prevent errors
    example = Example()

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


class SpeedBased(ConflictResolution):
    def __init__(self):
        super().__init__()

    def setprio(self, flag=None, priocode=''):
        """Set the prio switch and the type of prio """
        if flag is None:
            return True, "PRIORULES [ON/OFF] [PRIOCODE]" + \
                   "\nAvailable priority codes: " + \
                   "\n     FF1:  Free Flight Primary (No Prio) " + \
                   "\n     FF2:  Free Flight Secondary (Cruising has priority)" + \
                   "\n     FF3:  Free Flight Tertiary (Climbing/descending has priority)" + \
                   "\n     LAY1: Layers Primary (Cruising has priority + horizontal resolutions)" + \
                   "\n     LAY2: Layers Secondary (Climbing/descending has priority + horizontal resolutions)" + \
                   "\nPriority is currently " + ("ON" if self.swprio else "OFF") + \
                   "\nPriority code is currently: " + \
                   str(self.priocode)
        options = ["FF1", "FF2", "FF3", "LAY1", "LAY2"]
        if priocode not in options:
            return False, "Priority code Not Understood. Available Options: " + str(options)
        return super().setprio(flag, priocode)

    def applyprio(self, dv_mvp, dv1, dv2, vs1, vs2):
        """ Apply the desired priority setting to the resolution """

        # Primary Free Flight prio rules (no priority)
        if self.priocode == 'FF1':
            # since cooperative, the vertical resolution component can be halved, and then dv_mvp can be added
            dv_mvp[2] = dv_mvp[2] / 2.0
            dv1 = dv1 - dv_mvp
            dv2 = dv2 + dv_mvp

        # Secondary Free Flight (Cruising aircraft has priority, combined resolutions)
        if self.priocode == 'FF2':
            # since cooperative, the vertical resolution component can be halved, and then dv_mvp can be added
            dv_mvp[2] = dv_mvp[2] / 2.0
            # If aircraft 1 is cruising, and aircraft 2 is climbing/descending -> aircraft 2 solves conflict
            if abs(vs1) < 0.1 and abs(vs2) > 0.1:
                dv2 = dv2 + dv_mvp
            # If aircraft 2 is cruising, and aircraft 1 is climbing -> aircraft 1 solves conflict
            elif abs(vs2) < 0.1 and abs(vs1) > 0.1:
                dv1 = dv1 - dv_mvp
            else:  # both are climbing/descending/cruising -> both aircraft solves the conflict
                dv1 = dv1 - dv_mvp
                dv2 = dv2 + dv_mvp

        # Tertiary Free Flight (Climbing/descending aircraft have priority and crusing solves with horizontal
        # resolutions)
        elif self.priocode == 'FF3':
            # If aircraft 1 is cruising, and aircraft 2 is climbing/descending -> aircraft 1 solves conflict
            # horizontally
            if abs(vs1) < 0.1 and abs(vs2) > 0.1:
                dv_mvp[2] = 0.0
                dv1 = dv1 - dv_mvp
            # If aircraft 2 is cruising, and aircraft 1 is climbing -> aircraft 2 solves conflict horizontally
            elif abs(vs2) < 0.1 and abs(vs1) > 0.1:
                dv_mvp[2] = 0.0
                dv2 = dv2 + dv_mvp
            else:  # both are climbing/descending/cruising -> both aircraft solves the conflict, combined
                dv_mvp[2] = dv_mvp[2] / 2.0
                dv1 = dv1 - dv_mvp
                dv2 = dv2 + dv_mvp

        # Primary Layers (Cruising aircraft has priority and clmibing/descending solves. All conflicts solved horizontally)
        elif self.priocode == 'LAY1':
            dv_mvp[2] = 0.0
            # If aircraft 1 is cruising, and aircraft 2 is climbing/descending -> aircraft 2 solves conflict horizontally
            if abs(vs1) < 0.1 and abs(vs2) > 0.1:
                dv2 = dv2 + dv_mvp
            # If aircraft 2 is cruising, and aircraft 1 is climbing -> aircraft 1 solves conflict horizontally
            elif abs(vs2) < 0.1 and abs(vs1) > 0.1:
                dv1 = dv1 - dv_mvp
            else:  # both are climbing/descending/cruising -> both aircraft solves the conflict horizontally
                dv1 = dv1 - dv_mvp
                dv2 = dv2 + dv_mvp

        # Secondary Layers (Climbing/descending aircraft has priority and cruising solves. All conflicts solved horizontally)
        elif self.priocode == 'LAY2':
            dv_mvp[2] = 0.0
            # If aircraft 1 is cruising, and aircraft 2 is climbing/descending -> aircraft 1 solves conflict horizontally
            if abs(vs1) < 0.1 and abs(vs2) > 0.1:
                dv1 = dv1 - dv_mvp
            # If aircraft 2 is cruising, and aircraft 1 is climbing -> aircraft 2 solves conflict horizontally
            elif abs(vs2) < 0.1 and abs(vs1) > 0.1:
                dv2 = dv2 + dv_mvp
            else:  # both are climbing/descending/cruising -> both aircraft solves the conflic horizontally
                dv1 = dv1 - dv_mvp
                dv2 = dv2 + dv_mvp

        return dv1, dv2

    def resolve(self, conf, ownship, intruder):
        """ Resolve all current conflicts """
        # Initialize an array to store the resolution velocity vector for all A/C
        dv = np.zeros((ownship.ntraf, 3))

        # Call speed_based function to resolve conflicts
        for ((ac1, ac2), qdr, dist, tcpa, tLOS) in zip(conf.confpairs, conf.qdr, conf.dist, conf.tcpa, conf.tLOS):
            idx1 = ownship.id.index(ac1)
            idx2 = intruder.id.index(ac2)

            # If A/C indexes are found, then apply speed_based on this conflict pair
            # Because ADSB is ON, this is done for each aircraft separately
            # However, only one aircraft will resolve the conflict
            if idx1 > -1 and idx2 > -1:
                dv_v, idx = self.speed_based(ownship, intruder, conf, qdr, dist, tcpa, tLOS, idx1, idx2)

                current_dv_magnitude = np.linalg.norm(dv[idx])
                new_dv_magnitude = np.linalg.norm(dv_v)

                if new_dv_magnitude > current_dv_magnitude:
                    dv[idx] = dv_v

        # Resolution vector for all aircraft, cartesian coordinates
        dv = np.transpose(dv)

        # The old speed vector, cartesian coordinates
        v = np.array([ownship.gseast, ownship.gsnorth, ownship.vs])

        # The new speed vector, cartesian coordinates
        newv = v + dv

        # Limit resolution direction if required
        newtrack = ownship.trk
        newgs = np.sqrt(newv[0, :] ** 2 + newv[1, :] ** 2)
        newvs = ownship.vs

        # Cap the velocity
        newgscapped = np.maximum(ownship.perf.vmin, np.minimum(ownship.perf.vmax, newgs))

        # Cap the vertical speed
        vscapped = np.maximum(ownship.perf.vsmin, np.minimum(ownship.perf.vsmax, newvs))

        return newtrack, newgscapped, vscapped, ownship.selalt


    def speed_based(self, ownship, intruder, conf, qdr, dist, tcpa, tLOS, idx1, idx2):
        """Constrained speed-based resolution algorithm"""
        # Preliminary calculations

        if ownship != intruder:
            raise ValueError('ownship != intruder ??\n', ownship.tas, intruder.tas)

        # Convert qdr from degrees to radians.
        qdr = np.deg2rad(qdr)  # Bearing.

        # Relative position vector between id1 and id2.
        drel = np.array([np.sin(qdr) * dist,
                         np.cos(qdr) * dist,
                         intruder.alt[idx2] - ownship.alt[idx1]])
        mag_drel = np.linalg.norm(drel)

        # Write velocities as vectors and find relative velocity vector.
        v1 = np.array([ownship.gseast[idx1], ownship.gsnorth[idx1], ownship.vs[idx1]])
        v2 = np.array([intruder.gseast[idx2], intruder.gsnorth[idx2], intruder.vs[idx2]])
        mag_v1 = ownship.tas[idx1]
        mag_v2 = intruder.tas[idx2]
        vrel = np.array(v2 - v1)
        mag_vrel = np.linalg.norm(vrel)

        # Find horizontal distance at the tcpa (min horizontal distance).
        s_h = conf.rpz * self.resofach
        dcpa = drel + vrel * tcpa
        mag_dcpa = np.linalg.norm(dcpa)

        # Angle between bearing and own speed.
        approach_angle = (np.deg2rad(ownship.trk[idx1]) - qdr) % (2 * np.pi)

        # Angle between cpa and ownship track.
        dcpa_angle = np.arctan2(dcpa[0], dcpa[1])
        cpa_angle = (np.deg2rad(ownship.trk[idx1]) - dcpa_angle) % (2 * np.pi)

        # Original detection distance (to use as factor).
        mag_original_vrel = abs(intruder.ap.tas[idx2] - ownship.ap.tas[idx1])
        original_detection_distance = mag_original_vrel * conf.dtlookahead + s_h

        # Check if flying in same direction
        behind_angle = np.deg2rad(10)
        if approach_angle < behind_angle or approach_angle > 2 * np.pi - behind_angle:
            # Ownship coming from behind, needs to decelerate to intruder speed.
            # Within distance drel - 2S_h, as approaching still beneficial.

            # Decelerate factor.
            # original_detection_distance * a + b = 0.
            # 2*s_h * a + b = 1.
            scale_factor = -1 / (original_detection_distance - s_h)
            correction = - original_detection_distance * scale_factor
            decelerate_factor = scale_factor * dist + correction
            if decelerate_factor < 0:
                raise ValueError(f'Something went wrong in speed_based(), decelerate_factor={decelerate_factor}')
            return vrel * decelerate_factor, idx1
        elif np.pi - behind_angle < approach_angle < np.pi + behind_angle:
            # Intruder must resolve conflict from behind.
            # Do nothing.
            return v1 * 0., idx1
        elif np.pi / 2 < cpa_angle < 3 * np.pi / 2:
            # Intruder must resolve crossing conflict.
            # Do nothing.
            return v1 * 0., idx1
        elif cpa_angle < np.pi / 2 or cpa_angle > 3 * np.pi / 2:
            # Ownship must resolve crossing conflict.
            # Calculate cosine angle.
            cos_angle = np.arccos(np.dot(v2, vrel) / (mag_v2 * mag_vrel))

            # Determine time to incur for resolution
            tr = (2 * s_h - mag_dcpa) / (mag_v2 * np.sin(cos_angle))

            # Simple version, slow down until resolved
            return v1 * 0., idx1
        else:
            raise ValueError(f'Something went wrong in speed_based(),\n' +
                             f'cpa_angle={cpa_angle},\n' +
                             f'app_angle={approach_angle}')





### Entities in BlueSky are objects that are created only once (called singleton)
### which implement some traffic or other simulation functionality.
### To define an entity that ADDS functionality to BlueSky, create a class that
### inherits from bluesky.core.Entity.
### To replace existing functionality in BlueSky, inherit from the class that
### provides the original implementation (see for example the asas/eby plugin).
class Example(Entity):
    """ Example new entity object for BlueSky. """

    def __init__(self):
        super().__init__()
        # All classes deriving from Entity can register lists and numpy arrays
        # that hold per-aircraft data. This way, their size is automatically
        # updated when aircraft are created or deleted in the simulation.
        with self.settrafarrays():
            self.npassengers = np.array([])

    def create(self, n=1):
        """ This function gets called automatically when new aircraft are created. """
        # Don't forget to call the base class create when you reimplement this function!
        super().create(n)
        # After base creation we can change the values in our own states for the new aircraft
        self.npassengers[-n:] = [randint(0, 150) for _ in range(n)]

    # Functions that need to be called periodically can be indicated to BlueSky
    # with the timed_function decorator
    @core.timed_function(name='example', dt=5)
    def update(self):
        """ Periodic update function for our example entity. """
        return None
        # stack.stack('ECHO Example update: creating a random aircraft')
        # stack.stack('MCRE 1')

    # You can create new stack commands with the stack.command decorator.
    # By default, the stack command name is set to the function name.
    # The default argument type is a case-sensitive word. You can indicate different
    # types using argument annotations. This is done in the below function:
    # - The acid argument is a BlueSky-specific argument with type 'acid'.
    #       This converts callsign to the corresponding index in the traffic arrays.
    # - The count argument is a regular int.
    @stack.command
    def passengers(self, acid: 'acid', count: int = -1):
        """ Set the number of passengers on aircraft 'acid' to 'count'. """
        if count < 0:
            return True, f'Aircraft {traf.id[acid]} currently has {self.npassengers[acid]} passengers on board.'

        self.npassengers[acid] = count
        return True, f'The number of passengers on board {traf.id[acid]} is set to {count}.'
