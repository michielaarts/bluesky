""" Urban environment grid """
from random import randint
import numpy as np
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import navdb, core, stack, traf, settings, sim, scr, tools
from bluesky.core import Entity
from bluesky.traffic.asas import ConflictResolution, ConflictDetection
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

    ospeed = OriginalSpeed()

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
    stack.stack('CRE UAV003 M600 CG0202 180 100 29')
    stack.stack('HOLD')
    stack.stack('ADDWPT UAV001 CG0004 100 30')
    stack.stack('ADDWPT UAV001 CG0404 100 30')
    stack.stack('UAV001 DEST CG0409')
    stack.stack('ADDWPT UAV003 CG0002 100 29')
    stack.stack('ADDWPT UAV003 CG0004 100 29')
    stack.stack('POS UAV001')
    stack.stack('PAN UAV001')
    stack.stack('ZOOM 20')
    stack.stack('TRAIL ON')

    # Use entity class to prevent errors
    # example = Example()

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


class OriginalSpeed(Entity):
    def __init__(self):
        super().__init__()
        with self.settrafarrays():
            self.original_tas = np.array([])

    def create(self, n=1):
        super().create(n)
        self.original_tas[-n:] = traf.tas[-n:]


class SpeedBased(ConflictResolution):
    def __init__(self):
        super().__init__()
        self.behind_angle = np.deg2rad(10)

    @property
    def hdgactive(self):
        return False * super().hdgactive

    @property
    def tasactive(self):
        return True * super().tasactive

    @property
    def altactive(self):
        return False * super().altactive

    @property
    def vsactive(self):
        return False * super().vsactive

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
                    dv[idx] += dv_v

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

        if tcpa < 0:
            # Closest point already passed. Do nothing.
            return np.zeros(3), idx1

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

        # Angle between both tracks
        track_angle = np.deg2rad(abs(ownship.trk[idx1] - intruder.trk[idx2]))

        # Angle between cpa and ownship track.
        dcpa_angle = np.arctan2(dcpa[0], dcpa[1])
        cpa_angle = (np.deg2rad(ownship.trk[idx1]) - dcpa_angle) % (2 * np.pi)

        # Check if flying in same direction
        if track_angle < self.behind_angle:
            # Check if coming from behind (bearing and speed should be in opposite directions
            qdr_trk_angle = abs(np.deg2rad(ownship.trk[idx1]) - qdr)
            if qdr_trk_angle < np.pi / 2 or qdr_trk_angle > 3 * np.pi / 2:
                # Ownship coming from behind, needs to decelerate to intruder speed.
                # Within distance drel - 2S_h, as approaching still beneficial.

                # Time to LoS
                t_s_h = s_h / mag_v2
                time_to_los = tcpa - t_s_h
                if time_to_los < 0:
                    raise ValueError('Loss of separation!')

                # a * 0 + b = v1
                # a * time_to_los + b = v2
                # scale_factor = - mag_vrel / time_to_los
                # decelerate_dv = scale_factor * settings.asas_dt
                # decelerate_factor = decelerate_dv / mag_v1

                original_detection_dist = conf.dtlookahead * ownship.ap.tas[idx1]

                # a * original_detection_dist + b = ownship.ap.tas
                # a * s_h + b = v2
                scale_factor = (ownship.ap.tas[idx1] - mag_v2) / (original_detection_dist - s_h)
                correction_factor = mag_v2 - scale_factor * s_h
                resolution_spd = scale_factor * dist + correction_factor
                decelerate_factor = 1 - resolution_spd / mag_v1

                ownship_acid = ownship.id[idx1]
                intruder_acid = intruder.id[idx2]
                # Decelerate factor.
                if resolution_spd > ownship.ap.tas[idx1]:
                    raise ValueError(f'Something went wrong in speed_based(), accelerating {ownship.id[idx1]}!\n' +
                                     f'resolution_spd={resolution_spd}')
                return -v1 * decelerate_factor, idx1
            else:
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

            # Determine time to incur for resolution.
            tr = (s_h - mag_dcpa) / (mag_v2 * np.sin(cos_angle))

            vreso = v1 * tcpa / (tr + tcpa)
            return vreso - v1, idx1
        else:
            raise ValueError(f'Something went wrong in speed_based(),\n' +
                             f'cpa_angle={cpa_angle},\n' +
                             f'trk_angle={track_angle}')
