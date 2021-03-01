""" Urban environment speed-based CR algorithm """
import numpy as np
from bluesky.traffic.asas import ConflictResolution


def init_plugin():
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name': 'speedbased',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type': 'sim',
    }

    return config


class SpeedBased(ConflictResolution):
    def __init__(self):
        super().__init__()
        # Set behind_angle and behind_ratio to handle in-airway speed-based conflicts.
        # Behind_angle is the maximum track angle difference between two aircraft
        # for them to be in an in-airway conflict.
        # Behind_ratio is the ratio of dtlookahead that the following aircraft will use in its conflict resolution.
        self.behind_angle = np.deg2rad(10)
        self.behind_ratio = 0.9

        with self.settrafarrays():
            # Initialize is_leading traffic array.
            self.is_leading = np.array([], dtype=bool)

    @property
    def hdgactive(self):
        """ Heading inactive """
        return np.zeros(self.active.shape, dtype=bool)

    @property
    def tasactive(self):
        """ Speed-based only, all aircraft in conflict that are not leading need tasactive True """
        return ~self.is_leading & self.active

    @property
    def altactive(self):
        """ Altitude inactive """
        return np.zeros(self.active.shape, dtype=bool)

    @property
    def vsactive(self):
        """ Vertical speed inactive """
        return np.zeros(self.active.shape, dtype=bool)

    def resolve(self, conf, ownship, intruder):
        """ Resolve all current conflicts """
        # Initialize an array to store the resolution velocity vector for all A/C.
        dv = np.zeros((ownship.ntraf, 3))

        # Reset all is_leading flags.
        self.is_leading[:] = True

        # Call speed_based function to resolve conflicts.
        for ((ac1, ac2), qdr, dist, tcpa, tLOS) in zip(conf.confpairs, conf.qdr, conf.dist, conf.tcpa, conf.tLOS):
            idx1 = ownship.id.index(ac1)
            idx2 = intruder.id.index(ac2)

            # If A/C indexes are found, then apply speed_based on this conflict pair.
            # Because ADSB is ON, this is done for each aircraft separately.
            # However, only one aircraft will resolve the conflict.
            if idx1 > -1 and idx2 > -1:
                dv_v, idx = self.speed_based(ownship, intruder, conf, qdr, dist, tcpa, tLOS, idx1, idx2)

                current_dv_magnitude = np.linalg.norm(dv[idx])
                new_dv_magnitude = np.linalg.norm(dv_v)

                if new_dv_magnitude > current_dv_magnitude:
                    dv[idx] += dv_v

        # Resolution vector for all aircraft, cartesian coordinates.
        dv = np.transpose(dv)

        # The old speed vector, cartesian coordinates.
        v = np.array([ownship.gseast, ownship.gsnorth, ownship.vs])

        # The new speed vector, cartesian coordinates.
        newv = v + dv

        # Limit resolution direction if required.
        newtrack = ownship.trk
        newgs = np.sqrt(newv[0, :] ** 2 + newv[1, :] ** 2)
        newvs = ownship.vs

        # Cap the velocity.
        newgscapped = np.maximum(ownship.perf.vmin, np.minimum(ownship.perf.vmax, newgs))

        # Cap the vertical speed.
        vscapped = np.maximum(ownship.perf.vsmin, np.minimum(ownship.perf.vsmax, newvs))

        return newtrack, newgscapped, vscapped, ownship.selalt

    def speed_based(self, ownship, intruder, conf, qdr, dist, tcpa, tLOS, idx1, idx2):
        """ Constrained environment speed-based resolution algorithm """
        # Preliminary calculations.

        if tcpa < 0:
            # Closest point already passed. Do nothing.
            return np.zeros(3), idx1

        if ownship != intruder:
            raise ValueError('ownship != intruder ??\n', ownship.tas, intruder.tas)

        # Convert bearing from degrees to radians.
        qdr = np.deg2rad(qdr)

        # Relative position vector between id1 and id2.
        drel = np.array([np.sin(qdr) * dist,
                         np.cos(qdr) * dist,
                         intruder.alt[idx2] - ownship.alt[idx1]])

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

        # Difference between both track angles.
        track_angle = np.deg2rad(abs(ownship.trk[idx1] - intruder.trk[idx2]))

        # Angle between cpa and ownship track.
        dcpa_angle = np.arctan2(dcpa[0], dcpa[1])
        cpa_angle = (np.deg2rad(ownship.trk[idx1]) - dcpa_angle) % (2 * np.pi)

        # Check if flying in-airway.
        if track_angle < self.behind_angle:
            # Check if approaching from behind (bearing and speed should be in equal directions).
            qdr_trk_angle = (np.deg2rad(ownship.trk[idx1]) - qdr) % (2 * np.pi)
            if qdr_trk_angle < np.pi / 2 or qdr_trk_angle > 3 * np.pi / 2:
                # Ownship approaching from behind, needs to decelerate to intruder speed.
                self.is_leading[idx1] = False

                # Decelerate such that both a/c remain in conflict, but keep approaching slowly.
                # Approach such that the time_to_conflict becomes dtlookahead * behind_ratio.
                distance_to_conflict = dist - conf.rpz
                time_to_conflict = conf.dtlookahead * self.behind_ratio
                vrel_to_conflict = distance_to_conflict / time_to_conflict
                resolution_spd = mag_v2 + vrel_to_conflict
                decelerate_factor = 1 - resolution_spd / mag_v1

                if resolution_spd > ownship.ap.tas[idx1]:
                    # Acceleration advised, do nothing.
                    return v1 * 0, idx1
                return -v1 * decelerate_factor, idx1
            else:
                # Intruder must resolve conflict from behind.
                # Do nothing.
                return v1 * 0., idx1
        elif np.pi / 2 <= cpa_angle <= 3 * np.pi / 2:
            # Intruder must resolve crossing conflict.
            # Do nothing.
            return v1 * 0., idx1
        elif cpa_angle < np.pi / 2 or cpa_angle > 3 * np.pi / 2:
            # Ownship must resolve crossing conflict.
            self.is_leading[idx1] = False

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
