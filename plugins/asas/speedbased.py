"""
Urban environment speed-based CR algorithm.
Note: - Decelerating only
      - Climbing aircraft give way
      - Descending aircraft are ignored.

Created by Michiel Aarts - March 2021
"""
import numpy as np
from bluesky.traffic.asas import ConflictResolution
from bluesky import traf


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
        # Behind_ratio is the ratio of dtlookahead that the following aircraft
        # will use in its conflict resolution.
        self.behind_angle = np.deg2rad(10)
        self.behind_ratio = 0.9

        # Minimum speed (if allowed by A/C performance). Must be above 0 to prevent erroneous behaviour.
        self.min_speed = 1E-4

        # Initialize head-on conflict counter. These should not be present in urban airspace.
        self.num_head_on_conflicts = 0
        self.head_on_conflict_pairs = set()

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

    def create(self, n=1):
        super().create(n)
        self.is_leading[-n:] = True

    def reset(self):
        super().reset()
        self.num_head_on_conflicts = 0
        self.head_on_conflict_pairs = set()

    def resolve(self, conf, ownship, intruder):
        """ Resolve all current conflicts """
        # Initialize an array to store the resolution velocity vector for all A/C.
        dv = np.zeros((ownship.ntraf, 3))

        # Call speed_based function to resolve conflicts.
        for ((ac1, ac2), qdr, dist, tcpa, tLOS) in zip(conf.confpairs, conf.qdr, conf.dist, conf.tcpa, conf.tLOS):
            idx1 = ownship.id.index(ac1)
            idx2 = intruder.id.index(ac2)

            if 175. < (ownship.trk[idx1] - intruder.trk[idx2]) % 360. < 185.:
                # Head-on conflicts should not be present in an urban airspace, as these cannot be resolved.
                if (ac1, ac2) not in self.head_on_conflict_pairs \
                        and (ac2, ac1) not in self.head_on_conflict_pairs:
                    self.head_on_conflict_pairs.add((ac1, ac2))
                    self.num_head_on_conflicts += 1
                    print(f"WARNING: A head-on conflict found between {ac1} and {ac2}!")
                    print(f"         Total this run: {self.num_head_on_conflicts}")
                # Skip conflict resolution for this conflict.
                continue

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

        # Cap the velocity (and prevent reverse flying).
        # Drones that come to a complete stop result in erroneous headings / tracks / conflicts,
        # therefore: cap the minimum just above 0.
        min_speed = np.maximum(self.min_speed, ownship.perf.vmin)
        newgscapped = np.maximum(min_speed, np.minimum(ownship.perf.vmax, newgs))

        # Cap the vertical speed.
        vscapped = np.maximum(ownship.perf.vsmin, np.minimum(ownship.perf.vsmax, newvs))

        return newtrack, newgscapped, vscapped, ownship.selalt

    def speed_based(self, ownship, intruder, conf, qdr, dist, tcpa, tlos, idx1, idx2):
        """ Constrained environment speed-based resolution algorithm """
        # Convert bearing from degrees to radians.
        qdr = np.deg2rad(qdr)

        # Protected zones.
        s_h = conf.rpz * self.resofach
        s_v = conf.hpz * self.resofacv

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
        dcpa = drel + vrel * tcpa
        mag_dcpa = np.linalg.norm(dcpa)

        # Difference between both track angles.
        track_angle = np.deg2rad(abs(ownship.trk[idx1] - intruder.trk[idx2]))

        # Angle between cpa and ownship track.
        dcpa_angle = np.arctan2(dcpa[0], dcpa[1])
        cpa_angle = (np.deg2rad(ownship.trk[idx1]) - dcpa_angle) % (2 * np.pi)

        if track_angle < self.behind_angle or track_angle > np.pi * 2 - self.behind_angle:
            # In-airway conflict.
            distance_to_los = dist - s_h
            if distance_to_los < 0 and (abs(ownship.vs[idx1]) > 0 or abs(intruder.vs[idx2]) > 0):
                # Conflict with climbing / descending aircraft.
                if abs(ownship.vs[idx1]) > 1E-4:
                    if ownship.vs[idx1] < 0 or intruder.vs[idx2] < 0:
                        # TODO NOTE: this is a 2D version, for 3D effects with climbing / descending traffic
                        #  this needs to be updated!
                        # Aircraft already descending out of area, do nothing.
                        return v1 * 0, idx1
                    if intruder.vs[idx2] > 1E-4:
                        # If both are climbing, the lowest aircraft must resolve the conflict.
                        if ownship.alt[idx1] > intruder.alt[idx2]:
                            # Intruder must resolve, do nothing.
                            return v1 * 0, idx1
                        # Else: ownship must resolve, no change in algorithm.
                    # Ownship is climbing / descending, must resolve conflict.
                    # Decelerate such that both a/c remain in conflict, but keep approaching slowly.
                    # Approach such that the time to los becomes dtlookahead * behind_ratio.
                    # Due to this deceleration, the vertical conflict will result in a horizontal resolution.
                    self.is_leading[idx1] = False
                    vertical_distance_to_los = abs(intruder.alt[idx2] - ownship.alt[idx1]) - s_v
                    desired_tlos = conf.dtlookahead * self.behind_ratio
                    desired_vertical_vrel = vertical_distance_to_los / desired_tlos
                    resolution_vrel = intruder.vs[idx2] + desired_vertical_vrel
                    decelerate_factor = 1 - resolution_vrel / ownship.vs[idx1]

                    if decelerate_factor < 0:
                        # Acceleration advised, do nothing.
                        return v1 * 0, idx1
                    return -v1 * decelerate_factor, idx1
                else:
                    # Intruder must resolve conflict as it is climbing / descending.
                    # Do nothing.
                    return v1 * 0, idx1
            else:
                # Same-level, in-airway conflict.
                # Check if approaching from behind (bearing and speed should be in equal directions).
                qdr_trk_angle = (np.deg2rad(ownship.trk[idx1]) - qdr) % (2 * np.pi)
                if qdr_trk_angle < np.pi / 2 or qdr_trk_angle > 3 * np.pi / 2:
                    # Ownship approaching from behind, needs to decelerate to intruder speed.
                    self.is_leading[idx1] = False

                    # Decelerate such that both a/c remain in conflict, but keep approaching slowly.
                    # Approach such that the time to los becomes dtlookahead * behind_ratio.
                    desired_tlos = conf.dtlookahead * self.behind_ratio
                    desired_vrel = distance_to_los / desired_tlos
                    resolution_spd = mag_v2 + desired_vrel
                    decelerate_factor = 1 - resolution_spd / mag_v1

                    if decelerate_factor < 0:
                        # Acceleration advised, do nothing.
                        return v1 * 0, idx1
                    return -v1 * decelerate_factor, idx1
                else:
                    # Intruder must resolve conflict from behind.
                    # Do nothing.
                    return v1 * 0, idx1
        elif np.pi / 2 <= cpa_angle <= 3 * np.pi / 2:
            # Crossing conflict.
            # Intruder must resolve, do nothing.
            return v1 * 0, idx1
        elif cpa_angle < np.pi / 2 or cpa_angle > 3 * np.pi / 2:
            # Crossing conflict.
            # Ownship must resolve.
            self.is_leading[idx1] = False

            # Calculate cosine angle.
            cos_angle = np.arccos(np.dot(v2, vrel) / (mag_v2 * mag_vrel))

            # Determine time to incur for resolution.
            tr = (s_h - mag_dcpa) / (mag_v2 * np.sin(cos_angle))

            decelerate_factor = 1 - tcpa / (tr + tcpa)

            if decelerate_factor < 0:
                # Acceleration advised, do nothing.
                return v1 * 0, idx1
            return -v1 * decelerate_factor, idx1
        else:
            raise ValueError(f'Something went wrong in speed_based(),\n' +
                             f'cpa_angle={cpa_angle},\n' +
                             f'trk_angle={track_angle}')

    def resumenav(self, conf, ownship, intruder):
        """
            Decide for each aircraft in the conflict list whether the ASAS
            should be followed or not, based on if the aircraft pairs passed
            their CPA.
            Note: only difference with resolution.py/resumenav is the reset
             of the is_leading flag.
        """
        # Add new conflicts to resopairs and confpairs_all and new losses to lospairs_all.
        self.resopairs.update(conf.confpairs)

        # Conflict pairs to be deleted.
        delpairs = set()
        changeactive = dict()

        # Look at all conflicts, also the ones that are solved but CPA is yet to come.
        for conflict in self.resopairs:
            idx1, idx2 = traf.id2idx(conflict)
            # If the ownship aircraft is deleted remove its conflict from the list.
            if idx1 < 0:
                delpairs.add(conflict)
                continue

            if idx2 >= 0:
                # Distance vector using flat earth approximation.
                re = 6371000.
                dist = re * np.array([np.radians(intruder.lon[idx2] - ownship.lon[idx1]) *
                                      np.cos(0.5 * np.radians(intruder.lat[idx2] +
                                                              ownship.lat[idx1])),
                                      np.radians(intruder.lat[idx2] - ownship.lat[idx1])])

                # Relative velocity vector.
                vrel = np.array([intruder.gseast[idx2] - ownship.gseast[idx1],
                                 intruder.gsnorth[idx2] - ownship.gsnorth[idx1]])

                # Check if conflict is past CPA.
                past_cpa = np.dot(dist, vrel) > 0.0

                # hor_los:
                # Aircraft should continue to resolve until there is no horizontal
                # LOS. This is particularly relevant when vertical resolutions
                # are used.
                hdist = np.linalg.norm(dist)
                hor_los = hdist < conf.rpz

                # Bouncing conflicts:
                # If two aircraft are getting in and out of conflict continously,
                # then they it is a bouncing conflict. ASAS should stay active until
                # the bouncing stops.
                is_bouncing = \
                    abs(ownship.trk[idx1] - intruder.trk[idx2]) < 30.0 and \
                    hdist < conf.rpz * self.resofach

            # Start recovery for ownship if intruder is deleted, or if past CPA
            # and not in horizontal LOS or a bouncing conflict.
            if idx2 >= 0 and (not past_cpa or hor_los or is_bouncing):
                # Enable ASAS for this aircraft
                changeactive[idx1] = True
            else:
                # Switch ASAS off for ownship if there are no other conflicts.
                # that this aircraft is involved in.
                changeactive[idx1] = changeactive.get(idx1, False)
                # If conflict is solved, remove it from the resopairs list.
                delpairs.add(conflict)
                # Reset is_leading flag.
                self.is_leading[idx1] = True

        for idx, active in changeactive.items():
            # Loop a second time: this is to avoid that ASAS resolution is
            # turned off for an aircraft that is involved simultaneously in
            # multiple conflicts, where the first, but not all conflicts are
            # resolved.
            self.active[idx] = active
            if not active:
                # Waypoint recovery after conflict: Find the next active waypoint
                # and send the aircraft to that waypoint.
                iwpid = traf.ap.route[idx].findact(idx)
                if iwpid != -1:  # To avoid problems if there are no waypoints.
                    traf.ap.route[idx].direct(
                        idx, traf.ap.route[idx].wpname[iwpid])

        # Remove pairs from the list that are past CPA or have deleted aircraft.
        self.resopairs -= delpairs
