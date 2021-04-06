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
from bluesky.tools import geo
from bluesky.tools.aero import nm

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
            self.is_leading = np.array([], dtype=dict)

    @property
    def hdgactive(self):
        """ Heading inactive """
        return np.zeros(self.active.shape, dtype=bool)

    @property
    def tasactive(self):
        """ Speed-based only, all aircraft in conflict that are not leading need tasactive True """
        swleading = np.ones(self.active.shape, dtype=bool)
        for idx in range(len(swleading)):
            if self.is_leading[idx].values():
                swleading[idx] = all(self.is_leading[idx].values())
        return ~swleading & self.active

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
        self.is_leading[-n:] = {}

    def reset(self):
        super().reset()
        self.num_head_on_conflicts = 0
        self.head_on_conflict_pairs = set()

    def resolve(self, conf, ownship, intruder):
        """ Resolve all current conflicts """
        # Initialize an array to store the resolution velocity vector for all A/C.
        dv = np.zeros((ownship.ntraf, 3))
        current_decelerate_factor = np.ones(ownship.ntraf) * -1E4

        # The old speed vector, cartesian coordinates.
        v = np.array([ownship.gseast, ownship.gsnorth, ownship.vs])

        # Add new conflicts to resopairs.
        self.resopairs.update(conf.confpairs)

        # Detect conflict properties of all resopairs.
        conf_reso = self.detect(self.resopairs, ownship, intruder)

        # Call speed_based function to resolve all resopairs.
        for pair in self.resopairs:
            idx1 = conf_reso[pair]['idx1']
            idx2 = conf_reso[pair]['idx2']
            qdr = conf_reso[pair]['qdr']
            dist = conf_reso[pair]['dist']
            tcpa = conf_reso[pair]['tcpa']
            tlos = conf_reso[pair]['tlos']

            if 175. < (ownship.trk[idx1] - intruder.trk[idx2]) % 360. < 185.:
                # Head-on conflicts should not be present in an urban airspace, as these cannot be resolved.
                if pair not in self.head_on_conflict_pairs \
                        and pair not in self.head_on_conflict_pairs:
                    self.head_on_conflict_pairs.add(pair)
                    self.num_head_on_conflicts += 1
                    if self.num_head_on_conflicts == 1:
                        print(f"WARNING: A head-on conflict found between {pair}!")
                    elif self.num_head_on_conflicts % 10 == 0:
                        print(f"WARNING: Total number of head-on conflicts is now {self.num_head_on_conflicts}")
                # Skip conflict resolution for this conflict.
                continue

            # Call speed_based for each resopair.
            # Because ADSB is ON, this is done for each aircraft separately.
            # However, only one aircraft will resolve the conflict.
            dv_v, idx, decelerate_factor = self.speed_based(ownship, intruder, conf, qdr, dist, tcpa, tlos, idx1, idx2)

            # Select strongest deceleration.
            if not np.isnan(decelerate_factor) and decelerate_factor > current_decelerate_factor[idx]:
                dv[idx] = dv_v
                current_decelerate_factor[idx] = decelerate_factor

        # Resolution vector for all aircraft, cartesian coordinates.
        dv = np.transpose(dv)

        # The new speed vector, cartesian coordinates.
        newv = v + dv

        # Limit resolution direction if required.
        newtrack = ownship.trk
        newgs = np.sqrt(newv[0, :] ** 2 + newv[1, :] ** 2)
        newvs = ownship.vs

        # Cap the velocity at AP speed (and prevent reverse flying).
        # Drones that come to a complete stop result in erroneous headings / tracks / conflicts,
        # therefore: cap the minimum just above 0.
        min_speed = np.maximum(self.min_speed, ownship.perf.vmin)
        newgscapped = np.maximum(min_speed, np.minimum(ownship.ap.tas, newgs))

        # Cap the vertical speed.
        vscapped = np.maximum(ownship.perf.vsmin, np.minimum(ownship.perf.vsmax, newvs))

        return newtrack, newgscapped, vscapped, ownship.selalt

    def speed_based(self, ownship, intruder, conf, qdr, dist, tcpa, tlos, idx1, idx2):
        """ Constrained environment speed-based resolution algorithm """
        # Get second aircraft id for is_leading flag
        ac2 = intruder.id[idx2]

        # Ignore conflicts with descending aircraft.
        if ownship.vs[idx1] < -1E-4 or intruder.vs[idx2] < -1E-4:
            # NOTE: this is a 2D version, for 3D effects with climbing / descending traffic
            #  this needs to be updated!
            self.is_leading[idx1][ac2] = True
            return np.zeros(3), idx1, np.nan

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

        # Autopilot angles ----
        # Write velocities as vectors and find relative velocity vector.
        owntrkrad = np.deg2rad(ownship.trk[idx1])
        inttrkrad = np.deg2rad(intruder.trk[idx2])
        v1_ap = ownship.tas[idx1] * np.array([np.sin(owntrkrad), np.cos(owntrkrad)])
        v2_ap = intruder.tas[idx1] * np.array([np.sin(inttrkrad), np.cos(inttrkrad)])
        vrel_ap = np.array(v2_ap - v1_ap)
        mag_vrel_ap = np.linalg.norm(vrel_ap)

        du_ap = v2_ap[0] - v1_ap[0]
        dv_ap = v2_ap[1] - v1_ap[1]

        # Find horizontal distance at the tcpa (min horizontal distance).
        tcpa_ap = -(du_ap * drel[0] + dv_ap * drel[1]) / np.power(mag_vrel_ap, 2)
        dcpa_ap = drel[:2] + vrel_ap * tcpa_ap

        # Angle between cpa and ownship track.
        dcpa_angle_ap = np.arctan2(dcpa_ap[0], dcpa_ap[1])
        cpa_angle_ap = (owntrkrad - dcpa_angle_ap) % (2 * np.pi)

        if track_angle < self.behind_angle or track_angle > np.pi * 2 - self.behind_angle:
            # In-airway conflict.
            qdr_trk_angle = (np.deg2rad(ownship.trk[idx1]) - qdr) % (2 * np.pi)

            # Check which vehicle is in front.
            if qdr_trk_angle < np.pi / 2 or qdr_trk_angle > 3 * np.pi / 2:
                # Ownship approaching from behind, must resolve conflict.
                # Decelerate such that both a/c remain in conflict, but keep approaching slowly.
                # Approach such that the time to los becomes dtlookahead * behind_ratio.
                # Due to this deceleration, the vertical conflict will result in a horizontal resolution.
                self.is_leading[idx1][ac2] = False
                desired_tlos = conf.dtlookahead * self.behind_ratio

                horizontal_distance_to_los = dist - s_h * np.sqrt(2)
                horizontal_desired_vrel = horizontal_distance_to_los / desired_tlos
                horizontal_resolution_spd = mag_v2 + horizontal_desired_vrel
                horizontal_decelerate_factor = 1 - horizontal_resolution_spd / mag_v1

                vertical_distance_to_los = abs(intruder.alt[idx2] - ownship.alt[idx1]) - s_v
                vertical_desired_vrel = vertical_distance_to_los / desired_tlos
                vertical_resolution_vrel = intruder.vs[idx2] + vertical_desired_vrel
                vertical_decelerate_factor = 1 - vertical_resolution_vrel / ownship.vs[idx1]

                if vertical_distance_to_los < 0:
                    # Already at same level, use horizontal decelerate factor.
                    decelerate_factor = horizontal_decelerate_factor
                else:
                    decelerate_factor = max(horizontal_decelerate_factor, vertical_decelerate_factor)

                if decelerate_factor < 0:
                    # Acceleration advised, do not accelerate into rear.
                    return v1 * 0, idx1, 0
                if decelerate_factor > 1:
                    decelerate_factor = 1
                return -v1 * decelerate_factor, idx1, decelerate_factor
            else:
                # Intruder must resolve, do nothing.
                self.is_leading[idx1][ac2] = True
                return v1 * 0, idx1, np.nan
        # Else: crossing conflicts.
        elif np.pi / 2 <= cpa_angle <= 3 * np.pi / 2:
            if not (np.pi / 2 <= cpa_angle_ap <= 3 * np.pi / 2):
                # At autopilot speed, ownship would not cross first.
                # Ownship must resolve.
                self.is_leading[idx1][ac2] = False

                # Calculate cosine angle.
                cos_angle = np.arccos(np.dot(v2, vrel) / (mag_v2 * mag_vrel))

                # Determine time to incur for resolution.
                tr = (s_h + mag_dcpa) / (mag_v2 * np.sin(cos_angle))

                if tr > 0. and tlos < 1000.:
                    # Incur tr.
                    decelerate_factor = 1 - tlos / (tr + tlos)
                else:
                    # Acceleration possible.
                    # Gain tr.
                    if abs(tr) < tcpa:
                        decelerate_factor = 1 - tcpa / (tr + tcpa)
                    else:
                        # Accelerate back to AP speed.
                        decelerate_factor = -1

                if decelerate_factor > 1:
                    decelerate_factor = 1
                return -v1 * decelerate_factor, idx1, decelerate_factor
            else:
                # Ownship crossing first, intruder must resolve, do nothing.
                self.is_leading[idx1][ac2] = True
                return v1 * 0, idx1, np.nan
        elif cpa_angle < np.pi / 2 or cpa_angle > 3 * np.pi / 2:
            if not (cpa_angle_ap < np.pi / 2 or cpa_angle_ap > 3 * np.pi / 2):
                # At autopilot speed, ownship would cross first.
                # Intruder must resolve, do nothing.
                self.is_leading[idx1][ac2] = True
                return v1 * 0, idx1, np.nan
            else:
                # Intruder crossing first, ownship must resolve.
                self.is_leading[idx1][ac2] = False

                # Calculate cosine angle.
                cos_angle = np.arccos(np.dot(v2, vrel) / (mag_v2 * mag_vrel))

                # Determine time to incur for resolution.
                tr = (s_h - mag_dcpa) / (mag_v2 * np.sin(cos_angle))

                if tr > 0. and tlos < 1000.:
                    # Incur tr.
                    decelerate_factor = 1 - tlos / (tr + tlos)
                else:
                    # Acceleration possible.
                    # Gain tr.
                    if abs(tr) < tcpa:
                        decelerate_factor = 1 - tcpa / (tr + tcpa)
                    else:
                        # Accelerate back to AP speed.
                        decelerate_factor = -1

                if decelerate_factor > 1:
                    decelerate_factor = 1
                return -v1 * decelerate_factor, idx1, decelerate_factor
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

        # Remove all combinations not in resopairs from is_leading flag.
        allowed_combinations = {idx: [] for idx in range(ownship.ntraf)}
        for pair in self.resopairs:
            allowed_combinations[ownship.id.index(pair[0])].append(pair[1])
        for idx in range(ownship.ntraf):
            to_pop = [acid for acid in self.is_leading[idx].keys() if acid not in allowed_combinations[idx]]
            for acid in to_pop:
                self.is_leading[idx].pop(acid)

    def detect(self, pairs, ownship, intruder) -> dict:
        """
        Detects conflict properties for all pairs.
        Based on detect of statebased.py
        """
        # Extract indices of pairs.
        properties = {}
        for pair in pairs.copy():
            try:
                idx = [traf.id.index(ac) for ac in pair]
                properties[pair] = {'idx1': idx[0], 'idx2': idx[1]}
            except ValueError:
                # Aircraft already deleted.
                self.resopairs.remove(pair)

        # Identity matrix of order ntraf: avoid ownship-ownship detected conflicts
        I = np.eye(ownship.ntraf)

        # Horizontal conflict ------------------------------------------------------

        # qdlst is for [i,j] qdr from i to j, from perception of ADSB and own coordinates
        qdr, dist = geo.kwikqdrdist_matrix(np.asmatrix(ownship.lat), np.asmatrix(ownship.lon),
                                           np.asmatrix(intruder.lat), np.asmatrix(intruder.lon))

        # Convert back to array to allow element-wise array multiplications later on
        # Convert to meters and add large value to own/own pairs
        qdr = np.asarray(qdr)
        dist = np.asarray(dist) * nm + 1e9 * I

        # Calculate horizontal closest point of approach (CPA)
        qdrrad = np.radians(qdr)
        dx = dist * np.sin(qdrrad)  # is pos j rel to i
        dy = dist * np.cos(qdrrad)  # is pos j rel to i

        # Ownship track angle and speed
        owntrkrad = np.radians(ownship.trk)
        ownu = ownship.gs * np.sin(owntrkrad).reshape((1, ownship.ntraf))  # m/s
        ownv = ownship.gs * np.cos(owntrkrad).reshape((1, ownship.ntraf))  # m/s

        # Intruder track angle and speed
        inttrkrad = np.radians(intruder.trk)
        intu = intruder.gs * np.sin(inttrkrad).reshape((1, ownship.ntraf))  # m/s
        intv = intruder.gs * np.cos(inttrkrad).reshape((1, ownship.ntraf))  # m/s

        du = ownu - intu.T  # Speed du[i,j] is perceived eastern speed of i to j
        dv = ownv - intv.T  # Speed dv[i,j] is perceived northern speed of i to j

        dv2 = du * du + dv * dv
        dv2 = np.where(np.abs(dv2) < 1e-6, 1e-6, dv2)  # limit lower absolute value
        vrel = np.sqrt(dv2)

        tcpa = -(du * dx + dv * dy) / dv2 + 1e9 * I

        # Calculate distance^2 at CPA (minimum distance^2)
        dcpa2 = np.abs(dist * dist - tcpa * tcpa * dv2)

        # Check for horizontal conflict
        R2 = traf.cd.rpz * traf.cd.rpz
        swhorconf = dcpa2 < R2  # conflict or not

        # Calculate times of entering and leaving horizontal conflict
        dxinhor = np.sqrt(np.maximum(0., R2 - dcpa2))  # half the distance travelled inzide zone
        dtinhor = dxinhor / vrel

        tinhor = np.where(swhorconf, tcpa - dtinhor, 1e8)  # Set very large if no conf

        # Vertical conflict --------------------------------------------------------

        # Vertical crossing of disk (-dh,+dh)
        dalt = ownship.alt.reshape((1, ownship.ntraf)) - \
            intruder.alt.reshape((1, ownship.ntraf)).T  + 1e9 * I

        dvs = ownship.vs.reshape(1, ownship.ntraf) - \
            intruder.vs.reshape(1, ownship.ntraf).T
        dvs = np.where(np.abs(dvs) < 1e-6, 1e-6, dvs)  # prevent division by zero

        # Check for passing through each others zone
        tcrosshi = (dalt + traf.cd.hpz) / -dvs
        tcrosslo = (dalt - traf.cd.hpz) / -dvs
        tinver = np.minimum(tcrosshi, tcrosslo)

        # Combine vertical and horizontal conflict----------------------------------
        tinconf = np.maximum(tinver, tinhor)

        # --------------------------------------------------------------------------
        # Update conflict lists
        # --------------------------------------------------------------------------
        # Extract for pair in resopairs.
        for pair in pairs:
            idx1 = properties[pair]['idx1']
            idx2 = properties[pair]['idx2']
            properties[pair].update({
                'qdr': qdr[idx1, idx2],
                'dist': dist[idx1, idx2],
                'tcpa': tcpa[idx1, idx2],
                'tlos': tinconf[idx1, idx2]
            })

        return properties
