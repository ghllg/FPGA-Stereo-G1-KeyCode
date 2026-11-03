"""Candidate standing-motion envelope. It does not certify physical foot contact.

No coordinates, timestamps, quality flags, or BVH values are rewritten. Call
freeze with the SAME raw, quality-admitted initial prefix used for body lengths.
The thresholds are explicit experimental scope limits, not accuracy guarantees.
"""
import numpy as np


class StandingPreparationPending(ValueError):
    """A new observed, stationary prefix can satisfy this preparation check."""


class StandingScopeGate:
    """Only a proved outside frame latches this extra pause reason."""
    def __init__(self):
        self.paused_for_scope = False

    def blocks(self, result):
        if result['state'] == 'outside':
            self.paused_for_scope = True
        return self.paused_for_scope and result['state'] != 'within'

    def recovery_eligible(self, original_eligible, result):
        return original_eligible and (not self.paused_for_scope or result['state'] == 'within')

    def resumed(self):
        self.paused_for_scope = False


class StandingScope:
    WORLD = 'world_right_handed_X_forward_Y_left_Z_up'
    IDENTITY = ('session_id', 'clock_id', 'calibration_id', 'model_id')
    WARMUP_JOINTS = (5, 6, 11, 12, 13, 14, 15, 16)
    SCOPE_JOINTS = (5, 6, 11, 12, 15, 16)
    LIMITS = dict(ankle_vertical_m=.05, ankle_horizontal_m=.08,
                  hip_drop_m=.10, hip_rise_m=.05, hip_horizontal_m=.10,
                  torso_tilt_deg=20.)

    def __init__(self):
        self.identity = None
        self.ankles = self.hip = None
        self.previous = (-1, -1)

    @staticmethod
    def _measured(packet, indices):
        joints = packet['joints']
        return all(joints[i]['valid'] and not joints[i].get('estimated', False)
                   and joints[i].get('reason') == 'observed'
                   and np.isfinite(joints[i]['xyz']).all() for i in indices)

    @staticmethod
    def _geometry(packet):
        points = np.full((17, 3), np.nan)
        for i in StandingScope.WARMUP_JOINTS:
            if packet['joints'][i]['valid']:
                points[i] = packet['joints'][i]['xyz']
        hip = points[[11, 12]].mean(axis=0)
        torso = points[[5, 6]].mean(axis=0) - hip
        tilt = float(np.degrees(np.arctan2(np.linalg.norm(torso[:2]), torso[2])))
        return points, hip, tilt

    def freeze(self, prefix):
        """Reject bad initial calibration rather than silently anchoring later."""
        if self.identity is not None:
            raise ValueError('Standing reference is immutable; create a new session')
        if len(prefix) < 10:
            raise StandingPreparationPending('站立基准需要至少 10 帧可靠人体观测')
        identity = tuple(prefix[0][k] for k in self.IDENTITY)
        stamps = np.array([p['timestamp_ns'] for p in prefix], dtype=np.int64)
        seqs = np.array([p['seq'] for p in prefix], dtype=np.int64)
        if np.any(np.diff(stamps) <= 0) or np.any(np.diff(seqs) <= 0):
            raise ValueError('Standing prefix timestamps and seq must strictly increase')
        if stamps[-1] - stamps[0] < 1_400_000_000 or np.any(np.diff(stamps) > 200_000_000):
            raise StandingPreparationPending('站立观测需持续 1.4 秒，可靠帧间隔不超过 200 ms')
        geometry = []
        ankle_samples = []
        used_seqs = []
        for packet in prefix:
            if (packet['coordinate_system'] != self.WORLD or packet['unit'] != 'm'
                    or tuple(packet[k] for k in self.IDENTITY) != identity):
                raise ValueError('Standing prefix identity or coordinate system changed')
            if not self._measured(packet, (15, 16)):
                raise StandingPreparationPending('等待双脚踝当前真实观测，补点不能建立站立基准')
            ankle_samples.append([packet['joints'][i]['xyz'] for i in (15, 16)])
            # Missing hips cannot contribute to the personal reference. Other
            # observed frames in this SAME prefix may still establish it.
            if not self._measured(packet, self.WARMUP_JOINTS):
                continue
            points, hip, tilt = self._geometry(packet)
            if tilt > 15. or abs(points[15, 2] - points[16, 2]) > .04:
                raise StandingPreparationPending('站立基准尚未建立：躯干倾斜或双脚高度差超限；请站直、双脚放平')
            for a, b, c in ((11, 13, 15), (12, 14, 16)):
                upper, lower = points[b] - points[a], points[c] - points[b]
                cosine = np.dot(upper, lower) / max(np.linalg.norm(upper) * np.linalg.norm(lower), 1e-12)
                if np.degrees(np.arccos(np.clip(cosine, -1., 1.))) > 20.:
                    raise StandingPreparationPending('站立基准尚未建立：膝关节观测弯曲超限；请自然站直')
            geometry.append((points[[15, 16]], hip))
            used_seqs.append(packet['seq'])
        if len(geometry) < 10 or not self._measured(prefix[-1], self.WARMUP_JOINTS):
            raise StandingPreparationPending('等待至少 10 帧肩、髋、膝、踝真实观测，且当前帧完整')
        ankles = np.array(ankle_samples)
        hips = np.array([g[1] for g in geometry])
        ankle_reference = np.median(ankles, axis=0)
        hip_reference = np.median(hips, axis=0)
        if (np.max(np.linalg.norm(ankles - ankle_reference, axis=2)) > .025
                or np.max(np.linalg.norm(hips - hip_reference, axis=1)) > .04):
            raise StandingPreparationPending('站立观测仍有移动或抖动，正在重新积累稳定基准')
        self.identity, self.ankles, self.hip = identity, ankle_reference, hip_reference
        self.previous = (int(stamps[-1]), int(seqs[-1]))
        return dict(ankles_m=self.ankles.tolist(), hip_m=self.hip.tolist(),
                    first_seq=prefix[0]['seq'], last_seq=prefix[-1]['seq'],
                    observed_body_seqs=used_seqs,
                    limits=dict(self.LIMITS), physical_contact_verified=False)

    def assess(self, packet, quality_accepted):
        if self.identity is None:
            raise ValueError('Freeze the standing prefix before assessing motion')
        identity = tuple(packet[k] for k in self.IDENTITY)
        if (identity != self.identity or packet['coordinate_system'] != self.WORLD
                or packet['unit'] != 'm'):
            raise ValueError('Standing scope cannot cross an input identity or frame change')
        stamp, seq = packet['timestamp_ns'], packet['seq']
        if stamp <= self.previous[0] or seq <= self.previous[1]:
            raise ValueError('Standing scope input must strictly increase')
        self.previous = (stamp, seq)
        common = dict(source_seq=seq, source_timestamp_ns=stamp, physical_contact_verified=False)
        if not quality_accepted or not self._measured(packet, self.SCOPE_JOINTS):
            return dict(**common, state='unknown', decision='defer_to_existing_quality',
                        reasons=['standing_scope_current_observation_unavailable'])
        points, hip, tilt = self._geometry(packet)
        delta = points[[15, 16]] - self.ankles
        hip_delta = hip - self.hip
        values = dict(ankle_vertical_m=float(np.max(np.abs(delta[:, 2]))),
                      ankle_horizontal_m=float(np.max(np.linalg.norm(delta[:, :2], axis=1))),
                      hip_drop_m=float(max(0., -hip_delta[2])),
                      hip_rise_m=float(max(0., hip_delta[2])),
                      hip_horizontal_m=float(np.linalg.norm(hip_delta[:2])),
                      torso_tilt_deg=tilt)
        reasons = ['standing_scope_' + k for k, v in values.items() if v > self.LIMITS[k]]
        return dict(**common, state='outside' if reasons else 'within',
                    decision='pause' if reasons else 'pass',
                    reasons=reasons, measurements=values)


class StandingWarmup:
    """Validate the same raw prefix before LiveBVH can freeze its bone lengths."""
    def __init__(self, converter, scope):
        if converter.warmup < 10 or converter.warmup_seconds < 1.4:
            raise ValueError('Standing preparation requires at least 10 frames spanning 1.4 s')
        self.converter, self.scope = converter, scope
        self.prefix = []
        self.profile = None
        self.reason = '正在积累可靠站立观测'
        self.missing_joints = []

    def reset(self):
        self.prefix.clear()
        self.converter.prefix.clear()
        self.converter.prefix_start_ns = None

    def snapshot(self):
        return dict(ready=self.scope.identity is not None, reason=self.reason,
                    missing_joints=list(self.missing_joints),
                    accepted_frames=len(self.prefix),
                    span_s=(self.prefix[-1]['timestamp_ns']-self.prefix[0]['timestamp_ns'])/1e9
                    if self.prefix else 0.)

    def push(self, packet):
        if self.scope.identity is not None:
            return self.converter.push(packet)
        if (not self.converter.prefix or self.prefix and
                packet['timestamp_ns']-self.prefix[-1]['timestamp_ns'] > 200_000_000):
            self.reset()
        self.missing_joints = [i for i in (15, 16) if not self.scope._measured(packet, (i,))]
        if self.missing_joints:
            names = {15: '左脚踝', 16: '右脚踝'}
            self.reason = '等待'+'、'.join(names[i] for i in self.missing_joints)+'当前真实观测；补点不能建立站立基准'
            return None
        proposed = self.prefix + [packet]
        can_freeze = (len(proposed) >= self.converter.warmup and
                      (packet['timestamp_ns']-proposed[0]['timestamp_ns'])/1e9 >= self.converter.warmup_seconds)
        if can_freeze:
            try:
                StandingScope().freeze(proposed)
            except StandingPreparationPending as error:
                self.reset()
                self.reason = str(error)
                return None
        count = len(self.converter.prefix)
        frame = self.converter.push(packet)
        if frame is None:
            if len(self.converter.prefix) > count:
                self.prefix.append(packet)
            self.reason = '正在积累可靠站立观测' if len(self.converter.prefix) > count else '等待可转换的完整人体观测'
            return None
        # The identical raw prefix was checked before converter.push committed.
        self.profile = self.scope.freeze(proposed)
        self.prefix.clear()
        self.reason = '人体骨长与站立基准均已就绪'
        return frame
