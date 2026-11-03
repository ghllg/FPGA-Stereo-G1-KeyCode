"""Bound short joint dropouts for G1 only; never alter saved 3D observations."""
from copy import deepcopy

from conversion import CORE, validate


class JointHold:
    MAX_AGE_NS = 100_000_000
    MAX_MISSING = 6  # At least half of the twelve body joints must be current.

    def __init__(self):
        self.cache = {}
        self.identity = None
        self.previous = (-1, -1)

    def apply(self, packet, enabled):
        validate(packet)
        identity = tuple(packet[k] for k in ('session_id', 'clock_id', 'calibration_id', 'model_id'))
        stamp, seq = packet['timestamp_ns'], packet['seq']
        if self.identity is not None and identity != self.identity:
            raise ValueError('Cannot carry held joints across a source identity change')
        if stamp <= self.previous[0] or seq <= self.previous[1]:
            raise ValueError('Held-joint input must strictly increase')
        self.identity, self.previous = identity, (stamp, seq)
        missing = []
        for index in CORE:
            joint = packet['joints'][index]
            if joint['valid']:
                self.cache[index] = (stamp, deepcopy(joint))
            else:
                missing.append(index)
        held = []
        if (not enabled or not missing or len(missing)>self.MAX_MISSING or
                any(i not in self.cache or stamp-self.cache[i][0]>self.MAX_AGE_NS for i in missing)):
            return packet, held
        output = deepcopy(packet)
        for index in missing:
            old_stamp, old = self.cache[index]
            output['joints'][index] = dict(old, estimated=True, confidence=0.,
                reason='g1_causal_hold', raw_xyz=None, reprojection_px=None, parallax_deg=None)
            held.append(dict(index=int(index), source_timestamp_ns=old_stamp,
                             age_ms=(stamp-old_stamp)/1e6))
        return output, held
