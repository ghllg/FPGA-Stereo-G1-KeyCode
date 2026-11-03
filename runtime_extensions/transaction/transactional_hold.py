"""Stage the existing hold cache until local quality/mapping/freshness admission.

No new point selection, observation type, hold age or missing-count rule.
The caller must settle every preview. Seen identity/time are never rolled back.
"""
from copy import deepcopy

from joint_hold import JointHold


class TransactionalJointHold(JointHold):
    def __init__(self):
        super().__init__()
        self.pending = None

    def apply(self, packet, enabled):
        if self.pending is not None:
            raise RuntimeError('Previous hold preview was not settled')
        committed = self.cache
        self.cache = deepcopy(committed)
        try:
            output, held = super().apply(packet, enabled)
            self.pending = self.cache
        finally:
            self.cache = committed
        return output, held

    def commit(self):
        if self.pending is None:
            raise RuntimeError('No hold preview to commit')
        self.cache = self.pending
        self.pending = None

    def abort(self):
        self.pending = None


def recovery_eligible(admission, held, packet):
    """Preserve existing estimated inputs; held/newly rejected/forecast are not evidence."""
    semantic = packet.get('semantic_admission', {})
    return bool(admission['accepted'] and not held
                and semantic.get('reliable_for_recovery', True)
                and not packet.get('prediction_only', False))
