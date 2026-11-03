"""Candidate only: the existing metrics.py:156/173 displacement criterion.

No IK/recovery anchor, position-feedback origin, model, physics or source rewrite.
The baseline is the first actual physical output, exactly as offline metrics.py.
"""
import math

MAX_HORIZONTAL_ERROR_M = 0.2


class ActualTrackingGuard:
    def __init__(self):
        self.initial_root_xy = self.initial_human_xy = None
        self.latest = self.failure = None

    def observe(self, root, human_hip, source_time):
        root_xy = tuple(float(v) for v in root[:2])
        human_xy = tuple(float(v) for v in human_hip[:2])
        finite = all(math.isfinite(v) for v in (*root_xy, *human_xy, source_time))
        if finite and self.initial_root_xy is None:
            self.initial_root_xy, self.initial_human_xy = root_xy, human_xy
        error = (math.hypot(*[(root_xy[i]-self.initial_root_xy[i])-
                 (human_xy[i]-self.initial_human_xy[i]) for i in (0, 1)])
                 if finite else None)
        rejected = not finite or error > MAX_HORIZONTAL_ERROR_M
        self.latest = dict(source_time=float(source_time) if math.isfinite(source_time) else None, error_m=error,
            maximum_error_m=MAX_HORIZONTAL_ERROR_M,
            root_xy=tuple(v if math.isfinite(v) else None for v in root_xy),
            human_hip_xy=tuple(v if math.isfinite(v) else None for v in human_xy),
            initial_root_xy=self.initial_root_xy, initial_human_xy=self.initial_human_xy,
            accepted=not rejected,
            reason='nonfinite_actual_tracking_evidence' if not finite else
                   'actual_horizontal_tracking_error_exceeds_existing_0.2m' if rejected else None,
            scope='Actual root displacement minus same-source human hip displacement; immutable first physics baseline.')
        if rejected and self.failure is None:
            self.failure = self.latest.copy()
        return dict(self.latest, failed=self.failure is not None)

    def recovery_ready(self, recovery_start_source_time):
        return bool(self.failure is None and self.latest and self.latest['accepted']
                    and self.latest['source_time'] >= recovery_start_source_time)

    def failure_message(self):
        return ('G1 actual tracking guard failed; simulation stopped, new session required: '
                + str(self.failure['reason']))
