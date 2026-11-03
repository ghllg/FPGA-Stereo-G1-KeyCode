"""Track measured hip-to-ankle height changes through leg references."""
import mujoco
import numpy as np


class HeightFeedback:
    def __init__(self, physics, human_leg_length):
        if not np.isfinite(human_leg_length) or human_leg_length <= 0:
            raise ValueError("Source leg length must be finite and positive")
        rest = mujoco.MjData(physics.model)
        mujoco.mj_forward(physics.model, rest)
        robot_length = np.mean([
            np.linalg.norm(rest.body(s+"_hip_pitch_link").xpos-rest.body(s+"_knee_link").xpos)+
            np.linalg.norm(rest.body(s+"_knee_link").xpos-rest.body(s+"_ankle_roll_link").xpos)
            for s in ("left", "right")])
        self.scale = robot_length/human_leg_length
        feet = [i for i in range(physics.model.ngeom) if
                "foot" in (mujoco.mj_id2name(physics.model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
                and physics.model.geom_contype[i]]
        bottom = min(rest.geom_xpos[i,2]-abs(rest.geom_xmat[i,8])*physics.model.geom_size[i,1]
                     -physics.model.geom_size[i,0] for i in feet)
        self.standing_height = rest.qpos[2]-bottom
        self.origin = None
        self.filtered = 0.
        self.indices = [[physics.joints.index(side+"_"+joint+"_joint")
                         for joint in ("hip_pitch", "knee", "ankle_pitch")]
                        for side in ("left", "right")]
        self.limits = physics.model.jnt_range[1:]

    def apply(self, refs, height, actual, velocity, dt):
        if self.origin is None:
            self.origin = actual[2]-self.scale*height
        target = self.origin+self.scale*height
        # Preserve the source target for error reporting; do not chase a height
        # above the model's grounded, straight-leg standing posture.
        reachable = min(target, self.standing_height)
        desired = np.clip(-48.*(reachable-actual[2])+.4*velocity[2], -.7, .7)
        previous = self.filtered
        self.filtered += (desired-self.filtered)*(1-np.exp(-dt/.1))
        correction_velocity = (self.filtered-previous)/dt
        positions = refs["dof_pos"].copy()
        velocities = refs["dof_vel"].copy()
        for hip, knee, ankle in self.indices:
            knee_raw = positions[:,knee]+self.filtered
            knee_delta = np.clip(knee_raw, *self.limits[knee])-positions[:,knee]
            knee_velocity = np.where((knee_raw > self.limits[knee,0]) & (knee_raw < self.limits[knee,1]),
                                     velocities[:,knee]+correction_velocity, 0.)
            knee_delta_velocity = knee_velocity-velocities[:,knee]
            ankle_raw = positions[:,ankle]-.5*knee_delta
            ankle_delta = np.clip(ankle_raw, *self.limits[ankle])-positions[:,ankle]
            ankle_velocity = np.where((ankle_raw > self.limits[ankle,0]) & (ankle_raw < self.limits[ankle,1]),
                                      velocities[:,ankle]-.5*knee_delta_velocity, 0.)
            ankle_delta_velocity = ankle_velocity-velocities[:,ankle]
            hip_raw = positions[:,hip]-knee_delta-ankle_delta
            hip_velocity = np.where((hip_raw > self.limits[hip,0]) & (hip_raw < self.limits[hip,1]),
                                    velocities[:,hip]-knee_delta_velocity-ankle_delta_velocity, 0.)
            positions[:,knee] += knee_delta
            positions[:,ankle] += ankle_delta
            positions[:,hip] = np.clip(hip_raw, *self.limits[hip])
            velocities[:,hip], velocities[:,knee], velocities[:,ankle] = hip_velocity, knee_velocity, ankle_velocity
        result = dict(refs, dof_pos=positions, dof_vel=velocities)
        return result, target
