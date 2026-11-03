"""Causal coordinate adapter for the recovered E:/outputs BVH convention."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R


def recovered_skeleton(sk):
    expected = {"LeftLeg": (0,-1,0), "RightLeg": (0,-1,0),
                "Spine1": (0,1,0), "Spine2": (0,1,0),
                "LeftForeArm": (1,0,0), "RightForeArm": (-1,0,0),
                "LeftToe": (0,0,1), "RightToe": (0,0,1)}
    for name, direction in expected.items():
        if name not in sk.canonical:
            return False
        v = sk.offsets[sk.canonical.index(name)]
        if np.linalg.norm(v) < 1e-8 or np.dot(v / np.linalg.norm(v), direction) < .999:
            return False
    return True


def configure_recovered_solver(solver):
    """Prioritize measured torso/endpoints over unobserved limb axial twist."""
    for tasks in (solver.human_body_to_task1, solver.human_body_to_task2):
        tasks["Spine2"].set_orientation_cost(100)
        tasks["Hips"].set_position_cost(100)
        for side in ("Left", "Right"):
            tasks[side+"Arm"].set_orientation_cost(10)
            tasks[side+"UpLeg"].set_orientation_cost(0)
            tasks[side+"UpLeg"].set_position_cost(0)
            tasks[side+"Leg"].set_orientation_cost(0)
            tasks[side+"Leg"].set_position_cost(25)
            tasks[side+"FootMod"].set_position_cost(200)
    # The model's pelvis origin is above its anatomical hip-joint midpoint.
    data = mujoco.MjData(solver.model)
    mujoco.mj_forward(solver.model, data)
    hip_center = (data.body("left_hip_pitch_link").xpos + data.body("right_hip_pitch_link").xpos)/2
    solver.pos_offsets1["Hips"] = data.body("pelvis").xpos - hip_center
    # Select the forward-knee solution at initialization, avoiding a straight-leg
    # IK singularity. Subsequent frames solve freely; no joint is held fixed.
    q = solver.configuration.data.qpos.copy()
    for side in ("left", "right"):
        for joint, angle in (("hip_pitch", -.25), ("knee", .5), ("ankle_pitch", -.25)):
            q[solver.model.joint(side+"_"+joint+"_joint").qposadr[0]] = angle
    solver.configuration.update(q)


class RecoveredAdapter:
    def __init__(self, skeleton, solver):
        self.corrections = {}
        self.offsets = solver.rot_offsets1
        model = solver.model
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        # Recovered rest pose: +Z forward, +X left, +Y up.
        heading = skeleton.basis * R.from_matrix(np.array([[0,1,0],[0,0,1],[1,0,0]]))
        links = {}
        for side, prefix in [("Left", "left"), ("Right", "right")]:
            links.update({side+"UpLeg": (side+"Leg", prefix+"_knee_link"),
                          side+"Leg": (side+"Foot", prefix+"_ankle_roll_link"),
                          side+"Arm": (side+"ForeArm", prefix+"_elbow_link"),
                          side+"ForeArm": (side+"Hand", prefix+"_wrist_yaw_link")})
        for body, entry in solver.ik_match_table1.items():
            name = entry[0]
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            target = heading * R.from_quat(data.xquat[index], scalar_first=True)
            if name in links:
                child, robot_child = links[name]
                ci = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_child)
                v = heading.apply(data.xpos[ci] - data.xpos[index])
                w = skeleton.basis.apply(skeleton.offsets[skeleton.canonical.index(child)])
                swing, _ = R.align_vectors([w/np.linalg.norm(w)], [v/np.linalg.norm(v)])
                target = swing * target
            self.corrections[name] = skeleton.basis.inv() * target * solver.rot_offsets1[name].inv()
        # A leaf wrist has no measured twist; follow its forearm convention.
        for side in ("Left", "Right"):
            self.corrections[side+"Hand"] = self.corrections[side+"ForeArm"]

    def apply(self, human):
        result = {name: [value[0].copy(),
                       (R.from_quat(value[1], scalar_first=True) * self.corrections[name]).as_quat(scalar_first=True)
                       if name in self.corrections else value[1].copy()]
                for name, value in human.items()}
        # The exporter's pelvis twist is estimated. Keep its heading, but
        # constrain torso up to the observed hip-to-chest direction.
        up = human['Spine2'][0] - human['Hips'][0]
        if np.linalg.norm(up) > 1e-8:
            up = up / np.linalg.norm(up)
            for name in ('Hips', 'Spine2'):
                target = R.from_quat(result[name][1], scalar_first=True) * self.offsets[name]
                swing, _ = R.align_vectors([up], [target.apply([0, 0, 1])])
                result[name][1] = (swing * target * self.offsets[name].inv()).as_quat(scalar_first=True)
        return result
