import contextlib
import io
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from bvh import Skeleton
from adapter import RecoveredAdapter, recovered_skeleton, configure_recovered_solver
from vendor.gmr import GeneralMotionRetargeting
from vendor.gmr.params import ROBOT_XML_DICT


class Tracker:
    def __init__(self, message):
        self.skeleton = Skeleton(message["header"], message.get("unit", "cm"), message.get("up", "Y"), message.get("mapping"))
        config = json.loads((Path(__file__).parent / "vendor/gmr/bvh_lafan1_to_g1.json").read_text())
        needed = set(config["human_scale_table"]) - {"LeftFootMod", "RightFootMod"}
        needed |= {"LeftFoot", "RightFoot", "LeftToe", "RightToe"}
        missing = needed - set(self.skeleton.canonical)
        if missing:
            raise ValueError("缺少映射骨骼: " + ", ".join(sorted(missing)) + "。请使用 LAFAN1 骨架或填写名称映射。")
        with contextlib.redirect_stdout(io.StringIO()):
            self.solver = GeneralMotionRetargeting("bvh_lafan1", "unitree_g1", actual_human_height=1.75, verbose=False)
        self.adapter = RecoveredAdapter(self.skeleton, self.solver) if recovered_skeleton(self.skeleton) else None
        if self.adapter:
            configure_recovered_solver(self.solver)
        self.model = self.solver.model
        self.data = self.solver.configuration.data
        self.origin = None
        self.metadata = {
            "type": "ready", "name": str(message.get("name", "实时 BVH"))[:120],
            "channel_count": self.skeleton.count,
            "adapter": "录像恢复骨架坐标适配" if self.adapter else "原版 LAFAN1",
            "recovered_skeleton": self.adapter is not None,
            "human_names": self.skeleton.names, "human_parents": self.skeleton.parents,
            "body_names": [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(self.model.nbody)],
            "body_parents": self.model.body_parentid.tolist(),
            "joint_names": [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, self.model.njnt)],
            "joint_limits": self.model.jnt_range[1:].tolist(), "quaternion_order": "wxyz", "position_unit": "m",
        }

    def solve(self, values):
        start = time.perf_counter()
        human, xyz = self.skeleton.frame(values)
        if self.origin is None:
            self.origin = human["Hips"][0].copy()
            self.origin[2] = 0
        xyz -= self.origin
        for value in human.values():
            value[0] -= self.origin
        if self.adapter:
            human = self.adapter.apply(human)
        qpos = self.solver.retarget(human)
        if not np.isfinite(qpos).all():
            raise ValueError("求解结果包含非有限数值")
        mujoco.mj_forward(self.model, self.data)
        return {
            "qpos": qpos.tolist(), "root_pos": qpos[:3].tolist(), "root_quat": qpos[3:7].tolist(),
            "joint_pos": qpos[7:].tolist(), "body_pos": self.data.xpos.tolist(),
            "body_quat": self.data.xquat.tolist(), "human_pos": xyz.tolist(),
            "geom_pos": self.data.geom_xpos.tolist(), "geom_mat": self.data.geom_xmat.tolist(),
            "solve_ms": round((time.perf_counter() - start) * 1000, 3),
        }


def model_description():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["unitree_g1"]))
    meshes = []
    for i in range(model.nmesh):
        va, vn = model.mesh_vertadr[i], model.mesh_vertnum[i]
        fa, fn = model.mesh_faceadr[i], model.mesh_facenum[i]
        meshes.append({"vertices": model.mesh_vert[va:va+vn].flatten().tolist(),
                       "faces": model.mesh_face[fa:fa+fn].flatten().tolist()})
    geoms = [{"index": i, "mesh": int(model.geom_dataid[i]), "rgba": model.geom_rgba[i].tolist(),
              "body": int(model.geom_bodyid[i]), "local_pos": model.geom_pos[i].tolist(),
              "local_quat": model.geom_quat[i].tolist()}
             for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH]
    return {"meshes": meshes, "geoms": geoms}
