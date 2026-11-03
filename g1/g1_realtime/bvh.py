"""BVH header parsing and causal, one-frame forward kinematics (no future data)."""
import re
import numpy as np
from scipy.spatial.transform import Rotation


class Skeleton:
    def __init__(self, header, unit="cm", up="Y", mapping=None):
        if unit not in ("cm", "m") or up not in ("Y", "Z"):
            raise ValueError("单位须为 cm/m，竖直轴须为 Y/Z")
        tokens = re.findall(r"[{}]|[^\s{}]+", header.split("MOTION")[0])
        self.names, self.parents, self.offsets, self.channels = [], [], [], []
        self.count = 0
        cursor = 0

        def take(expected=None):
            nonlocal cursor
            if cursor >= len(tokens):
                raise ValueError("BVH 骨架头不完整")
            value = tokens[cursor]
            cursor += 1
            if expected is not None and value != expected:
                raise ValueError(f"BVH 头应包含 {expected}，实际为 {value}")
            return value

        def joint(parent):
            if len(self.names) >= 128:
                raise ValueError("骨架超过 128 个节点")
            name = take()
            if name in self.names:
                raise ValueError(f"重复骨骼名称: {name}")
            idx = len(self.names)
            self.names.append(name)
            self.parents.append(parent)
            self.offsets.append([0., 0., 0.])
            self.channels.append([])
            take("{")
            while True:
                token = take()
                if token == "}":
                    break
                if token == "OFFSET":
                    self.offsets[idx] = [float(take()) for _ in range(3)]
                elif token == "CHANNELS":
                    count = int(take())
                    if count not in (3, 6):
                        raise ValueError("目前支持每节点 3 或 6 个 BVH 通道")
                    channel_names = [take() for _ in range(count)]
                    valid = {axis + kind for axis in "XYZ" for kind in ("position", "rotation")}
                    if any(c not in valid for c in channel_names) or len(set(channel_names)) != count:
                        raise ValueError("无效或重复 BVH 通道")
                    rotations = [c[0] for c in channel_names if c.endswith("rotation")]
                    if len(rotations) != 3 or len(set(rotations)) != 3:
                        raise ValueError("每节点需要 XYZ 三个旋转通道，顺序可不同")
                    self.channels[idx] = [(c, self.count + n) for n, c in enumerate(channel_names)]
                    self.count += count
                elif token == "JOINT":
                    joint(idx)
                elif token == "End":
                    take("Site")
                    take("{")
                    take("OFFSET")
                    for _ in range(3):
                        float(take())
                    take("}")
                else:
                    raise ValueError(f"不支持的 BVH 头字段: {token}")

        take("HIERARCHY")
        take("ROOT")
        joint(-1)
        if cursor != len(tokens) or any(not c for c in self.channels):
            raise ValueError("BVH 骨架结构或通道定义不完整")
        self.offsets = np.asarray(self.offsets)
        if not np.isfinite(self.offsets).all():
            raise ValueError("骨架偏移包含非有限数值")
        self.scale = .01 if unit == "cm" else 1.
        self.basis = Rotation.from_euler("x", 90, degrees=True) if up == "Y" else Rotation.identity()
        self.mapping = mapping or {}
        if not isinstance(self.mapping, dict):
            raise ValueError("骨骼映射须为 JSON 对象：标准名称 → 输入名称")
        self.canonical = [next((k for k, v in self.mapping.items() if v == n), n.split(":")[-1]) for n in self.names]
        self.canonical = [{"LeftToeBase": "LeftToe", "RightToeBase": "RightToe"}.get(n, n) for n in self.canonical]
        if len(set(self.canonical)) != len(self.canonical):
            raise ValueError("映射后骨骼名称重复")

    def frame(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (self.count,) or not np.isfinite(values).all():
            raise ValueError(f"动作帧必须包含 {self.count} 个有限数值")
        positions, rotations = [], []
        for i, channels in enumerate(self.channels):
            local_pos = self.offsets[i].copy()
            axes, angles = "", []
            for channel, index in channels:
                if channel.endswith("position"):
                    local_pos["XYZ".index(channel[0])] += values[index]
                else:
                    axes += channel[0]
                    angles.append(values[index])
            local_rot = Rotation.from_euler(axes, angles, degrees=True)
            parent = self.parents[i]
            if parent < 0:
                positions.append(local_pos)
                rotations.append(local_rot)
            else:
                positions.append(positions[parent] + rotations[parent].apply(local_pos))
                rotations.append(rotations[parent] * local_rot)
        xyz = self.basis.apply(np.asarray(positions)) * self.scale
        human = {name: [xyz[i].copy(), (self.basis * rotations[i]).as_quat(scalar_first=True)]
                 for i, name in enumerate(self.canonical)}
        for side in ("Left", "Right"):
            foot, toe = side + "Foot", side + "Toe"
            if foot in human and toe in human:
                human[foot + "Mod"] = [human[foot][0].copy(), human[toe][1].copy()]
        return human, xyz


def read_clip(text):
    parts = re.split(r"\bMOTION\b", text, maxsplit=1)
    if len(parts) != 2:
        raise ValueError("缺少 MOTION 段")
    lines = [line.strip() for line in parts[1].splitlines() if line.strip()]
    count = int(lines[0].split(":", 1)[1])
    dt = float(lines[1].split(":", 1)[1])
    frames = [[float(v) for v in line.split()] for line in lines[2:]]
    if len(frames) != count or not 0 < dt <= 1:
        raise ValueError("帧数或 Frame Time 无效")
    return parts[0].strip(), dt, frames
