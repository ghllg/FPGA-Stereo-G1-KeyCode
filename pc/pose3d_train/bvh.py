"""Read the LAFAN1 ROOT-6/JOINT-3 BVH convention and compute forward kinematics."""
import re
import numpy as np
from scipy.spatial.transform import Rotation


def read_bvh(text):
    hierarchy, motion = text.split('MOTION', 1)
    names, parents, offsets, channels, stack = [], [], [], [], []
    pending = -1
    for line in hierarchy.splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] in ('ROOT', 'JOINT'):
            pending = len(names)
            names.append(words[1])
            parents.append(stack[-1] if stack else -1)
            offsets.append([0., 0., 0.])
            channels.append([])
        elif words[:2] == ['End', 'Site']:
            pending = -1
        elif words[0] == '{':
            stack.append(pending)
        elif words[0] == '}':
            stack.pop()
        elif words[0] == 'OFFSET' and stack[-1] != -1:
            offsets[stack[-1]] = list(map(float, words[1:4]))
        elif words[0] == 'CHANNELS':
            channels[stack[-1]] = words[2:]
    frames = int(re.search(r'Frames:\s*(\d+)', motion)[1])
    time_match = re.search(r'Frame Time:\s*([\d.eE+-]+)', motion)
    dt = float(time_match[1])
    values = np.fromstring(motion[time_match.end():], sep=' ')
    width = sum(map(len, channels))
    if frames < 1 or dt <= 0 or values.size != frames * width or not np.isfinite(values).all():
        raise ValueError('Invalid BVH motion length, time, or numeric values.')
    values = values.reshape(frames, width)
    positions = np.broadcast_to(np.array(offsets) / 100., (frames, len(names), 3)).copy()
    rotations = np.empty((frames, len(names), 3, 3))
    column = 0
    for j, channel in enumerate(channels):
        expected = 6 if j == 0 else 3
        if len(channel) != expected or (j == 0 and channel[:3] != ['Xposition', 'Yposition', 'Zposition']):
            raise ValueError('Only LAFAN1 ROOT-6/JOINT-3 channels are supported.')
        order = ''.join(c[0] for c in channel[-3:])
        if set(order) != set('XYZ') or any(not c.endswith('rotation') for c in channel[-3:]):
            raise ValueError('BVH must have three distinct Euler rotation channels.')
        if j == 0:
            # LAFAN1 root position channels replace the root OFFSET (official reader convention).
            positions[:, 0] = values[:, column:column+3] / 100.
        rotations[:, j] = Rotation.from_euler(order, values[:, column+expected-3:column+expected], degrees=True).as_matrix()
        column += expected
    global_positions = np.empty_like(positions)
    global_rotations = np.empty_like(rotations)
    for j, parent in enumerate(parents):
        if parent == -1:
            global_positions[:, j] = positions[:, j]
            global_rotations[:, j] = rotations[:, j]
        else:
            global_positions[:, j] = global_positions[:, parent] + np.einsum('nij,nj->ni', global_rotations[:, parent], positions[:, j])
            global_rotations[:, j] = global_rotations[:, parent] @ rotations[:, j]
    return names, np.array(parents), global_positions, dt
