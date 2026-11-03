"""Render BVH forward-kinematics results and the validation learning curve."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio_ffmpeg
from .bvh import read_bvh
from mocap.io import write_json


def preview(run_dir):
    run = Path(run_dir)
    history = json.loads((run/'history.json').read_text(encoding='utf-8'))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
    axes[0].plot([row['epoch'] for row in history], [row['training_loss'] for row in history])
    axes[0].set(xlabel='Epoch', ylabel='Training objective', title='Training (subjects 1-3)')
    axes[1].plot([row['epoch'] for row in history], [row['validation_missing_mpjpe_m']*1000 for row in history], label='Missing joints')
    axes[1].plot([row['epoch'] for row in history], [row['validation_observed_mpjpe_m']*1000 for row in history], label='Observed joints')
    axes[1].set(xlabel='Epoch', ylabel='Joint error (mm)', title='Validation (subject 4)')
    axes[1].legend()
    fig.savefig(run/'learning_curve.png', dpi=150)
    plt.close(fig)
    report = dict(segments=[], note='Rendered from BVH rotations via FK. This is not independent accuracy validation.')
    clips = []
    for bvh_file in sorted((run/'bvh').glob('motion_*.bvh')):
        names, parents, bvh_points, dt = read_bvh(bvh_file.read_text(encoding='ascii'))
        with np.load(bvh_file.with_suffix('.npz'), allow_pickle=False) as saved:
            maximum = float(np.max(abs(bvh_points-saved['global_positions_cm']/100.)))
            if not maximum < 1e-6:
                raise ValueError(f'BVH / saved FK mismatch: {bvh_file}')
        report['segments'].append(dict(file=bvh_file.name, frames=len(bvh_points), max_bvh_vs_npz_m=maximum))
        clips.append(bvh_points[..., [2, 0, 1]])
    fig = plt.figure(figsize=(11, 3*len(clips)), layout='constrained')
    for segment, points in enumerate(clips):
        center = points.mean(axis=(0, 1))
        for column, frame in enumerate([0, len(points)//2, len(points)-1]):
            axis = fig.add_subplot(len(clips), 3, segment*3+column+1, projection='3d')
            for j in range(1, len(names)):
                pair = points[frame, [parents[j], j]]
                color = '#168fbd' if names[j].startswith('Left') else ('#d96848' if names[j].startswith('Right') else '#354454')
                axis.plot(pair[:, 0], pair[:, 1], pair[:, 2], color=color, marker='o', markersize=2)
            axis.set(xlim=(center[0]-.8, center[0]+.8), ylim=(center[1]-.8, center[1]+.8), zlim=(-.2, 1.8),
                     title=f'Segment {segment+1} / frame {frame+1}', zlabel='Z (m)')
            axis.set_box_aspect([1, 1, 1.3])
            axis.view_init(elev=10, azim=-62)
    fig.suptitle('Experimental BVH: every segment, start / middle / end')
    fig.savefig(run/'segments_overview.png', dpi=130)
    plt.close(fig)
    first = run/'bvh/motion_001.bvh'
    names, parents, bvh_points, dt = read_bvh(first.read_text(encoding='ascii'))
    points = bvh_points[..., [2, 0, 1]]  # BVH Y-up -> project world Z-up.
    low, high = points.min(axis=(0, 1)), points.max(axis=(0, 1))
    center = (low+high)/2
    extent = max(high-low)*.6
    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    canvas = FigureCanvasAgg(fig)
    axis = fig.add_subplot(111, projection='3d')
    axis.set(xlim=(center[0]-extent, center[0]+extent), ylim=(center[1]-extent, center[1]+extent),
             zlim=(max(-.2, low[2]-.15), high[2]+.15), xlabel='World X (m)', ylabel='World Y (m)', zlabel='Up Z (m)')
    axis.set_box_aspect([1, 1, 1.1])
    axis.view_init(elev=16, azim=-62)
    lines = []
    for j in range(1, len(names)):
        color = '#168fbd' if names[j].startswith('Left') else ('#d96848' if names[j].startswith('Right') else '#354454')
        lines.append(axis.plot([], [], [], color=color, linewidth=3, marker='o', markersize=3)[0])
    title = axis.set_title('')
    writer = imageio_ffmpeg.write_frames(str(run/'preview.mp4'), (960, 540), fps=1/(dt*2),
                                       codec='libx264', pix_fmt_in='rgb24', macro_block_size=2, ffmpeg_log_level='error')
    writer.send(None)
    try:
        for frame in range(0, len(points), 2):
            for j, line in enumerate(lines, 1):
                pair = points[frame, [parents[j], j]]
                line.set_data_3d(pair[:, 0], pair[:, 1], pair[:, 2])
            title.set_text(f'Experimental BVH | segment 1 of {len(report["segments"])} | t = {frame*dt:.2f} s')
            canvas.draw()
            writer.send(np.asarray(canvas.buffer_rgba())[:, :, :3].tobytes())
            if frame == 0:
                fig.savefig(run/'preview.png', dpi=130)
    finally:
        writer.close()
        plt.close(fig)
    report['video'] = 'preview.mp4; first continuous segment only, sampled every 2 frames; original duration retained.'
    write_json(run/'bvh_validation.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    preview(parser.parse_args().run)
