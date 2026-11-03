"""PP-TinyPose heatmap postprocessing, separate from model execution.

DARK Gaussian/log-Taylor method follows PaddleDetection HRNetPostProcess:
https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/deploy/python/keypoint_postprocess.py
No inference, image capture, or model substitution occurs in this module.
"""
import cv2
import numpy as np


def dark_decode(heat, roi):
    heat = np.asarray(heat)
    if heat.shape != (17, 32, 24) or not np.isfinite(heat).all():
        raise ValueError('Expected a finite COCO17 17x32x24 heatmap.')
    if np.shape(roi) != (4,) or not np.isfinite(roi).all() or min(roi[2:]) <= 0:
        raise ValueError('Invalid original-image ROI.')
    rx, ry, rw, rh = roi
    result = []
    for raw in heat:
        y, x = divmod(int(raw.argmax()), 24)
        point = np.array([float(x), float(y)])
        score = float(raw.max())
        if score > 0 and 1 < x < 22 and 1 < y < 30:
            blurred = cv2.GaussianBlur(np.pad(raw.astype(np.float64), 1), (3, 3), 0)[1:-1, 1:-1]
            blurred = (blurred * (score / blurred.max())).astype(np.float32)
            h = np.log(np.maximum(blurred, np.float32(1e-10)))
            gradient = np.array([.5*(h[y,x+1]-h[y,x-1]), .5*(h[y+1,x]-h[y-1,x])])
            xx = .25*(h[y,x+2]-2*h[y,x]+h[y,x-2])
            yy = .25*(h[y+2,x]-2*h[y,x]+h[y-2,x])
            xy = .25*(h[y+1,x+1]-h[y-1,x+1]-h[y+1,x-1]+h[y-1,x-1])
            matrix = np.array([[xx, xy], [xy, yy]])
            if np.linalg.det(matrix) != 0:
                offset = np.linalg.solve(matrix, gradient)
                if np.isfinite(offset).all():
                    point -= offset
        result.append([rx+point[0]*rw/24, ry+point[1]*rh/32, score])
    return np.array(result)


def guided_roi(points):
    """Single known subject in 1280x720; padded pose bounds, no future frames."""
    points = np.asarray(points)
    if points.shape != (17, 3):
        raise ValueError('Expected COCO17 x/y/raw-score points.')
    visible = points[(points[:,2] >= .35) & np.isfinite(points).all(axis=1), :2]
    if len(visible) < 8:
        return [435, 180, 405, 540]
    lo, hi = visible.min(axis=0), visible.max(axis=0)
    height = min(max((hi[1]-lo[1])*1.3, (hi[0]-lo[0])*1.3/.75, 320), 720)
    width = height*.75
    center = np.clip((lo+hi)/2, [width/2, height/2], [1280-width/2, 720-height/2])
    return [int(round(center[0]-width/2)), int(round(center[1]-height/2)),
            int(round(width)), int(round(height))]
