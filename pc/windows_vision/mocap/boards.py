"""Board definitions: checkerboard inner corners vs ChArUco square counts."""
from dataclasses import dataclass, asdict
from pathlib import Path
import cv2
import numpy as np
from .io import write_json

# Several GUI workers can run at once. Avoid each OpenCV process creating a
# machine-wide pool (and exhausting Windows threads/resources).
cv2.setNumThreads(2)


@dataclass(frozen=True)
class BoardSpec:
    kind: str = 'checkerboard'
    cols: int = 8
    rows: int = 5
    square_m: float = .025
    marker_m: float | None = None
    dictionary: str = 'DICT_5X5_100'
    legacy: bool = False
    first_marker_id: int = 0
    top_left_white: bool = False

    def __post_init__(self):
        if self.kind not in ('checkerboard', 'charuco') or self.cols < 3 or self.rows < 3 or not 0 < self.square_m < 1:
            raise ValueError('板型、阵列数量或格子尺寸错误。')
        if self.kind == 'charuco':
            if self.marker_m is None: object.__setattr__(self, 'marker_m', self.square_m*.8)
            if not 0 < self.marker_m < self.square_m: raise ValueError('标记边长必须小于格子边长。')

    def to_dict(self):
        return dict(asdict(self), count_semantics='square_count' if self.kind == 'charuco' else 'inner_corner_count')


def charuco_board(spec):
    if not hasattr(cv2, 'aruco') or not hasattr(cv2.aruco, spec.dictionary) or not spec.dictionary.startswith('DICT_'):
        raise ValueError('当前 OpenCV 不支持指定 ArUco 字典。')
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))
    # OpenCV's odd-row board cannot express this supplier's white-first parity.
    # Use a virtual extra bottom row with legacy even-row parity, never use its
    # markers/corners outside the physical board. XY metric coordinates stay unchanged.
    detector_rows = spec.rows + 1 if spec.top_left_white and spec.rows % 2 else spec.rows
    legacy = True if spec.top_left_white else spec.legacy
    provisional = cv2.aruco.CharucoBoard((spec.cols, detector_rows), spec.square_m, spec.marker_m, dictionary)
    count = len(provisional.getIds())
    if spec.first_marker_id < 0 or spec.first_marker_id+count > len(dictionary.bytesList):
        raise ValueError('标记 ID 数量超出字典容量。')
    board = cv2.aruco.CharucoBoard((spec.cols, detector_rows), spec.square_m, spec.marker_m, dictionary,
                                 ids=np.arange(spec.first_marker_id, spec.first_marker_id+count, dtype=np.int32))
    board.setLegacyPattern(legacy)
    return board


class Target:
    def __init__(self, spec):
        self.spec = spec
        self.last_counts = (0, 0, 0)
        if spec.kind == 'charuco':
            self.board = charuco_board(spec)
            params = cv2.aruco.CharucoParameters(); params.tryRefineMarkers = True
            self.detector = cv2.aruco.CharucoDetector(self.board, params)

    def detect(self, rgb):
        if self.spec.kind == 'checkerboard':
            from .calibration import corners, board_points
            points = corners(rgb, self.spec.cols, self.spec.rows)
            if points is None: return None
            obj = board_points(self.spec.cols, self.spec.rows, self.spec.square_m)
            return obj, points, np.arange(len(obj), dtype=np.int32)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        points, ids, _, _ = self.detector.detectBoard(gray)
        # A small board in a wide full-body image can be missed even though its
        # native pixels are readable. Try overlapping native-resolution regions;
        # never resize or alter the camera calibration pixel coordinates.
        best_count = 0 if ids is None else int(np.count_nonzero(ids < (self.spec.cols-1)*(self.spec.rows-1)))
        if best_count < 12 and max(gray.shape) > 1000:
            h, w = gray.shape; cw, ch = (w+1)//2, (h*2+2)//3
            for y in sorted({0, (h-ch)//2, h-ch}):
                for x in sorted({0, (w-cw)//2, w-cw}):
                    candidate, candidate_ids, _, _ = self.detector.detectBoard(gray[y:y+ch, x:x+cw])
                    count = 0 if candidate_ids is None else int(np.count_nonzero(candidate_ids < (self.spec.cols-1)*(self.spec.rows-1)))
                    if count > best_count:
                        points = candidate.copy()
                        points[..., 0] += x; points[..., 1] += y
                        ids = candidate_ids; best_count = count
                    if best_count == (self.spec.cols-1)*(self.spec.rows-1): break
                if best_count == (self.spec.cols-1)*(self.spec.rows-1): break
        if ids is None or points is None: return None
        ids = ids.ravel(); order = np.argsort(ids); ids = ids[order]
        keep = ids < (self.spec.cols-1)*(self.spec.rows-1)
        ids = ids[keep]; points = points[order][keep]
        if not len(ids): return None
        return self.board.getChessboardCorners()[ids].astype(np.float32), points.astype(np.float32), ids

    def common(self, rgb0, rgb1, minimum=12):
        a, b = self.detect(rgb0), self.detect(rgb1)
        self.last_counts = (0 if a is None else len(a[2]),
                            0 if b is None else len(b[2]),
                            0 if a is None or b is None else len(np.intersect1d(a[2], b[2])))
        return self.match(a, b, minimum)

    def match(self, a, b, minimum=12):
        if a is None or b is None: return None
        ids, ia, ib = np.intersect1d(a[2], b[2], return_indices=True)
        if len(ids) < minimum: return None
        obj = a[0][ia]
        if np.linalg.matrix_rank(obj[:, :2]-obj[0, :2], tol=1e-6) < 2: return None
        if self.spec.kind == 'charuco' and np.any(np.ptp(obj[:, :2], axis=0) < self.spec.square_m*2): return None
        return obj, a[1][ia], b[1][ib], ids


def generate_charuco_svg(path, spec):
    """Vector squares and marker modules, exact physical dimensions, plus matching config."""
    board = charuco_board(spec)
    dictionary = board.getDictionary()
    square_mm = spec.square_m*1000
    margin = 15.; pattern_width = spec.cols*square_mm; pattern_height = spec.rows*square_mm
    width = pattern_width+2*margin; height = pattern_height+2*margin+10
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:g}mm" height="{height:g}mm" viewBox="0 0 {width:g} {height:g}">',
             f'<rect width="{width:g}" height="{height:g}" fill="white"/>']
    # Inspect OpenCV-generated square parity; even-row legacy boards differ.
    detector_rows = spec.rows + 1 if spec.top_left_white and spec.rows % 2 else spec.rows
    probe = board.generateImage((spec.cols*100, detector_rows*100), marginSize=0)[:spec.rows*100]
    for y in range(spec.rows):
        for x in range(spec.cols):
            if probe[y*100+5, x*100+5] < 128:
                lines.append(f'<rect x="{margin+x*square_mm:g}" y="{margin+y*square_mm:g}" width="{square_mm:g}" height="{square_mm:g}" fill="black"/>')
    modules = dictionary.markerSize+2
    for marker_id, vertices in zip(board.getIds().ravel(), board.getObjPoints()):
        if vertices[:,1].max() > spec.rows*spec.square_m: continue
        bits = cv2.aruco.generateImageMarker(dictionary, int(marker_id), modules)
        origin = vertices[0, :2]*1000; marker_mm = float(np.linalg.norm(vertices[1]-vertices[0])*1000)
        cell = marker_mm/modules
        for y in range(modules):
            for x in range(modules):
                if bits[y, x] < 128:
                    lines.append(f'<rect x="{margin+origin[0]+x*cell:.6f}" y="{margin+origin[1]+y*cell:.6f}" width="{cell:.6f}" height="{cell:.6f}" fill="black"/>')
    lines.append(f'<text x="15" y="{height-4:g}" font-size="3">ChArUco {spec.cols}x{spec.rows}; square {square_mm:g}mm; marker {spec.marker_m*1000:g}mm; {spec.dictionary}; legacy={spec.legacy}</text></svg>')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')
    write_json(path.with_suffix('.board.json'), dict(spec.to_dict(), page_size_mm=[width, height],
               pattern_size_mm=[pattern_width, pattern_height], marker_ids=[int(i) for i,v in zip(board.getIds().ravel(),board.getObjPoints()) if v[:,1].max() <= spec.rows*spec.square_m],
               note='Generated reference; purchased board must match marker layout, IDs and legacy parity.'))
    print(f'ChArUco SVG：{path}；图案 {pattern_width:g}×{pattern_height:g} mm；页面 {width:g}×{height:g} mm', flush=True)
