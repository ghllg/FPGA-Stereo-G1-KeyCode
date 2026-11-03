"""Freeze lossless source images and OpenCV preprocessing inputs; no inference."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / 'reports/image_preprocess_source_images'

def main():
    assert not OUT.exists() or not any(OUT.iterdir()), 'Preserve existing fixtures'
    original = json.loads((ROOT/'board_light_pose/experiments/v11_static_quant_streaming/outputs/dataset/manifest.json').read_text(encoding='utf-8'))
    source = Path(original['source'])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original['source_sha256']
    OUT.mkdir(exist_ok=True)
    records = []
    def save(index, bgr, roi, previous=None):
        image = OUT/f'{index}.png'
        assert cv2.imwrite(str(image), bgr)
        transform = np.array([[96/roi[2], 0, -roi[0]*96/roi[2]], [0, 128/roi[3], -roi[1]*128/roi[3]]], np.float32)
        crop = cv2.warpAffine(bgr, transform, (96,128), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
        rgb = cv2.cvtColor(crop,cv2.COLOR_BGR2RGB).astype(np.float32)/255
        tensor = ((rgb-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)).transpose(2,0,1).astype('<f4')
        expected = OUT/f'{index}.expected.f32'
        tensor.tofile(expected)
        legacy_exact = previous is not None and expected.read_bytes() == previous.read_bytes()
        legacy_error = None if previous is None else float(np.max(np.abs(tensor.ravel()-np.fromfile(previous,dtype='<f4'))))
        records.append(dict(index=index, source_size=[bgr.shape[1],bgr.shape[0]], roi_xywh=roi,
            image=image.name, expected=expected.name, image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            expected_sha256=hashlib.sha256(expected.read_bytes()).hexdigest(), legacy_input_byte_exact=legacy_exact,
            max_abs_vs_legacy_input=legacy_error))
    cap = cv2.VideoCapture(str(source))
    assert cap.isOpened()
    wanted = {120+30*i:i for i in range(10)}
    for frame in range(391):
        ok,bgr = cap.read()
        assert ok,frame
        if frame in wanted:
            index = wanted[frame]
            meta = json.loads((ROOT/f'board_light_pose/outputs/samples/{index}.json').read_text(encoding='utf-8'))
            save(index,bgr,meta['roi_xywh'],ROOT/f'board_light_pose/outputs/samples/{index}.f32')
    cap.release()
    rng = np.random.default_rng(20261010)
    for index,roi in enumerate([[-7.3,3.7,43.8,58.4],[.137,-8.27,79.3,105.733333],
        [1000,-1000,50,66.666667],[0,0,113,79],[13.73,19.29,2.7,3.6]],10):
        save(index,rng.integers(0,256,size=(79,113,3),dtype=np.uint8),roi)
    (OUT/'manifest.json').write_text(json.dumps(dict(source=str(source),source_sha256=original['source_sha256'],
        records=records,random_seed=20261010,scope='Decode/affine/normalization oracle, no PC model inference; newly decoded images are not assumed to reproduce legacy tensors'),indent=2))
    print('U02_IMAGE_FIXTURES_READY real=10 affine_border=5')

if __name__ == '__main__':
    main()
