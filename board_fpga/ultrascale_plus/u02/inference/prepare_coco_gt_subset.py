"""Fetch a bounded, deterministic COCO keypoint diagnostic subset; no board access.

Only the validation annotation member is fetched with HTTP ranges. This is an
oracle person-box diagnostic, not COCO AP or a person-detector evaluation.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import struct
import urllib.request
import zlib

URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
DOWNLOAD_SOURCE = 'https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url, limit, start=None, end=None):
    headers = {} if start is None else {'Range': f'bytes={start}-{end}'}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
        if start is not None:
            assert response.status == 206, 'Server ignored range; refuse whole archive'
            assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
        raw = response.read(limit + 1)
        assert len(raw) <= limit, 'Download exceeds bounded size'
        if start is not None:
            assert len(raw) == end - start + 1
        return raw


def validation_annotations():
    with urllib.request.urlopen(urllib.request.Request(URL, method='HEAD'), timeout=30) as response:
        size = int(response.headers['Content-Length'])
    assert 65536 < size < 300_000_000
    tail = fetch(URL, 65536, size - 65536, size - 1)
    pos = tail.rfind(b'PK\x05\x06')
    assert pos >= 0
    end = struct.unpack_from('<4s4H2IH', tail, pos)
    assert end[1] == end[2] == 0 and end[3] == end[4] and end[5] < 65536
    directory = fetch(URL, 65536, end[6], end[6] + end[5] - 1)
    offset = 0
    for _ in range(end[4]):
        entry = struct.unpack_from('<4s6H3I5H2I', directory, offset)
        assert entry[0] == b'PK\x01\x02'
        name = directory[offset + 46:offset + 46 + entry[10]].decode('utf8')
        offset += 46 + sum(entry[10:13])
        if name != 'annotations/person_keypoints_val2017.json':
            continue
        assert entry[3] & 1 == 0 and entry[4] == 8
        crc, compressed, expanded, local = entry[7], entry[8], entry[9], entry[16]
        assert compressed < 16_000_000 and expanded < 64_000_000
        header = fetch(URL, 30, local, local + 29)
        fields = struct.unpack('<4s5H3I2H', header)
        assert fields[0] == b'PK\x03\x04'
        data_offset = local + 30 + fields[9] + fields[10]
        packed = fetch(URL, compressed, data_offset, data_offset + compressed - 1)
        decoder = zlib.decompressobj(-15)
        raw = decoder.decompress(packed, expanded + 1)
        assert decoder.eof and not decoder.unused_data and len(raw) == expanded
        assert zlib.crc32(raw) == crc
        return raw, dict(url=URL, archive_bytes=size, member=name,
                        compressed_bytes=compressed, expanded_bytes=expanded,
                        annotation_sha256=digest(raw), crc32=f'{crc:08x}')
    raise RuntimeError('Validation keypoints member missing')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=20)
    args = parser.parse_args()
    assert 2 <= args.count <= 40 and not args.output.exists()
    assert shutil.disk_usage(args.output.parent).free > 10 * 1024**3
    args.output.mkdir()
    raw, source = validation_annotations()
    data = json.loads(raw)
    images = {item['id']: item for item in data['images']}
    people = defaultdict(list)
    for item in data['annotations']:
        if item['category_id'] == 1 and not item['iscrowd'] and item['num_keypoints']:
            people[item['image_id']].append(item)
    candidates = []
    for image_id, instances in people.items():
        if len(instances) != 1:
            continue
        ann = instances[0]
        x, y, w, h = ann['bbox']
        visible = sum(v == 2 for v in ann['keypoints'][2::3])
        if ann['num_keypoints'] >= 12 and visible >= 8 and min(w, h) >= 96 and ann['area'] >= 5000:
            candidates.append(ann)
    candidates.sort(key=lambda a: (a['image_id'], a['id']))
    assert len(candidates) >= args.count
    selected = [candidates[i * (len(candidates) - 1) // (args.count - 1)] for i in range(args.count)]
    result = dict(complete=False, source=source, official_download_source=DOWNLOAD_SOURCE,
                  selection='Evenly spaced sorted image IDs; exactly one labeled noncrowd person; >=12 labeled and >=8 visible joints; bbox min dimension >=96; area>=5000',
                  candidate_count=len(candidates), inference_performed=False, calibration_used=False,
                  scope='Small oracle-box GT diagnostic; not COCO AP, detector evaluation, or deployment accuracy acceptance',
                  category=data['categories'][0], licenses=data['licenses'], samples=[])
    (args.output/'person_keypoints_val2017.json').write_bytes(raw)
    for ann in selected:
        image = images[ann['image_id']]
        name = image['file_name']
        assert Path(name).name == name and name.endswith('.jpg')
        jpeg = fetch(image['coco_url'], 3_000_000)
        assert jpeg.startswith(b'\xff\xd8') and jpeg.endswith(b'\xff\xd9')
        (args.output/name).write_bytes(jpeg)
        x, y, w, h = ann['bbox']
        # Expand the GT box to 3:4, with 25% margin, preserving its center.
        rh = max(h, w * 4 / 3) * 1.25
        rw = rh * 3 / 4
        roi = [x + w / 2 - rw / 2, y + h / 2 - rh / 2, rw, rh]
        result['samples'].append(dict(image=image, annotation=ann, image_sha256=digest(jpeg), roi_xywh=roi))
        print('COCO_GT_SAMPLE_SAVED', name, len(jpeg), flush=True)
    result['complete'] = True
    (args.output/'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print('COCO_GT_SUBSET_READY', args.output, flush=True)


if __name__ == '__main__':
    main()
