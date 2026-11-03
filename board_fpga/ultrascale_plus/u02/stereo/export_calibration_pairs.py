"""Export original 720p/1080p JPEG pairs and source identities for PC calibration.

No inference, scaling, image recompression, calibration or exposure-sync claim.
Uses only the Python standard library. Header checks do not prove full decode.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def jpeg_size(data):
    if data[:2]!=b'\xff\xd8' or data[-2:]!=b'\xff\xd9':raise ValueError('Incomplete JPEG markers')
    i=2
    while i<len(data):
        if data[i]!=255:raise ValueError('Invalid JPEG header marker')
        while i<len(data) and data[i]==255:i+=1
        if i>=len(data):break
        marker=data[i];i+=1
        if marker==0xDA:break
        if marker in (0x01,0xD8) or 0xD0<=marker<=0xD7:continue
        if i+2>len(data):break
        size=int.from_bytes(data[i:i+2],'big')
        if size<2 or i+size>len(data):raise ValueError('Invalid JPEG segment length')
        if marker in (0xC0,0xC1,0xC2):
            if size<8:raise ValueError('Invalid JPEG frame header')
            return int.from_bytes(data[i+5:i+7],'big'),int.from_bytes(data[i+3:i+5],'big')
        i+=size
    raise ValueError('Supported JPEG frame header not found')


def pair_frames(left,right,limit):
    i=j=0;pairs=[]
    while i<len(left) and j<len(right):
        a,b=left[i],right[j];delta=int(b['source_timestamp_us'])-int(a['source_timestamp_us'])
        if abs(delta)<=limit:pairs.append((a,b,delta));i+=1;j+=1
        elif delta>0:i+=1
        else:j+=1
    return pairs


def export(source,output,mode,limit=10000):
    assert 0<limit<=10000
    source=Path(source).resolve();output=Path(output).resolve()
    assert not output.exists(),'Use a new output directory'
    expected=(1920,1080) if mode=='1080p' else (1280,720) if mode=='720p' else None
    assert expected is not None
    rows=[];hashes={}
    for side in ('left','right'):
        path=source/(side+'.csv');hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(newline='') as stream:values=list(csv.DictReader(stream))
        assert values
        for key in ('source_timestamp_us','sequence'):
            assert all(int(b[key])>int(a[key]) for a,b in zip(values,values[1:])),key
        assert len({row['capture_index'] for row in values})==len(values)
        for row in values:assert row['capture_index'].isdigit()
        rows.append(values)
    pairs=pair_frames(*rows,limit);assert pairs,'No source timestamp pairs inside limit'
    manifest=[];copies=[]
    for index,(left,right,delta) in enumerate(pairs):
        entry=dict(pair_id=index,pair_delta_us=delta,left=left,right=right,images={})
        for side,row in (('left',left),('right',right)):
            src=source/(side+'_'+row['capture_index']+'.jpg');raw=src.read_bytes()
            assert jpeg_size(raw)==expected,'JPEG dimensions differ from explicitly selected capture mode'
            assert len(raw)==int(row['jpeg_bytes']),'CSV/JPEG size mismatch'
            target=side+f'/{index:06d}.jpg';digest=hashlib.sha256(raw).hexdigest()
            entry['images'][side]=dict(path=target,sha256=digest,source_file=src.name,bytes=len(raw))
            copies.append((src,target,digest))
        manifest.append(entry)
    output.mkdir(parents=True)
    for side in ('left','right'):(output/side).mkdir()
    for src,target,digest in copies:
        dst=output/target;shutil.copyfile(src,dst);assert hashlib.sha256(dst.read_bytes()).hexdigest()==digest
    result=dict(schema='u02_calibration_original_pairs_v1',source_size=list(expected),
        source_directory=str(source),coordinate_space='full_source_image_pixels',
        timestamp_source='v4l2_monotonic_unverified_exposure',exposure_sync_verified=False,
        reencoded=False,resized=False,all_images_fully_decoded_by_exporter=False,
        max_pair_delta_us=limit,source_csv_sha256=hashes,pairs=manifest,
        instruction='Select sharp stationary calibration-board frames; software timestamp pairing does not prove simultaneous exposures. Calibrate in the same capture mode and final camera arrangement.')
    (output/'pairs.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('capture_directory',type=Path);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--source-mode',choices=('720p','1080p'),required=True)
    ap.add_argument('--max-delta-us',type=int,default=10000);a=ap.parse_args()
    result=export(a.capture_directory,a.output,a.source_mode,a.max_delta_us)
    print('EXPORTED_ORIGINAL_PAIRS',len(result['pairs']),a.output)


if __name__=='__main__':main()
