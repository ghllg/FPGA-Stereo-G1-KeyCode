"""Import timestamped stereo clips for identical-input board comparison."""
import csv
from pathlib import Path
import shutil
import cv2
from .core import read_json,write_json
from .services import new_session,snapshot_assets,import_capture


def import_videos(files,config,progress,cancel):
    path=new_session('导入双目视频');raw=path/'raw';raw.mkdir();state=read_json(path/'session.json')
    state.update(assets=snapshot_assets(path,config),roi_xywh=dict(left=config['left_roi'],right=config['right_roi']),imported_media=True)
    write_json(path/'session.json',state)
    try:
        total_bytes=0
        for side in ('left','right'):
            video=Path(files[side]);timestamps=Path(files[side+'_timestamps'])
            with timestamps.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
            if not 2<=len(rows)<=9000:raise ValueError('每路需要 2–9000 条时间戳记录')
            stamps=[int(r['timestamp_us']) for r in rows];indices=[int(r['frame_index']) for r in rows]
            if stamps[0]<0 or indices[0]<0 or any(b<=a for a,b in zip(stamps,stamps[1:])) or any(b<=a for a,b in zip(indices,indices[1:])):
                raise ValueError('帧号和同一时钟下的时间戳必须严格递增')
            if stamps[-1]-stamps[0]>300_000_000:raise ValueError('视频片段不得超过 300 秒')
            cap=cv2.VideoCapture(str(video))
            try:
                if not cap.isOpened():raise ValueError('无法打开 '+str(video))
                with (raw/(side+'.csv')).open('w',encoding='utf-8',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=['capture_index','sequence','source_timestamp_us','jpeg_bytes']);writer.writeheader()
                    for i,(frame_index,stamp) in enumerate(zip(indices,stamps)):
                        if cancel.is_set():raise RuntimeError('导入已取消')
                        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index);ok,frame=cap.read()
                        if not ok:raise ValueError(f'{side} 无法解码第 {frame_index} 帧')
                        if frame.shape[:2]!=(1080,1920):raise ValueError('模型输入需要原始 1920×1080 视频，不能缩放后沿用原标定')
                        ok,jpeg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,95])
                        if not ok:raise ValueError('JPEG 编码失败')
                        total_bytes+=jpeg.nbytes
                        if total_bytes>int(config['max_mib'])*1024**2:raise ValueError('解码图像超过配置的空间上限')
                        jpeg.tofile(str(raw/f'{side}_{i}.jpg'))
                        writer.writerow(dict(capture_index=i,sequence=frame_index,source_timestamp_us=stamp,jpeg_bytes=jpeg.nbytes))
                        if i%30==0:progress(f'导入 {side} {i+1}/{len(rows)}')
            finally:cap.release()
            shutil.copy2(timestamps,path/(side+'_source_timestamps.csv'))
        import_capture(path,int(config['pair_limit_us']));return str(path)
    except Exception as error:
        state.update(state='failed',error=str(error));write_json(path/'session.json',state);raise
