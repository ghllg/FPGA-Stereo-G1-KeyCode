"""Review actual causal replay poses on their original time axis; never repair gaps."""
import json
from pathlib import Path
import av
import cv2
import numpy as np
from mocap.io import read_json
from mocap.lafan import PARENTS
from mocap.preview import EDGES


def make_review(folder, session, output, fps=15):
    folder, session, output = Path(folder), Path(session), Path(output)
    if output.exists(): raise ValueError('对照视频已存在。')
    poses=[json.loads(x) for x in (folder/'pose2d.jsonl').read_text(encoding='utf-8').splitlines()]
    live=[json.loads(x) for x in (folder/'live_lafan.jsonl').read_text(encoding='utf-8').splitlines()]
    if not poses or len(poses)!=len(live): raise ValueError('缺少对应姿态记录。')
    selected=[{},{}]
    for c,side in enumerate(('left','right')):
        wanted={row['views'][c]['frame_index'] for row in poses}
        with av.open(str(session/(side+'.mkv'))) as video:
            index=-1
            for packet in video.demux(video=0):
                if not packet.size: continue
                index+=1
                if index not in wanted: continue
                im=cv2.imdecode(np.frombuffer(bytes(packet),np.uint8),cv2.IMREAD_REDUCED_COLOR_4)
                if im is None: raise ValueError(f'对照源帧 {side}/{index} 解码失败。')
                selected[c][index]=cv2.resize(im,(480,270))
                if len(selected[c])==len(wanted): break
    times=np.array([(p['timestamp_ns']-poses[0]['timestamp_ns'])/1e9 for p in poses])
    writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1440,340))
    if not writer.isOpened(): raise RuntimeError('无法写入预览视频。')
    centers=[np.array(r['fk_positions_cm'])[0] for r in live if r['valid']]
    center=centers[0] if centers else np.zeros(3)
    try:
        for t in np.arange(int(times[-1]*fps)+1)/fps:
            i=max(0,int(np.searchsorted(times,t,side='right'))-1)
            row,pose=poses[i],live[i]; stale=t-times[i]>.25
            panels=[]
            for c in range(2):
                view=row['views'][c]; image=selected[c][view['frame_index']].copy()
                scale=np.array([480,270])/np.array(view['image_size'])
                points={j:tuple(np.int32(np.array(p['xy'])*scale)) for j,p in enumerate(view['joints']) if p['valid']}
                for a,b in EDGES:
                    if a in points and b in points: cv2.line(image,points[a],points[b],(0,180,0),1)
                for xy in points.values(): cv2.circle(image,xy,2,(0,255,0),-1)
                cv2.putText(image,f'CAM{c} original frame {view["frame_index"]}',(8,20),0,.48,(0,255,255),1)
                panels.append(image)
            panel=np.full((270,480,3),25,np.uint8)
            if pose['valid'] and not stale:
                p=np.array(pose['fk_positions_cm'])-center
                xy=np.column_stack([p[:,0]+.35*p[:,2],-p[:,1]+.15*p[:,2]])*1.35+[240,150]
                for j,parent in enumerate(PARENTS):
                    if parent>=0:cv2.line(panel,tuple(np.int32(xy[parent])),tuple(np.int32(xy[j])),(170,220,40),2)
                text='Actual causal LAFAN1 pose'
            else:text='NO VALID LAFAN POSE' if not stale else 'STALE INPUT - pose hidden'
            cv2.putText(panel,text,(8,20),0,.5,(0,160,255),1)
            cv2.putText(panel,f'valid={pose["valid"]}; {pose["reason"]}',(8,255),0,.38,(180,180,180),1)
            panels.append(panel)
            canvas=np.full((340,1440,3),20,np.uint8);canvas[:270]=np.hstack(panels)
            timing=row['timing']
            cv2.putText(canvas,f't={t:.2f}s  processing={timing["processing_ms"]:.1f}ms  pair delta={timing["pair_delta_ms"]:.2f}ms',
                        (12,294),0,.52,(230,230,230),1)
            cv2.putText(canvas,'OFFLINE CONTENT-ALIGNED REPLAY / actual causal poses / gaps not repaired / no ground truth',
                        (12,321),0,.52,(0,210,255),1)
            writer.write(canvas)
    finally:writer.release()
    return str(output)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--session',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(make_review(a.folder,a.session,a.output))
