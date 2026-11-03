"""One bounded log-quadratic decoder candidate on saved actual board heatmaps.

Offline diagnostic only; no neural inference, no replacement of accepted poses.
"""
import copy
import json
import math
from pathlib import Path
import time
import numpy as np
from reconstruct_board_stereo import BoardStereoReconstructor,load_calibration

U02=Path(__file__).resolve().parents[1]

def refine_heatmap(heat,roi):
    assert heat.shape==(17,32,24) and np.isfinite(heat).all()
    points=[];refined=0;fallback=0
    for plane in heat:
        best=int(np.argmax(plane));y,x=divmod(best,24)
        dx=dy=0.
        if 0<x<23 and 0<y<31:
            dx=float(np.sign(plane[y,x+1]-plane[y,x-1]))*.25
            dy=float(np.sign(plane[y+1,x]-plane[y-1,x]))*.25
            patch=plane[y-1:y+2,x-1:x+2].astype(np.float64)
            if np.all(patch>1e-10):
                log=np.log(patch)
                gx=(log[1,2]-log[1,0])*.5;gy=(log[2,1]-log[0,1])*.5
                xx=log[1,2]-2*log[1,1]+log[1,0]
                yy=log[2,1]-2*log[1,1]+log[0,1]
                xy=(log[2,2]-log[0,2]-log[2,0]+log[0,0])*.25
                determinant=xx*yy-xy*xy
                if xx<0 and yy<0 and determinant>1e-8:
                    ox=(-yy*gx+xy*gy)/determinant
                    oy=(xy*gx-xx*gy)/determinant
                    if math.isfinite(ox+oy) and max(abs(ox),abs(oy))<=1:
                        dx,dy=ox,oy;refined+=1
                    else:fallback+=1
                else:fallback+=1
            else:fallback+=1
        else:fallback+=1
        points.append([float(np.float32(roi[0]+(x+dx)*roi[2]/24)),
                       float(np.float32(roi[1]+(y+dy)*roi[3]/32)),float(plane[y,x])])
    return points,refined,fallback

def main():
    output=U02/('reports/stereo_peak_refinement_'+str(time.time_ns()));output.mkdir()
    measurements={}
    for name,number in [('raw','1791803053839887700'),('content','1791804056937887000')]:
        record=U02/('reports/stereo_recorded_board_'+number)
        result_path=record/('recovered_result.json' if name=='raw' else 'result.json')
        result=json.loads(result_path.read_text(encoding='utf-8'));assert result['complete']
        inputs=json.loads((Path(result['inputs'])/'manifest.json').read_text(encoding='utf-8'))
        packets=[json.loads(line) for line in (record/'board_pose.jsonl').read_text(encoding='utf-8').splitlines()]
        calibration=load_calibration(Path(result['inputs'])/'calibration.json')
        processors={kind:BoardStereoReconstructor(calibration) for kind in ('accepted_quarter','bounded_log_quadratic')}
        counts={kind:[] for kind in processors};refined=fallback=0;diagnostic=[]
        for packet,image in zip(packets,inputs['images']):
            heat=np.fromfile(record/('pl_stream_'+image['file'][:-4]+'.json.heatmap'),dtype='<f4').reshape(17,32,24)
            points,good,bad=refine_heatmap(heat,packet['roi_xywh']);refined+=good;fallback+=bad
            assert all(a[2]==float(np.float32(b[2])) for a,b in zip(points,packet['keypoints']))
            for kind,processor in processors.items():
                copy_packet=copy.deepcopy(packet)
                if kind=='bounded_log_quadratic':copy_packet['keypoints']=points
                skeleton=processor.push(copy_packet)
                if skeleton is not None:
                    counts[kind].append(sum(p['valid'] for p in skeleton['joints']))
            diagnostic.append(dict(file=image['file'],points=points,refined=good,fallback=bad))
        measurements[name]=dict(valid_joints={kind:sum(values) for kind,values in counts.items()},
            per_pair_valid=counts,refined_points=refined,fallback_points=fallback,total_points=204)
        (output/(name+'_diagnostic_points.json')).write_text(json.dumps(diagnostic,indent=2)+'\n')
    report=dict(complete=True,hardware_deployed=False,neural_network_outputs_modified=False,
        diagnostic_only=True,confidence_and_highest_peak_preserved=True,
        measurements=measurements,ground_truth_available=False,
        scope='Single decoder candidate evaluated on saved actual PL heatmaps; geometric gates unchanged. Offline result is not a new board Host, physical accuracy or performance acceptance.')
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(measurements,indent=2));print('PEAK_REFINEMENT_RECORD',output)

if __name__=='__main__':main()
