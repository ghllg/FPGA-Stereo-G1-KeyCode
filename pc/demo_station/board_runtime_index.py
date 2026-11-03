"""Index completed on-board qualification reports locally; never contacts hardware."""
import argparse
import json
from pathlib import Path
import re

from .core import HOME, TRAIN, digest, read_json


def create_index(board_record, output):
    directory=Path(board_record).resolve();output=Path(output).resolve()
    names=('live_stage','run','preflight','verification')
    paths={name:directory/(name+'.json') for name in names}
    reports={name:read_json(path) for name,path in paths.items()}
    stage,run,preflight,verification=(reports[name] for name in names)
    boot=stage.get('boot_id','')
    if not re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',boot):
        raise ValueError('现场报告缺少有效 boot_id')
    if not all(v.get('complete') is True and v.get('boot_id')==boot for v in reports.values()):
        raise ValueError('四份报告尚未完成或不是同一次启动')
    if (run.get('remote')!=stage.get('remote') or preflight.get('remote')!=stage.get('remote')
            or run.get('dma_ambiguous') is not False or run.get('dma_abi')!=9
            or run.get('actual_PL_calls',0)<=0 or run.get('all_full_heatmaps_exact') is not True
            or run.get('cleanup_verified') is not True or preflight.get('cleanup_verified') is not True
            or verification.get('CHW_programmed_and_idle_verified') is not True
            or verification.get('TCP_listener_no_camera_preflight_passed') is not True):
        raise ValueError('真实 PL、无 DMA 歧义、完整热图、清理或无相机预检资格未通过')
    formal=read_json(TRAIN.parent/'board_resume_20261023_112300/live_stage.json')['files']
    for name in ('stereo_host','model.stream.param','model.opt.bin','u02_stream_pipeline.ko','u02_uvc1080.ko'):
        if stage.get('files',{}).get(name)!=formal[name]:raise ValueError('正式制品 SHA 不符：'+name)
    value=dict(schema='board_runtime_current/1',evidence={name:dict(path=str(path),sha256=digest(path)) for name,path in paths.items()},
               scope='Local index only. No board was contacted; the UI must verify the actual current boot and runtime before starting.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--board-record',type=Path,required=True,help='New completed qualification directory; never edit old reports to claim a new boot')
    parser.add_argument('--output',type=Path,default=HOME/'data/board_runtime_current.json',help='New file only; existing files are never overwritten')
    args=parser.parse_args()
    create_index(args.board_record,args.output)
    print('本地索引已生成：'+str(args.output.resolve()))
    print('尚未连接开发板。UI 将再次核对真实当前 boot；旧报告不能证明本次现场资格。')


if __name__=='__main__':main()
