"""Prepare the existing safe capture owner for video/metadata evidence, offline."""
import argparse
import ast
import json
from pathlib import Path
from prepare_dw128_v2_board_ab import U02, sha
from pair_host_elf import verify_pair_host_elf


def generate():
    template = U02/'stereo/run_native_1080p_probe.py'
    source = template.read_text()
    replacements = [
        ('U02=Path(__file__).resolve().parents[1]', 'U02=Path('+repr(str(U02))+')'),
        ("    args=ap.parse_args()", """    ap.add_argument('--restored-owner',type=Path,required=True)
    ap.add_argument('--hwtimestamps',type=int,choices=(0,1),default=0)
    args=ap.parse_args()
    restored=json.loads((args.restored_owner/'state.json').read_text())
    from restored_board_session import load_restored_board_session
    restored_session=load_restored_board_session(args.restored_owner,'1de1bf6d4c13a9bf2be443554f142baabccb69e8464b608a9c14cece836861fb')
    assert args.candidate_module, 'Use the exact proven 1080p UVC module'"""),
        ("        module=args.candidate_module/'u02_uvc1080.ko';assert sha(module)==module_proof['module_sha256']", """        module=args.candidate_module/'u02_uvc1080.ko';assert sha(module)==module_proof['module_sha256']
        from prepare_dw128_v2_board_ab import verify_frozen
        fixed=U02/'milestones/uvc1080_metadata_fix_prepared_20261017';verify_frozen(fixed)
        assert sha(module)==sha(fixed/'u02_uvc1080.ko'), 'Metadata capture requires the verified queue-owner fix'"""),
        ("binary=U02/'inference/build_fast_nearest/stereo_1080p_host/test_dual_capture_1080p'",
         "binary=U02/'reports/uvc_timestamp_probe_prepared_20261017/uvc_timestamp_probe'"),
        ("'native_1080p_probe_'", "'uvc_timestamp_probe_'"),
        ('frames_requested_per_camera=90', 'frames_requested_per_camera=120'),
        ("        no_fds();state['modules_before']", "        assert state['cpu_before']['boot_id']==restored_session['boot_id']\n        no_fds();state['modules_before']"),
        ("'(?m)^u02_\\S+\\s'", "'(?m)^u02_(?!stream_resident\\s)\\S+\\s'"),
        ("./capture 90 1920 1080 --record-all", "./capture"),
        ("module_command('insmod '+remote+'/u02_uvc1080.ko payload_limit=2048')", "module_command('insmod '+remote+'/u02_uvc1080.ko payload_limit=2048 hwtimestamps='+str(args.hwtimestamps))"),
        ("        state['capture_process_uncertain']=True;save()", """        state['uvc_hwtimestamps']=checked('cat /sys/module/u02_uvc1080/parameters/hwtimestamps')
        assert re.search(r'(?m)^'+str(args.hwtimestamps)+r'\\r?$',state['uvc_hwtimestamps'])
        state['hwtimestamps_requested']=args.hwtimestamps
        state['capture_process_uncertain']=True;save()"""),
        ("        checked(f'cd {remote} && tar -czf diagnostics.tar.gz capture.log capture.exit capture.pid launcher.log kernel_before.log kernel_after.log')",
         "        checked(f'cd {remote} && tar -czf diagnostics.tar.gz capture.log capture.exit capture.pid launcher.log kernel_before.log kernel_after.log video0.ndjson video1.ndjson meta0.ndjson meta1.ndjson')"),
        ("        receive('diagnostics.tar.gz',4*1024**2)", """        raw=receive('diagnostics.tar.gz',8*1024**2)
        with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:
            names={'capture.log','capture.exit','capture.pid','launcher.log','kernel_before.log','kernel_after.log','video0.ndjson','video1.ndjson','meta0.ndjson','meta1.ndjson'}
            assert {m.name for m in t.getmembers()}==names
            for m in t.getmembers():
                assert m.isfile() and m.size<=4*1024**2
                (out/m.name).write_bytes(t.extractfile(m).read())"""),
        ("log.close();state['terminal']=True;save()", "log.close();state['terminal']=not(state['capture_process_uncertain'] or state['module_command_uncertain']);save()"),
    ]
    # The module exclusion occurs twice; all other replacements are single sites.
    for before, after in replacements:
        assert source.count(before)==(2 if before=="'(?m)^u02_\\S+\\s'" else 1), before
        source=source.replace(before,after)
    start=source.index("        if state['capture_exit_code']==0:")
    end=source.index("        state['cpu_after']=snapshot_cpu_state(c)",start)
    source=source[:start]+"""        if state['capture_exit_code']==0:
            assert 'U02_UVC_TIMESTAMP_PROBE_DONE' in state['capture_log']
            for side in (0,1):
                rows=[json.loads(line) for line in (out/f'video{side}.ndjson').read_text().splitlines()]
                assert len(rows)==120 and all((row['flags']&0xe000)==0x2000 for row in rows)
                assert all(b['sequence']>a['sequence'] and b['timestamp_us']>a['timestamp_us'] for a,b in zip(rows,rows[1:]))
            state['complete']=True
"""+source[end:]
    compile(source,'<uvc-timestamp-owner>','exec')
    original=ast.parse(template.read_text());generated=ast.parse(source)
    helpers=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main').body if isinstance(n,ast.FunctionDef)}
    a,b=helpers(original),helpers(generated)
    for name in ('checked','no_fds','module_command','receive'):assert a[name]==b[name],name
    return source


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    binary=U02/'reports/uvc_timestamp_probe_prepared_20261017/uvc_timestamp_probe'
    elf=verify_pair_host_elf(binary)
    source=generate();out=args.output.resolve();out.mkdir(exist_ok=False)
    (out/'run_probe.py').write_text(source)
    (out/'preparation.json').write_text(json.dumps(dict(board_accessed=False,elf=elf,binary_sha256=sha(binary),
        owner_sha256=sha(out/'run_probe.py'),source_sha256={str(p):sha(p) for p in (Path(__file__),U02/'inference/host/uvc_timestamp_probe.cpp',U02/'inference/host/camera_capture.cpp')}),indent=2)+'\n')
    print('UVC_TIMESTAMP_OWNER_PREPARED',out)


if __name__=='__main__':main()
