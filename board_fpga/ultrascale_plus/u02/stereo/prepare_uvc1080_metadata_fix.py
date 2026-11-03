"""Build only an isolated UVC module with the metadata STREAMOFF owner fixed."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

U02=Path(__file__).resolve().parents[1]
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
original=U02/'milestones/stereo1080_capture_decode_verified_20261016/module'
prior=json.loads((original/'state.json').read_text())
assert prior['complete'] and prior['terminal']
assert sha(original/'u02_uvc1080.ko')==prior['module_sha256']
kernel=Path('/home/gby/axu5evb_e_boot/kernel-build')
assert sha(kernel/'.config')==prior['kernel_config_sha256']
assert sha(kernel/'Module.symvers')==prior['module_symvers_sha256']
out=U02/'reports'/('uvc1080_metadata_fix_'+str(time.time_ns()));work=out/'source';work.mkdir(parents=True)
state=dict(complete=False,terminal=False,module_name='u02_uvc1080',board_accessed=False,
           kernel_rebuilt=False,formal_promoted=False,source_module_sha256=prior['module_sha256'],
           kernel_config_sha256=prior['kernel_config_sha256'],module_symvers_sha256=prior['module_symvers_sha256'])
(out/'executed_script.py').write_bytes(Path(__file__).read_bytes())
try:
    for name,expected in prior['source_sha256'].items():
        path=original/'source'/name;assert sha(path)==expected,name
        shutil.copyfile(path,work/name)
    path=work/'uvc_queue.c';source=path.read_text()
    before='''static void uvc_stop_streaming(struct vb2_queue *vq)
{
	struct uvc_video_queue *queue = vb2_get_drv_priv(vq);
	struct uvc_streaming *stream = uvc_queue_to_stream(queue);'''
    after='''static void uvc_stop_streaming(struct vb2_queue *vq)
{
	struct uvc_video_queue *queue = vb2_get_drv_priv(vq);
	/* Metadata owns stream->meta.queue, not stream->queue. */
	struct uvc_streaming *stream = vq->type == V4L2_BUF_TYPE_META_CAPTURE
		? container_of(queue, struct uvc_streaming, meta.queue)
		: uvc_queue_to_stream(queue);'''
    assert source.count(before)==1
    path.write_text(source.replace(before,after))
    state['source_sha256']={p.name:sha(p) for p in work.iterdir() if p.is_file()}
    with (out/'build.log').open('w') as log:
        result=subprocess.run(['make','-C',str(kernel),'M='+str(work),'ARCH=arm64','CROSS_COMPILE=aarch64-linux-gnu-','-j2','modules'],stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0,(out/'build.log').read_text()[-4000:]
    module=work/'u02_uvc1080.ko';shutil.copyfile(module,out/module.name)
    info=subprocess.check_output(['modinfo',str(module)],text=True);(out/'modinfo.txt').write_text(info)
    assert '6.6.40' in info
    assert sha(kernel/'.config')==prior['kernel_config_sha256'] and sha(kernel/'Module.symvers')==prior['module_symvers_sha256']
    state.update(complete=True,module_sha256=sha(module),capture_or_cleanup_verified=False)
finally:
    state['terminal']=True;(out/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    print('UVC1080_METADATA_FIX_BUILD',out,flush=True)
