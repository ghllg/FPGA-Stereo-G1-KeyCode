"""Build a separate 1080p30-only H200S UVC reservation candidate in existing WSL.

Uses frozen source and the existing matching kernel build, with no kernel,
source-tree, config, installed driver, startup media or 720p artifact changes.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

U02=Path(__file__).resolve().parents[1]
frozen=U02/'milestones/stereo_mjpeg_candidate_prepared'
manifest=json.loads((frozen/'hashes.json').read_text())
for item in manifest:
    name=item['Path'].replace('\\','/')
    assert hashlib.sha256((frozen/name).read_bytes()).hexdigest()==item['Hash'].lower(),name
prior=json.loads((frozen/'state.json').read_text())
kernel=Path('/home/gby/axu5evb_e_boot/kernel-build')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(kernel/'.config')==prior['kernel_config_sha256']
assert sha(kernel/'Module.symvers')==prior['module_symvers_sha256']
assert 'UTS_RELEASE "6.6.40"' in (kernel/'include/generated/utsrelease.h').read_text()
out=U02/'reports'/('uvc1080_module_'+str(time.time_ns()));out.mkdir();work=out/'source';work.mkdir()
state=dict(complete=False,terminal=False,scope='1080p30 explicit candidate only; not capture acceptance',
    board_accessed=False,kernel_rebuilt=False,source_archive=str(frozen),module_name='u02_uvc1080',
    kernel_config_sha256=prior['kernel_config_sha256'],module_symvers_sha256=prior['module_symvers_sha256'])
def replace_one(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b)
try:
    for p in (frozen/'source').iterdir():
        if p.suffix in ('.c','.h') and not p.name.endswith('.mod.c'):shutil.copyfile(p,work/p.name)
    p=work/'uvc_video.c';s=p.read_text()
    s=replace_one(s,'frame->wWidth == 1280 && frame->wHeight == 720 &&','frame->wWidth == 1920 && frame->wHeight == 1080 &&')
    s=replace_one(s,'U02 experimental 720p30 MJPEG payload reservation','U02 experimental 1080p30 MJPEG payload reservation');p.write_text(s)
    p=work/'uvc_driver.c';s=p.read_text();s=replace_one(s,'"u02_uvcvideo"','"u02_uvc1080"')
    s=replace_one(s,'H200S 720p30 MJPEG','H200S 1080p30 MJPEG');p.write_text(s)
    p=work/'uvc_debugfs.c';s=replace_one(p.read_text(),'debugfs_create_dir("u02_uvcvideo", usb_debug_root)','debugfs_create_dir("u02_uvc1080", usb_debug_root)');p.write_text(s)
    s=(frozen/'source/Makefile').read_text().replace('u02_uvcvideo','u02_uvc1080');(work/'Makefile').write_text(s)
    state['source_sha256']={p.name:sha(p) for p in work.iterdir() if p.is_file()}
    with (out/'build.log').open('w') as log:
        p=subprocess.run(['make','-C',str(kernel),'M='+str(work),'ARCH=arm64','CROSS_COMPILE=aarch64-linux-gnu-','-j2','modules'],stdout=log,stderr=subprocess.STDOUT)
    assert p.returncode==0,(out/'build.log').read_text()[-4000:]
    module=work/'u02_uvc1080.ko';shutil.copyfile(module,out/module.name)
    info=subprocess.check_output(['modinfo',str(module)],text=True);(out/'modinfo.txt').write_text(info)
    assert '6.6.40' in info and re.findall(r'(?m)^alias:\s+(\S+)',info)==['usb:v10BBp2B08d*dc*dsc*dp*ic0Eisc01ip00in*']
    assert sha(kernel/'.config')==prior['kernel_config_sha256'] and sha(kernel/'Module.symvers')==prior['module_symvers_sha256']
    state.update(complete=True,module_sha256=sha(module))
finally:
    state['terminal']=True;(out/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    print('UVC1080_MODULE_RECORD',out,flush=True)
