"""Prepare an isolated, explicitly enabled H200S MJPEG negotiation candidate.

Run in the existing WSL build environment. Does not install/load a module,
rebuild the kernel, or edit the matching kernel source/build configuration.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

U02=Path(__file__).resolve().parents[1]
source=Path('/home/gby/axu5evb_e_boot/src/linux-xlnx-xlnx_rebase_v6.6_LTS_2024.2/drivers/media/usb/uvc')
kernel=Path('/home/gby/axu5evb_e_boot/kernel-build')
output=U02/'reports/uvc_mjpeg_candidate_module'
work=output/'source'
assert not output.exists()
assert 'UTS_RELEASE "6.6.40"' in (kernel/'include/generated/utsrelease.h').read_text()
assert (kernel/'Module.symvers').stat().st_size>0
assert 'CONFIG_USB_VIDEO_CLASS=y' in (kernel/'.config').read_text()
output.mkdir();work.mkdir()
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original={p.name:digest(p) for p in source.iterdir() if p.suffix in ('.c','.h')}
assert original and not any('EXPORT_SYMBOL' in (source/name).read_text() for name in original)
for name in original:shutil.copyfile(source/name,work/name)
state=dict(complete=False,kernel_release='6.6.40',kernel_rebuilt=False,module_loaded=False,
    kernel_config_sha256=digest(kernel/'.config'),module_symvers_sha256=digest(kernel/'Module.symvers'),
    original_source_sha256=original,preparer_sha256=digest(Path(__file__)),
    scope='Only VID10bb/PID2b08, MJPEG 1280x720 at frame interval333333, explicit payload_limit=2048. Candidate negotiation, not capture acceptance.')
def replace_one(text,old,new):
    assert text.count(old)==1,old[:80]
    return text.replace(old,new)
try:
    p=work/'uvc_driver.c';text=p.read_text()
    text=replace_one(text,'unsigned int uvc_dbg_param;',
        'unsigned int uvc_dbg_param;\nunsigned int u02_mjpeg_payload_limit;')
    text=replace_one(text,'module_param_named(timeout, uvc_timeout_param, uint, 0644);',
        'module_param_named(payload_limit, u02_mjpeg_payload_limit, uint, 0444);\n'
        'MODULE_PARM_DESC(payload_limit, "Experimental H200S 720p30 MJPEG payload reservation: 0 (unchanged) or 2048");\n'
        'module_param_named(timeout, uvc_timeout_param, uint, 0644);')
    begin=text.index('static const struct uvc_device_info uvc_ctrl_power_line_limited = {')
    end=text.index('MODULE_DEVICE_TABLE(usb, uvc_ids);',begin)+len('MODULE_DEVICE_TABLE(usb, uvc_ids);')
    text=text[:begin]+'''/* This external candidate matches only the two verified H200S devices. */
static const struct usb_device_id uvc_ids[] = {
    { USB_DEVICE_AND_INTERFACE_INFO(0x10bb, 0x2b08, USB_CLASS_VIDEO, 1, 0) },
    {}
};
MODULE_DEVICE_TABLE(usb, uvc_ids);'''+text[end:]
    text=replace_one(text,'.name\t\t= "uvcvideo",','.name\t\t= "u02_uvcvideo",')
    text=replace_one(text,'\tuvc_debugfs_init();',
        '\tif (u02_mjpeg_payload_limit && u02_mjpeg_payload_limit != 2048)\n'
        '\t\treturn -EINVAL;\n\n\tuvc_debugfs_init();')
    p.write_text(text)
    p=work/'uvcvideo.h';p.write_text(p.read_text()+'\nextern unsigned int u02_mjpeg_payload_limit;\n')
    p=work/'uvc_debugfs.c';p.write_text(replace_one(p.read_text(),
        'debugfs_create_dir("uvcvideo", usb_debug_root)',
        'debugfs_create_dir("u02_uvcvideo", usb_debug_root)'))
    p=work/'uvc_video.c';text=p.read_text()
    boundary='\n}\n\nstatic size_t uvc_video_ctrl_size'
    text=replace_one(text,boundary,'''

    /* Explicit opt-in diagnostic for the verified camera and required mode.
     * This changes reservation negotiation, not JPEG data or frame timestamps.
     * Cameras must accept COMMIT and deliver complete decoded frames before
     * the setting can be accepted. Original driver is preserved for rollback.
     */
    if (u02_mjpeg_payload_limit == 2048 &&
        le16_to_cpu(stream->dev->udev->descriptor.idVendor) == 0x10bb &&
        le16_to_cpu(stream->dev->udev->descriptor.idProduct) == 0x2b08 &&
        format->fcc == V4L2_PIX_FMT_MJPEG &&
        frame->wWidth == 1280 && frame->wHeight == 720 &&
        ctrl->dwFrameInterval == 333333 &&
        ctrl->dwMaxPayloadTransferSize > u02_mjpeg_payload_limit) {
        uvc_dbg(stream->dev, VIDEO,
            "U02 experimental 720p30 MJPEG payload reservation %u -> %u\\n",
            ctrl->dwMaxPayloadTransferSize, u02_mjpeg_payload_limit);
        ctrl->dwMaxPayloadTransferSize = u02_mjpeg_payload_limit;
    }
}

static size_t uvc_video_ctrl_size''')
    p.write_text(text)
    (work/'Makefile').write_text('''obj-m := u02_uvcvideo.o
u02_uvcvideo-y := uvc_driver.o uvc_queue.o uvc_v4l2.o uvc_video.o uvc_ctrl.o uvc_status.o uvc_isight.o uvc_debugfs.o uvc_metadata.o
ifeq ($(CONFIG_MEDIA_CONTROLLER),y)
u02_uvcvideo-y += uvc_entity.o
endif
''')
    state['modified_source_sha256']={p.name:digest(p) for p in work.iterdir() if p.is_file()}
    command=['make','-C',str(kernel),'M='+str(work),'ARCH=arm64','CROSS_COMPILE=aarch64-linux-gnu-','-j2','modules']
    with (output/'build.log').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    state['build_exit_code']=result.returncode
    assert result.returncode==0,(output/'build.log').read_text()[-4000:]
    module=work/'u02_uvcvideo.ko';shutil.copyfile(module,output/module.name)
    info=subprocess.check_output(['modinfo',str(module)],text=True)
    (output/'modinfo.txt').write_text(info)
    assert '6.6.40' in info and re.findall(r'(?m)^alias:\s+(\S+)',info)==['usb:v10BBp2B08d*dc*dsc*dp*ic0Eisc01ip00in*']
    assert all(digest(source/name)==value for name,value in original.items())
    assert digest(kernel/'.config')==state['kernel_config_sha256']
    assert digest(kernel/'Module.symvers')==state['module_symvers_sha256']
    state.update(complete=True,module_sha256=digest(module))
finally:
    (output/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    print('U02_MJPEG_DRIVER_PREPARED',output,flush=True)
