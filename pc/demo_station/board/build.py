"""Build isolated AArch64 utilities using an existing SDK and NCNN build."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import re
import shlex
import struct


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, default=Path('E:/FPGAchina/full_chain_build_20261022'),help='Qualified build root containing src, deps and pipeline_assets')
    parser.add_argument('--built', type=Path, help='Qualified generated-header directory; default BASELINE/build_ascii')
    parser.add_argument('--sdk', type=Path, default=Path('E:/FPGAchina/FPGA2026/pose3d_train/runs/arm13/bin'))
    args = parser.parse_args()
    out = here/'build';out.mkdir(exist_ok=True)
    source = repo/'pose3d_train/handoff_20261023/board_changes/src'
    host = source/'ultrascale_plus/u02/inference/host'
    built = args.built or args.baseline/'build_ascii'
    deps = args.baseline/'deps'
    ncnn,ncnn_generated = deps/'ncnn_source',deps/'ncnn_generated'
    jpeg,jpeg_src = deps/'jpeg_build',deps/'jpeg_source'
    prefix = 'aarch64-none-linux-gnu-' if (args.sdk/'aarch64-none-linux-gnu-g++.exe').exists() else 'aarch64-linux-gnu-'
    compiler = args.sdk/(prefix+'g++.exe')
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    dependency_manifest=repo/'pose3d_train/handoff_20261023/workspace/outputs/full_chain_20261022/dependency_hashes.json'
    expected=json.loads(dependency_manifest.read_text(encoding='utf8'))
    for name,value in expected.items():
        if sha(deps/name)!=value:raise ValueError('Qualified dependency changed: '+name)
    qualified={built/'stream_model_generated.h':'e6b43e9f42c81b86b90359455327204f27c6bc3363b23f0bee55c37505b02ce0',
        built/'direct_layout_headers/stream_schedule_generated.h':'395beb513e203248fd83471a6c55c98ee6e1473b0d90fa2aebc7c55641cab9ba',
        args.baseline/'pipeline_assets/model.opt.bin':'edf5c65c812c4e1b441d66bd7dd0bea3fb2a12f27f1e847124edd32cfa1354fc',
        args.baseline/'pipeline_assets/model.stream.param':'4f1ba4b86403b80ce523e2686aa420460f1df4c96947b1aedad6a316ecd52a12',
        built/'qualified_1080p_host/pose_stereo_pipeline_qualified_1080p':'ffa254104dc8822d9bb751395cf66961a908c291a53f91ba4ce7797901154b42'}
    for path,value in qualified.items():
        if sha(path)!=value:raise ValueError('Qualified input changed: '+str(path))
    for path in source.rglob('*'):
        if path.suffix in ('.h','.cpp') and sha(path)!=sha(args.baseline/'src'/path.relative_to(source)):
            raise ValueError('Archived source differs from qualified source: '+str(path))
    common = [str(compiler), '-std=c++11','-O2','-static','-pthread','-ffp-contract=off','-march=armv8-a','-mtune=cortex-a53']
    # Only this private copy broadens enumeration to the V4L2 nodes identified by bus.
    camera = (host/'camera_capture.cpp').read_text()
    if camera.count('node=="/dev/video0" || node=="/dev/video1" || node=="/dev/video2"')!=1:raise ValueError('Unexpected camera-node allowlist source')
    camera = camera.replace('node=="/dev/video0" || node=="/dev/video1" || node=="/dev/video2"',
                            'node.compare(0,10,"/dev/video")==0 && node.size()<=12 && node.find_first_not_of("0123456789",10)==std::string::npos')
    (out/'camera_capture.cpp').write_text(camera)
    commands = [];binaries={};dependency_files=set(qualified)|{compiler,Path(__file__),dependency_manifest,ncnn_generated/'libncnn.a',jpeg/'libturbojpeg.a',host/'camera_capture.cpp'}
    report=dict(commands=commands,binaries=binaries,board_tested=False,completed=False,qualified_dependency_checks=len(expected),
        limitations=['Local cross-compile only; not deployed or run on board','Recorded-image comparison is not continuous live throughput',
                     'CPU/PL full-heatmap equality and camera operation must pass on board before acceptance'])
    def run(command, name):
        commands.append(command)
        process = subprocess.run(command, cwd=out, capture_output=True, text=True)
        (out/(name+'.log')).write_text(process.stdout+process.stderr, encoding='utf-8')
        if process.returncode:
            print((process.stdout+process.stderr)[-12000:]);raise RuntimeError(name+' failed: '+str(process.returncode))
    def build(name,options,inputs,libraries):
        objects=[]
        for index,path in enumerate(inputs):
            obj=out/(name+'_'+str(index)+'.o');dep=obj.with_suffix('.d')
            run(common+options+['-MMD','-MF',str(dep),'-c',str(path),'-o',str(obj)],name+'_'+str(index))
            dependency_files.add(path)
            text=dep.read_text().replace('\\\n',' ')
            dependency_files.update(Path(value) for value in shlex.split(text.split(': ',1)[1],posix=False))
            objects.append(str(obj))
        target=out/name
        run(common+options+objects+libraries+['-Wl,--whole-archive','-lpthread','-Wl,--no-whole-archive','-o',str(target)],name)
        run([str(args.sdk/(prefix+'readelf.exe')),'-h','-l',str(target)],name+'_elf')
        data=target.read_bytes()
        if data[:6]!=b'\x7fELF\x02\x01' or struct.unpack_from('<H',data,18)[0]!=183:raise ValueError('Expected little-endian AArch64 ELF64')
        offset=struct.unpack_from('<Q',data,32)[0];size,count=struct.unpack_from('<HH',data,54)
        if any(struct.unpack_from('<I',data,offset+i*size)[0]==3 for i in range(count)):raise ValueError('Unexpected dynamic interpreter')
        binaries[name]=dict(sha256=sha(target),bytes=len(data),elf='ELF64 little-endian AArch64',static_no_interpreter=True)
    defines = ['U02_STREAM_MODEL','U02_STREAM_SCHEDULE','U02_STREAM_FAST_NEAREST','U02_STREAM_FUSED_LAYOUT',
               'U02_STREAM_FAST_AFFINE','U02_STREAM_RESIDENT_PACKETS','U02_STREAM_PARALLEL_AFFINE',
               'U02_ARM_STEM_SMLAL','U02_STREAM_DEQUANT_UNROLL4','U02_STREAM_BOUNDED_QUANT',
               'U02_STREAM_PIPELINE','U02_IMAGE_JPEG_TURBO','U02_IMAGE_SPARSE_ROWS']
    for name in ('stream_model_generated.h','stream_schedule_generated.h'):
        original=built/name if name=='stream_model_generated.h' else built/'direct_layout_headers'/name
        generated=original.read_text()
        generated=re.sub(r'#include "[^"]*stream_contract.h"', '#include "'+(source/'ultrascale_plus/u02/inference/hls/stream_contract.h').as_posix()+'"', generated)
        (out/name).write_text(generated)
    include = [host,source,source/'board_single_pose/host',source/'board_single_pose/hls',out,built,
               ncnn,ncnn_generated,jpeg,jpeg_src]
    inputs = [here/'compare_sequence.cpp',host/'backend.cpp',host/'pl_stream.cpp',host/'image_preprocess.cpp',
              source/'board_single_pose/hls/pose_gemm.cpp']
    try:
        run([str(compiler),'--version'],'compiler_version')
        build('record_stereo',['-DU02_CAPTURE_1080P=1','-I'+str(host)],[here/'record_stereo.cpp',out/'camera_capture.cpp'],[])
        build('compare_sequence',['-fopenmp']+['-D'+x+'=1' for x in defines]+['-I'+str(x) for x in include],inputs,
            [str(ncnn_generated/'libncnn.a'),str(jpeg/'libturbojpeg.a')])
        report['completed']=True
    except Exception as error:
        report['error']=repr(error);raise
    finally:
        report['source_dependencies']=[dict(path=str(path.resolve()),sha256=sha(path)) for path in sorted(dependency_files)]
        report['qualified_inputs']={str(path):value for path,value in qualified.items()}
        (out/'build.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(binaries,indent=2))


if __name__=='__main__':main()
