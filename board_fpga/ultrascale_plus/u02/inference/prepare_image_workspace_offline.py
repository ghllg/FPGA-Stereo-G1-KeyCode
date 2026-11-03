"""Build and check the exact-image candidate without contacting the board.

Uses only the existing WSL compiler, SDK cross-compiler and pinned local JPEG
source. Outputs, codec build, logs and temporary files stay on E:/D:.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import os
import statistics
import struct
import subprocess
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
BASELINE = 'cf34738310044b0a596468a0a4b925679dd3bb9a'
SDK = Path('D:/Develop/Xilink/SDK/2018.3/gnu/aarch64/nt/aarch64-linux/bin')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def linux(path):
    p = Path(path).resolve().as_posix()
    return '/mnt/' + p[0].lower() + p[2:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-codec', type=Path, help='Reuse an existing local scalar 3.1.2 PC build')
    parser.add_argument('--reuse-level', type=int, choices=(0, 1, 2), default=2)
    parser.add_argument('--pc-only-compare', action='store_true', help='Bounded release comparison; no sanitizer or ARM build')
    options = parser.parse_args()
    out = ROOT / 'reports' / ('image_workspace_offline_' + str(time.time_ns()))
    out.mkdir(parents=True)
    print(out, flush=True)
    env = dict(os.environ, TMP=str(ROOT / 'toolchain/tmp'), TEMP=str(ROOT / 'toolchain/tmp'))
    commands = []

    def run(name, args):
        if str(args[0]) == 'wsl.exe':
            args = args[:2] + ['env', 'TMPDIR=' + linux(ROOT / 'toolchain/tmp')] + args[2:]
        before = time.monotonic()
        result = subprocess.run([str(a) for a in args], cwd=ROOT, env=env, capture_output=True)
        (out / (name + '.stdout')).write_bytes(result.stdout)
        (out / (name + '.stderr')).write_bytes(result.stderr)
        commands.append(dict(name=name, argv=[str(a) for a in args], returncode=result.returncode,
                             seconds=time.monotonic() - before))
        (out / 'commands.json').write_text(json.dumps(commands, indent=2), encoding='utf-8')
        if result.returncode:
            raise RuntimeError(name + ' failed; see ' + str(out))
        print(name + ' passed', flush=True)
        return result

    reference = out / 'reference'
    reference.mkdir()
    for name in ('image_preprocess.cpp', 'image_preprocess.h', 'sparse_jpeg_rows.h'):
        content = subprocess.check_output(['git', 'show', BASELINE + ':ultrascale_plus/u02/inference/host/' + name], cwd=REPO)
        (reference / name).write_bytes(content)
    fixtures = ROOT / 'reports/stereo1080_decode_pairs_1792168113447916600'
    fixture_manifest = json.loads((fixtures / 'pairs.json').read_text(encoding='utf-8'))
    for name, digest in fixture_manifest['files_sha256'].items():
        if sha(fixtures / name) != digest:
            raise RuntimeError('Real input hash differs: ' + name)
    # A small RGB PNG exercises the STB fallback and changes source geometry.
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    raw = b''.join(b'\0' + bytes((x * 71 + y * 97 + c * 113) & 255 for x in range(19) for c in range(3)) for y in range(13))
    png = out / 'fixture.png'
    png.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 19, 13, 8, 2, 0, 0, 0))
                    + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))
    codec = ROOT / 'toolchain/libjpeg-turbo-3.1.2-source/libjpeg-turbo-3.1.2'
    native_codec = options.native_codec.resolve() if options.native_codec else out / 'native_jpeg'
    # No assembler is installed in this existing WSL environment. Scalar x86
    # is a controlled PC comparison, never an A53 NEON performance estimate.
    if options.native_codec:
        cache = (native_codec / 'CMakeCache.txt').read_text()
        if 'WITH_SIMD:BOOL=OFF' not in cache or 'CMAKE_BUILD_TYPE:STRING=Release' not in cache:
            raise RuntimeError('Reused PC library must be a scalar Release build')
        (out / 'native_codec_reuse.json').write_text(json.dumps(dict(path=str(native_codec),
            library_sha256=sha(native_codec / 'libturbojpeg.a'), cache_sha256=sha(native_codec / 'CMakeCache.txt')), indent=2))
    else:
        run('native_codec_configure', ['wsl.exe', '-e', 'cmake', '-S', linux(codec), '-B', linux(native_codec),
            '-DCMAKE_BUILD_TYPE=Release', '-DENABLE_SHARED=OFF', '-DENABLE_STATIC=ON', '-DWITH_SIMD=OFF',
            '-DWITH_TOOLS=OFF', '-DWITH_TESTS=OFF'])
        run('native_codec_build', ['wsl.exe', '-e', 'cmake', '--build', linux(native_codec), '--target', 'turbojpeg-static', '-j', '4'])
    host = ROOT / 'inference/host'
    source_hashes = {str((host / name).relative_to(ROOT)): sha(host / name) for name in
        ('image_preprocess.cpp', 'image_preprocess.h', 'sparse_jpeg_rows.h',
         'prepared_image_geometry.h', 'test_image_workspace.cpp')}
    stb = REPO / 'board_pose_benchmark/outputs/ncnn_source/src'
    defines = ['-DU02_IMAGE_JPEG_TURBO=1', '-DU02_IMAGE_SPARSE_ROWS=1', '-DSTB_IMAGE_STATIC=1',
               '-DU02_IMAGE_WORKSPACE_REUSE_LEVEL=' + str(options.reuse_level)]
    renames = ['-Dprepare_pose_image=reference_prepare_pose_image',
               '-Dprepare_pose_image_roi=reference_prepare_pose_image_roi', '-Dread_pose_image=reference_read_pose_image']

    def compile_suite(prefix, compiler, codec_build, convert, sanitize=False):
        common = ['-std=c++11', '-O3', '-ffp-contract=off', '-pthread'] + defines
        if sanitize:
            common += ['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
        includes = ['-I' + convert(p) for p in (host, stb, codec_build, codec / 'src')]
        if prefix == 'aarch64':
            common += ['-march=armv8-a', '-mtune=cortex-a53', '-static']
        baseline = out / (prefix + '_reference.o')
        candidate = out / (prefix + '_candidate.o')
        executable = out / (prefix + '_test_image_workspace')
        run(prefix + '_reference_build', compiler + common + includes + renames + ['-c', convert(reference / 'image_preprocess.cpp'), '-o', convert(baseline)])
        run(prefix + '_candidate_build', compiler + common + includes + ['-DU02_IMAGE_WORKSPACE=1', '-c', convert(host / 'image_preprocess.cpp'), '-o', convert(candidate)])
        # SDK 2018.3's static libstdc++ uses weak pthread references. -pthread
        # alone can omit pthread_join; match the accepted live Host's link.
        thread_link = ['-Wl,--whole-archive', '-lpthread', '-Wl,--no-whole-archive'] if prefix == 'aarch64' else []
        run(prefix + '_suite_link', compiler + common + includes + ['-DU02_IMAGE_WORKSPACE=1', convert(host / 'test_image_workspace.cpp'),
            convert(baseline), convert(candidate), convert(codec_build / 'libturbojpeg.a')] + thread_link + ['-o', convert(executable)])
        return executable

    executable = compile_suite('native', ['wsl.exe', '-e', 'g++'], native_codec, linux)
    result = run('native_suite', ['wsl.exe', '-e', linux(executable), linux(fixtures), linux(png)])
    if b'U02_IMAGE_WORKSPACE_PASS' not in result.stderr:
        raise RuntimeError('Missing complete suite proof')
    rows = list(csv.DictReader(io.StringIO(result.stdout.decode())))
    summary = {}
    for mode in ('serial', 'parallel'):
        summary[mode] = {}
        for variant in ('baseline', 'workspace'):
            selected = [r for r in rows if r['mode'] == mode and r['variant'] == variant and int(r['round']) >= 0]
            if len(selected) != 80:
                raise RuntimeError('Incomplete paired benchmark')
            def median(field):
                return statistics.median(float(r[field]) for r in selected)
            summary[mode][variant] = dict(pairs=len(selected), pair_wall_median_ms=median('pair_wall_ms'),
                image_decode_median_ms=statistics.median(float(r[k]) for r in selected for k in ('left_decode_ms', 'right_decode_ms')),
                image_preprocess_median_ms=statistics.median(float(r[k]) for r in selected for k in ('left_pre_ms', 'right_pre_ms')))
        summary[mode]['stages'] = [dict(stage=stage, variant='workspace' if stage in (1, 2) else 'baseline',
            pair_wall_median_ms=statistics.median(float(r['pair_wall_ms']) for r in rows
                if r['mode'] == mode and int(r['stage']) == stage and int(r['round']) >= 0)) for stage in range(4)]
    if not options.pc_only_compare:
        sanitized = compile_suite('asan', ['wsl.exe', '-e', 'g++'], native_codec, linux, sanitize=True)
        run('asan_suite', ['wsl.exe', '-e', linux(sanitized), linux(fixtures), linux(png)])
        arm = compile_suite('aarch64', [str(SDK / 'aarch64-linux-gnu-g++.exe')],
                            ROOT / 'toolchain/libjpeg-turbo-3.1.2-aarch64', lambda p: str(p))
        run('aarch64_elf', [SDK / 'aarch64-linux-gnu-readelf.exe', '-h', '-l', '-d', arm])
    if any(sha(ROOT / name) != digest for name, digest in source_hashes.items()):
        raise RuntimeError('Test source changed during the run')
    (out / 'summary.json').write_text(json.dumps(dict(scope='PC-only decode/preprocess; no board or NN execution',
        board_accessed=False, baseline_commit=BASELINE, native_simd=False, pc_results=summary,
        reuse_level=options.reuse_level, pc_only_compare=options.pc_only_compare,
        source_sha256=source_hashes,
        suite_proof=result.stderr.decode(errors='replace'), inputs=fixture_manifest,
        files_sha256={str(p.relative_to(out)): sha(p) for p in out.iterdir() if p.is_file()}), indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
