"""Explicit qualification of the FPGA-guided PC refinement candidate."""
from pathlib import Path, PurePosixPath
import re
import sys

from .core import HOME, digest, read_json

MODEL_SHA256 = '5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c'
PROTOCOL = 'u02h_paired_jpeg/1'


def runtime_directory(backend='cpu'):
    if backend not in ('cpu', 'dml'):
        raise ValueError('Unknown hybrid inference backend')
    return HOME/'deps'/('rtmpose_dml_py311' if backend == 'dml' else 'rtmpose_py311')


def activate_runtime(backend='cpu'):
    directory = runtime_directory(backend).resolve()
    sys.path.insert(0, str(directory))
    import onnxruntime as ort
    if not Path(ort.__file__).resolve().is_relative_to(directory):
        raise RuntimeError('A different ONNX runtime is already loaded; start a fresh candidate process')
    provider = 'DmlExecutionProvider' if backend == 'dml' else 'CPUExecutionProvider'
    if provider not in ort.get_available_providers():
        raise RuntimeError('The selected inference provider is unavailable')
    return ort


def hip_flip_enabled(config):
    enabled = config.get('hybrid_hip_flip_check', False)
    if type(enabled) is not bool:
        raise ValueError('hybrid_hip_flip_check must be an explicit boolean')
    if enabled and (config.get('hybrid_refinement') is not True or config.get('full_body_check') is not True
                    or config.get('motion_mode', 'full_body') != 'full_body'):
        raise ValueError('Hip flip refinement requires the hybrid full-body check path')
    return enabled


def latency_settings(config):
    enabled = config.get('hybrid_latency_check', False)
    if type(enabled) is not bool:
        raise ValueError('hybrid_latency_check must be an explicit boolean')
    if enabled and not (config.get('hybrid_refinement') is True and config.get('full_body_check') is True
                        and config.get('standing_small_motion_check') is True
                        and config.get('motion_mode', 'full_body') == 'full_body'):
        raise ValueError('Latency candidate requires the hybrid standing-small-motion check')
    return dict(reference_lag_ms=360 if enabled else 240, board_inflight_pairs=1 if enabled else 2)


def reprojection_limit(config):
    limit = config.get("hybrid_reprojection_limit_px", 5.)
    if type(limit) not in (int, float) or limit not in (5., 6.):
        raise ValueError("Only baseline 5 px or reviewed candidate 6 px is supported")
    if limit != 5. and not (config.get("hybrid_refinement") is True
                           and config.get("standing_small_motion_check") is True
                           and config.get("full_body_check") is True):
        raise ValueError("6 px candidate requires the hybrid standing-small-motion check")
    return float(limit)


def local_qualification(config):
    limit = reprojection_limit(config)
    latency_settings(config)
    hip_flip_check = hip_flip_enabled(config)
    if (config.get('motion_mode') != 'full_body' or not config.get('full_body_check')
            or config.get('pair_limit_us') != 20000 or config.get('pairing_policy') != 'nearest_raw_20ms'):
        raise ValueError('协同候选只用于明确的 20 ms 全身短测')
    model = HOME/'models/rtmpose_m/end2end.onnx'
    if digest(model) != MODEL_SHA256: raise ValueError('协同候选精修模型 SHA 不一致')
    backend = config.get('hybrid_backend', 'cpu')
    dependency = runtime_directory(backend)/'qualification.json'
    proof = read_json(dependency)
    provider = 'DmlExecutionProvider' if backend == 'dml' else 'CPUExecutionProvider'
    if (proof.get('complete') is not True or proof.get('model_sha256') != MODEL_SHA256
            or provider not in proof.get('import_verified',{}).get('providers',[])):
        raise ValueError('隔离精修运行环境尚未完成验证')
    expected = {'onnxruntime.dll','onnxruntime_providers_shared.dll','onnxruntime_pybind11_state.pyd'}
    if backend == 'dml': expected.add('DirectML.dll')
    binaries = proof.get('core_binaries',[])
    if len(binaries)!=len(expected) or {Path(item['path']).name for item in binaries}!=expected:
        raise ValueError('隔离运行环境的核心文件清单不完整')
    for item in binaries:
        path=(dependency.parent/item['path']).resolve()
        if not path.is_relative_to(dependency.parent.resolve()) or digest(path)!=item['sha256']:
            raise ValueError('隔离运行环境核心文件 SHA 不一致')
    return dict(protocol=PROTOCOL,model_path=str(model),model_sha256=MODEL_SHA256,backend=backend,
        dependency=dict(path=str(dependency),sha256=digest(dependency)),hybrid_hip_flip_check=hip_flip_check,
        reconstruction_reprojection_limit_px=limit,
        fpga_role='Complete PP-TinyPose inference supplies the per-frame person proposal',
        pc_role='RTMPose-m computes the refined final 2D joints; existing 3D and G1 follow',
        trained_3d_distribution_compatibility='Candidate validation pending; original checkpoint and guards retained')


def board_qualification(pair20, boot):
    path = HOME/'profiles/hybrid_board.json'; candidate = read_json(path)
    verification = Path(candidate['verification']['path']); proof = read_json(verification)
    executable = Path(candidate['executable']['path']); sha = candidate['executable']['sha256']
    expected = pair20['files']['stereo_host']
    required = ('complete','all_full_heatmaps_exact','listener_no_camera_passed','cleanup_verified','transport_fixture_verified')
    if (candidate.get('schema') != 'hybrid_board_candidate/1' or candidate.get('protocol') != PROTOCOL
            or candidate.get('source_pair20_sha256') != expected or proof.get('source_pair20_sha256') != expected
            or digest(verification) != candidate['verification']['sha256'] or digest(executable) != sha
            or proof.get('stereo_host_sha256') != sha
            or proof.get('boot_id') != boot or not all(proof.get(key) is True for key in required)
            or proof.get('actual_PL_calls',0) <= 0):
        raise ValueError('协同板端候选的来源、同 boot 数值或传输验证不完整')
    remote = candidate['remote']
    if not re.fullmatch(r'/tmp/[A-Za-z0-9_/-]+',remote) or '..' in PurePosixPath(remote).parts:
        raise ValueError('协同板端候选必须位于独立临时目录')
    return dict(remote=remote,files={**pair20['files'],'stereo_host':sha},boot_id=boot,
        profile=dict(path=str(path),sha256=digest(path)),verification=candidate['verification'],protocol=PROTOCOL)
