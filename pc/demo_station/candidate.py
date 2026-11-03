"""Read-only selection and identity checks for the frozen ordinary-motion candidate."""
import hashlib
import importlib.util
import json
from pathlib import Path
from urllib.parse import urlparse
from .core import ROOT, digest, read_json, write_json

TASK = Path(r'C:\Users\developer\Documents\Codex\2026-10-23\new-chat')
PC = Path(r'C:\Users\developer\Documents\Codex\2026-10-20\iq\work\full_pipeline_exec_20261022')
G1 = Path(r'E:\FPGAchina\G1_1080p_Candidate_20261020')
VISION = Path(r'E:\FPGAchina\FPGA2026\windows_vision\.venv\Scripts\python.exe')
RELEASE = TASK/'outputs/remaining_failures_20261024/release'
SERVER = TASK/'work/offline_closure_20261024/timing/combined4e4/prediction_server.py'
STREAM = RELEASE/'sources/semantic/transaction_stream.py'
RETENTION = TASK/'work/final_gap_closure_20261024/retention'
TRAIN = PC.parents[1]/'outputs/training_1080p_20261021'
DEFAULT_ASSETS = dict(calibration=TRAIN/'calibration.json', checkpoint=TRAIN/'model/checkpoint.pt',
    binding=TASK/'outputs/stage2_14hz_20261023/fixtures/normal_14hz_latest/audit/binding.json')


def calibration_hash(calibration):
    keys = ('calibration_id','camera_ids','image_size','K','distortion','R_cam0_to_cam1',
            't_cam0_to_cam1_m','world_from_cam0','world_coordinate_system')
    return hashlib.sha256(json.dumps({k:calibration.get(k) for k in keys},sort_keys=True,separators=(',',':')).encode()).hexdigest()


def pairing_settings(config):
    limit=config.get('pair_limit_us',10000)
    if type(limit) is not int or limit not in (10000,16667,20000):
        raise ValueError('实时配对候选仅支持明确的 10000、16667 或 20000 μs 配置')
    if limit==20000:
        if config.get('pairing_policy')!='nearest_raw_20ms':
            raise ValueError('20 ms 必须明确选择 nearest_raw_20ms 配对策略')
        return dict(pairing_policy='nearest_raw_20ms',transport_pair_limit_us=20000,reconstruction_pair_limit_ms=20.)
    if config.get('pairing_policy') not in (None,'bounded_half_frame_raw'):
        raise ValueError('配置的配对策略与门限不一致')
    # Preserve the frozen live path: 16.667 ms transport, 10 ms reconstruction.
    return dict(pairing_policy='bounded_half_frame_raw',transport_pair_limit_us=16667,reconstruction_pair_limit_ms=10.)


def verify_pair20_candidate():
    path=ROOT/'demo_station/profiles/pairing20_candidate.json';manifest=read_json(path)
    if (manifest.get('schema')!='pairing20_candidate/1' or manifest.get('pair_limit_us')!=20000
            or manifest.get('pairing_policy')!='nearest_raw_20ms'):
        raise ValueError('20 ms 候选清单声明不一致')
    expected={ROOT/'pose3d_train/online_pair20.py':ROOT/'pose3d_train/online.py',
              ROOT/'demo_station/pair20_performance_receiver.py':PC/'performance_receiver.py'}
    rows=manifest.get('files',[])
    if len(rows)!=len(expected) or {Path(r['path']).resolve() for r in rows}!=set(expected):
        raise ValueError('20 ms 候选清单文件范围不一致')
    for row in rows:
        target=Path(row['path']).resolve();source=Path(row['source']).resolve()
        if source!=expected[target] or digest(source)!=row['source_sha256'] or digest(target)!=row['sha256']:
            raise ValueError('20 ms 候选副本或原冻结来源 SHA 不一致：'+str(target))
    geometry=manifest['geometry_candidate']
    source=ROOT/'pose3d_train/geometry.py'
    target=ROOT/'pose3d_train/geometry_reprojection.py'
    if digest(source)!=geometry['source_sha256'] or digest(target)!=geometry['sha256']:
        raise ValueError('Reprojection candidate or frozen geometry source SHA mismatch')
    return dict(path=str(path),sha256=digest(path),files=rows,geometry_candidate=geometry)


def local_preflight(config):
    """Read files/hashes only: no sockets, subprocesses, models or camera access."""
    pairing=pairing_settings(config)
    runtime = Path(config.get('g1_dir') or G1).resolve()
    if runtime != G1.resolve(): raise ValueError('当前验收候选绑定指定 G1 目录，不能无声切换其他模型环境')
    url=urlparse(config.get('g1_url') or 'http://127.0.0.1:8771/ws')
    if url.scheme!='http' or url.hostname not in ('127.0.0.1','localhost') or url.port!=8771 or url.path!='/ws':
        raise ValueError('推荐候选使用固定本机地址 http://127.0.0.1:8771/ws')
    python = Path(config.get('vision_python') or VISION).resolve()
    for file in (python,runtime/'runtime/python.exe',SERVER,STREAM):
        if not file.is_file(): raise ValueError('缺少候选运行文件：'+str(file))
    assets = {k:Path(config.get(k) or value).resolve() for k,value in DEFAULT_ASSETS.items()}
    for file in assets.values():
        if not file.is_file(): raise ValueError('缺少会话资产：'+str(file))
    cal, binding = read_json(assets['calibration']), read_json(assets['binding'])
    if binding.get('calibration_hash') != calibration_hash(cal): raise ValueError('绑定与标定内容不一致')
    if cal.get('image_size') != [1920,1080] or binding.get('camera_physical_ids') != cal.get('camera_physical_ids'):
        raise ValueError('候选要求原 1080p 标定与物理左右相机身份')
    for side in ('left','right'):
        if config.get(side+'_roi') != binding.get('roi_xywh',{}).get(side):
            raise ValueError('配置 ROI 与已选标定绑定不一致：'+side)
    if pairing['transport_pair_limit_us']==20000 and (binding.get('pair_limit_us')!=20000 or binding.get('pairing_policy')!=pairing['pairing_policy']):
        raise ValueError('20 ms 配置与会话绑定声明不一致')
    if digest(assets['checkpoint']) != digest(DEFAULT_ASSETS['checkpoint']):
        raise ValueError('三维模型与本轮冻结候选不一致')
    if calibration_hash(cal) != calibration_hash(read_json(DEFAULT_ASSETS['calibration'])):
        raise ValueError('标定与本轮冻结候选不一致')
    ports=read_json(runtime/'ports.json')
    if any(ports.get(k)!=v for k,v in dict(web_port=8770,official_web_port=8771,dds_domain=74).items()):
        raise ValueError('G1 本机服务端口或 DDS 域与冻结候选不一致')
    spec = importlib.util.spec_from_file_location('motion_candidate_verifier',RELEASE/'reproduce_increment.py')
    verifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(verifier)
    _, checks = verifier.verify(RELEASE)
    pair20=verify_pair20_candidate() if pairing['transport_pair_limit_us']==20000 else None
    hybrid=None
    if config.get('hybrid_refinement'):
        from .hybrid_candidate import local_qualification
        hybrid=local_qualification(config)
    provisional = bool(cal.get('diagnostic_only') or not cal.get('accepted_for_reconstruction',False))
    return dict(ok=True,variant='combined4e4_transaction_live_adapter_pair20_v1' if pair20 else 'combined4e4_transaction_live_adapter_v1',checks=checks,
        pairing=pairing,pair20_candidate=pair20,hybrid_refinement=hybrid,
        python=str(python),g1_dir=str(runtime),server=str(SERVER),stream=str(STREAM),
        lag_ms=240,smooth_ms=80,quality_guard=True,max_prediction_ms=160,
        assets={k:dict(path=str(p),sha256=digest(p)) for k,p in assets.items()},
        calibration_provisional=provisional,hardware_checked=False,ssh_identity_checked=False,
        pending=['现场 SSH 身份与同 boot 运行资格尚未检查','相机机位需现场确认；旧标定不证明当前机位未变化'],
        warnings=['当前标定仅用于已授权诊断；机位变化必须重新核实，不能声称绝对地面或单腿已验收'] if provisional else [])


def snapshot_candidate(config, output):
    output = Path(output); proof = local_preflight(config)
    files = {}
    for key,item in proof['assets'].items():
        source = Path(item['path']); target = output/'assets'/(key+source.suffix)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if digest(target) != item['sha256']: raise ValueError('已有会话资产不一致：'+key)
        else: target.write_bytes(source.read_bytes())
        files[key] = str(target)
    write_json(output/'candidate_preflight.json',proof)
    return proof,files


def validate_live_identity(binding, preflight, template, config):
    pairing=pairing_settings(config)
    if not isinstance(preflight,dict) or preflight.get('ok') is not True:
        raise ValueError('需先完成本轮只读板端预检；旧模板 boot_id 不能作为现场证据')
    if not preflight.get('checked_at_utc') or preflight.get('host') != config['host']:
        raise ValueError('板端预检缺少时间或与目标主机不符')
    if config.get('hybrid_refinement') and preflight.get('host_candidate',{}).get('protocol')!='u02h_paired_jpeg/1':
        raise ValueError('协同候选必须确认同帧双原图传输程序')
    if not isinstance(binding,dict) or 'session_id' in binding:
        raise ValueError('现场绑定不得携带旧 session_id；新 session 仅由第一条合格消息建立')
    for key in ('calibration_hash','camera_physical_ids','model_sha256','roi_xywh','input_profile'):
        if binding.get(key) != template.get(key): raise ValueError('现场绑定改变已选择资产：'+key)
    for key in ('boot_id','model_sha256','camera_physical_ids','roi_xywh'):
        if not binding.get(key) or binding[key] != preflight.get(key): raise ValueError('板端预检与绑定不一致：'+key)
    if pairing['transport_pair_limit_us']==20000:
        for key,value in (('pair_limit_us',20000),('pairing_policy','nearest_raw_20ms')):
            if any(item.get(key)!=value for item in (binding,preflight,template)):
                raise ValueError('20 ms 策略未在配置、绑定和板端预检中一致确认：'+key)
    if preflight.get('simulated_board') and config['host'] not in ('localhost','127.0.0.1'):
        raise ValueError('模拟板证据只允许 localhost')
    return dict(binding)
