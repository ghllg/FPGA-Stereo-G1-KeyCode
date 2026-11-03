"""Export the released ProtoMotions MuJoCo configuration for Unitree's runner."""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'g1_physical'))
from physical import MODEL, load_mujoco_model
from deployment.test_tracker_mujoco import _resolve_mjcf_path


def build():
    meta = yaml.safe_load(MODEL.with_suffix('.yaml').read_text())
    c,t = meta['control'],meta['timing']
    model,_ = load_mujoco_model(meta['robot']['mjcf_path'],c['stiffness'],c['damping'],t['physics_dt'])
    output = Path(__file__).parent/'simulate_python/policy_scene.xml'
    mujoco.mj_saveLastXML(str(output),model)
    tree = ET.parse(output)
    root = tree.getroot()
    compiler = root.find('compiler')
    compiler.set('meshdir',str((_resolve_mjcf_path(meta['robot']['mjcf_path']).parent/compiler.get('meshdir','')).resolve()))
    # MuJoCo's save helper omits runtime-changed actuator enum flags. Preserve
    # them explicitly: bias coefficients alone do not enable affine feedback.
    for i,actuator in enumerate(root.find('actuator')):
        actuator.set('gaintype','fixed')
        actuator.set('biastype','affine')
        actuator.set('ctrllimited','false')
        actuator.set('forcelimited','true' if model.actuator_forcelimited[i] else 'false')
        actuator.set('gainprm',' '.join(map(str,model.actuator_gainprm[i])))
        actuator.set('biasprm',' '.join(map(str,model.actuator_biasprm[i])))
        actuator.set('forcerange',' '.join(map(str,model.actuator_forcerange[i])))
    asset = root.find('asset')
    ET.SubElement(asset,'texture',name='tracking_ground_tex',type='2d',builtin='checker',rgb1='.16 .24 .32',rgb2='.28 .39 .49',width='512',height='512')
    ET.SubElement(asset,'material',name='tracking_ground',texture='tracking_ground_tex',texrepeat='4 4',texuniform='true')
    for geom in root.find('worldbody').findall('geom'):
        if geom.get('type')=='plane':geom.set('material','tracking_ground')
    # mj_saveLastXML preserves compiled dynamics. Only sensors and view settings follow.
    sensor = ET.SubElement(root,'sensor')
    for tag in ('jointpos','jointvel','jointactuatorfrc'):
        for name in meta['joint_names']:
            ET.SubElement(sensor,tag,name=tag+'_'+name,joint=name)
    for tag,name in [('framequat','imu_quat'),('gyro','imu_gyro'),('accelerometer','imu_acc'),('framepos','frame_pos'),('framelinvel','frame_vel')]:
        attrs={'name':name}
        if tag in ('gyro','accelerometer'):attrs['site']='imu_in_pelvis'
        else:attrs.update(objtype='site',objname='imu_in_pelvis')
        ET.SubElement(sensor,tag,attrs)
    ET.SubElement(root,'statistic',center='0 0 0.6',extent='1.5')
    tree.write(output,encoding='unicode')
    verify = mujoco.MjModel.from_xml_path(str(output))
    import numpy as np
    for field in ('body_mass','body_inertia','geom_friction','jnt_stiffness','dof_damping','actuator_gainprm','actuator_biasprm','actuator_biastype','actuator_gaintype','actuator_ctrllimited','actuator_forcelimited','actuator_forcerange'):
        np.testing.assert_allclose(getattr(model,field),getattr(verify,field),rtol=1e-5,atol=1e-7)
    print(output)


if __name__=='__main__':build()
