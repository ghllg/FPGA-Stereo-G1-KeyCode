"""Read saved qpos and render meshes; never advance the physics integrator."""
import base64
import io
import json
from pathlib import Path
import sys
import struct
import zlib
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

def png(pixels):
    h,w,_=pixels.shape
    def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    raw=b''.join(b'\0'+row.tobytes() for row in pixels)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw,1))+chunk(b'IEND',b'')

root=Path(sys.argv[1])
xml=root/'g1_realtime/vendor/gmr/unitree_g1/g1_mocap_29dof.xml'
def scene(dark):
    tree=ET.fromstring(xml.read_text());tree.find('compiler').set('meshdir',str(xml.parent/'meshes'))
    sky='.145 .145 .145' if dark else '.98 .98 .98'
    ground='.24 .24 .24' if dark else '.88 .88 .88'
    grid='.30 .30 .30' if dark else '.74 .74 .74'
    for texture in tree.findall('.//texture'):
        if texture.get('type')=='skybox':
            texture.set('builtin','flat');texture.set('rgb1',sky);texture.set('rgb2',sky)
        if texture.get('name')=='groundplane':
            texture.set('rgb1',ground);texture.set('rgb2',ground);texture.set('markrgb',grid)
    tree.find('visual/rgba').set('haze',sky+' 1')
    for material in tree.findall('.//material'):
        if material.get('name')=='groundplane':material.set('emission','1');material.set('specular','0')
    model=mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'))
    model.vis.global_.offwidth=640;model.vis.global_.offheight=800
    return model,mujoco.MjData(model),mujoco.Renderer(model,height=800,width=640)
current_dark=False;model,data,renderer=scene(current_dark)
camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
options=mujoco.MjvOption();options.geomgroup[3:]=0
for line in sys.stdin:
    try:
        request=json.loads(line);q=np.asarray(request['qpos'],float)
        if request.get('dark',False)!=current_dark:
            renderer.close();current_dark=request['dark'];model,data,renderer=scene(current_dark)
        if q.shape!=(model.nq,) or not np.isfinite(q).all():raise ValueError('Invalid saved qpos')
        data.qpos[:]=q;mujoco.mj_forward(model,data)
        expected=np.asarray(request.get('body_pos',[]))
        if expected.shape!=data.xpos.shape or not np.allclose(expected,data.xpos,atol=2e-5):
            raise ValueError('Saved state does not match G1 model body ordering')
        camera.lookat[:]=[q[0],q[1],.8];camera.distance=2.8/request.get('zoom',1)
        camera.azimuth=request.get('yaw',.65)*180/np.pi;camera.elevation=-request.get('pitch',.24)*180/np.pi
        renderer.update_scene(data,camera=camera,scene_option=options)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE]=False
        pixels=renderer.render()
        print(json.dumps({'id':request['id'],'image':base64.b64encode(png(pixels)).decode('ascii')}),flush=True)
    except Exception as error:print(json.dumps({'id':request.get('id'), 'error':str(error)}),flush=True)
renderer.close()
