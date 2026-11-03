"""Control admission and bounded recovery; never rewrites measured skeletons."""
import numpy as np


class SkeletonQuality:
    # An estimated leg must have a measured anchor in this session. These are
    # admission limits, not claims about human accuracy or maximum human speed.
    ANCHOR_AGE_NS = 2_000_000_000
    ESTIMATED_SPEED_MPS = 2.
    BONES = ((11,13),(13,15),(12,14),(14,16))

    def __init__(self):
        self.identity = None
        self.previous = (-1,-1)
        self.observations = {}
        self.lengths = {bone:[] for bone in self.BONES}
        self.accepted = None

    def assess(self, packet):
        identity = tuple(packet[k] for k in ('session_id','clock_id','calibration_id','model_id'))
        stamp, seq = packet['timestamp_ns'], packet['seq']
        if self.identity is not None and identity != self.identity:
            raise ValueError('Quality state cannot cross a session/clock/model/calibration change')
        if stamp <= self.previous[0] or seq <= self.previous[1]:
            raise ValueError('Quality input must strictly increase')
        self.identity, self.previous = identity, (stamp,seq)
        joints = packet['joints']
        reasons = []
        for j in range(11,17):
            p=joints[j]
            if p['valid'] and not p.get('estimated'):
                history=self.observations.setdefault(j,[])
                history.append(stamp)
                del history[:-3]
        for a,b in self.BONES:
            x,y=joints[a],joints[b]
            if x['valid'] and y['valid'] and not x.get('estimated') and not y.get('estimated'):
                values=self.lengths[(a,b)]
                values.append(float(np.linalg.norm(np.array(x['xyz'])-y['xyz'])))
                del values[:-30]
        for j in (15,16):
            history=self.observations.get(j,[])
            if len(history)<3:
                reasons.append(dict(joint=j,reason='ankle_anchor_not_initialized'))
            elif stamp-history[-1]>self.ANCHOR_AGE_NS:
                reasons.append(dict(joint=j,reason='estimated_leg_without_recent_observation'))
        for j in range(5,17):
            if not joints[j]['valid']:
                reasons.append(dict(joint=j,reason='missing_body_joint'))
        dt=(stamp-self.accepted['timestamp_ns'])/1e9 if self.accepted else None
        for j in range(11,17):
            p=joints[j]
            if not p['valid'] or not p.get('estimated'):continue
            if dt is not None and 0<dt<=.2:
                old=self.accepted['joints'][j]
                if old['valid']:
                    speed=float(np.linalg.norm(np.array(p['xyz'])-old['xyz'])/dt)
                    if speed>self.ESTIMATED_SPEED_MPS:
                        reasons.append(dict(joint=j,reason='estimated_leg_speed',speed_mps=speed))
        for a,b in self.BONES:
            x,y=joints[a],joints[b]
            history=self.lengths[(a,b)]
            if x['valid'] and y['valid'] and (x.get('estimated') or y.get('estimated')) and len(history)>=3:
                length=float(np.linalg.norm(np.array(x['xyz'])-y['xyz']))
                reference=float(np.median(history))
                if not .65*reference <= length <= 1.35*reference:
                    reasons.append(dict(joint=b,reason='estimated_leg_length',length_m=length,reference_m=reference))
        accepted=not reasons
        if accepted:self.accepted=packet
        return dict(accepted=accepted,reasons=reasons,
                    observed_body=sum(j['valid'] and not j.get('estimated') for j in joints[5:]),
                    estimated_body=sum(j['valid'] and j.get('estimated',False) for j in joints[5:]))


def recovery_step(previous, target, dt):
    """Limit recovery targets only, in the existing physics reference frame."""
    previous, target=np.asarray(previous),np.asarray(target)
    dt=min(max(float(dt),0.),.2)
    output=target.copy()
    displacement=target[:3]-previous[:3]
    distance=float(np.linalg.norm(displacement))
    output[:3]=previous[:3]+displacement*min(1.,.5*dt/max(distance,1e-12))
    a,b=previous[3:7].copy(),target[3:7].copy()
    a/=np.linalg.norm(a);b/=np.linalg.norm(b)
    if np.dot(a,b)<0:b=-b
    angle=float(np.arccos(np.clip(np.dot(a,b),-1.,1.)))
    fraction=min(1.,1.5*dt/max(2*angle,1e-12))
    if angle>1e-6:
        output[3:7]=(np.sin((1-fraction)*angle)*a+np.sin(fraction*angle)*b)/np.sin(angle)
    else:output[3:7]=b
    output[7:]=previous[7:]+np.clip(target[7:]-previous[7:],-2.5*dt,2.5*dt)
    finished=bool(np.linalg.norm(output[:3]-target[:3])<.001 and
                  np.max(np.abs(output[7:]-target[7:]))<.005 and fraction==1.)
    return output,finished
