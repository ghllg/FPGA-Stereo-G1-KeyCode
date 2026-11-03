"""Control admission and bounded recovery; never rewrites measured skeletons."""
import numpy as np


class SkeletonQuality:
    # An estimated leg must have a measured anchor in this session. These are
    # admission limits, not claims about human accuracy or maximum human speed.
    ANCHOR_AGE_NS = 2_000_000_000
    ESTIMATED_SPEED_MPS = 2.
    BONES = ((5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16))
    # Broad scale-free checks before a personal bone profile exists. A stable
    # sequence of collapsed ankles must not calibrate its own short shins.
    PROPORTIONS = ((0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7))

    def __init__(self):
        self.identity = None
        self.previous = (-1,-1)
        self.observations = {}
        self.lengths = {bone:[] for bone in self.BONES}
        self.frozen_lengths = None
        self.accepted = None

    def freeze_bone_lengths(self, lengths):
        if self.frozen_lengths is not None:return
        if set(lengths)!=set(self.BONES) or any(not np.isfinite(v) or v<=0 for v in lengths.values()):
            raise ValueError('A complete finite warmup bone profile is required')
        self.frozen_lengths = dict(lengths)

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
        current = {bone:float(np.linalg.norm(np.array(joints[bone[0]]['xyz'])-joints[bone[1]]['xyz']))
                   for bone in self.BONES if all(joints[j]['valid'] for j in bone)}
        unreliable = set()
        for left,right in self.PROPORTIONS:
            a,b=self.BONES[left],self.BONES[right]
            if a in current and b in current and not .5*current[b] <= current[a] <= 2.*current[b]:
                unreliable.update(a+b)
                reasons.append(dict(joint=a[1],reason='body_bone_proportion',bones=[list(a),list(b)],
                                    lengths_m=[current[a],current[b]],profile_ready=self.frozen_lengths is not None))
        for bone,length in current.items():
            history=self.lengths[bone]
            reference=(self.frozen_lengths[bone] if self.frozen_lengths is not None else
                       float(np.median(history)) if len(history)>=3 else None)
            if not np.isfinite(length) or length<=0:
                unreliable.update(bone);reasons.append(dict(joint=bone[1],reason='body_bone_length_invalid'))
            elif reference is not None and not .65*reference <= length <= 1.35*reference:
                unreliable.update(bone)
                estimated=any(joints[j].get('estimated') for j in bone)
                reasons.append(dict(joint=bone[1],reason='estimated_leg_length' if estimated and bone[0]>=11 else
                                    'estimated_body_bone_length' if estimated else 'observed_body_bone_length',
                                    length_m=length,reference_m=reference,profile_ready=self.frozen_lengths is not None))
        for j in range(11,17):
            p=joints[j]
            if p['valid'] and not p.get('estimated') and j not in unreliable:
                history=self.observations.setdefault(j,[])
                history.append(stamp)
                del history[:-3]
        for a,b in self.BONES:
            x,y=joints[a],joints[b]
            if (self.frozen_lengths is None and (a,b) in current and a not in unreliable and b not in unreliable
                    and not x.get('estimated') and not y.get('estimated')):
                values=self.lengths[(a,b)]
                values.append(current[(a,b)])
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
