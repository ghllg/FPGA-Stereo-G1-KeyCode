import {Vector3, Quaternion, Matrix4} from './vendor/three.module.js';

const vector = a => new Vector3(...a);
const quaternion = a => new Quaternion(a[1], a[2], a[3], a[0]);
const wxyz = q => [q.w, q.x, q.y, q.z];
const mix = (a, b, t) => vector(a).lerp(vector(b), t).toArray();

// Causal presentation only: no future-frame buffer or extrapolation.
// Robot rotations are blended in their parent frames, preserving rigid links.
export class PoseSmoother {
  constructor(geoms, durationMs = 80) {
    this.geoms = geoms;
    this.durationMs = durationMs;
    this.enabled = true;
    this.reset();
  }
  reset(meta = this.meta) {
    this.meta = meta;
    this.target = null;
    this.active = false;
    this.lowCount = 0;
    this.lastArrival = null;
    this.lowRate = false;
  }
  setEnabled(enabled) {
    this.enabled = enabled;
    this.active = false;
  }
  localBodies(p) {
    const positions = p.body_pos.map(vector), rotations = p.body_quat.map(quaternion);
    return positions.map((position, i) => {
      const parent = this.meta.body_parents[i];
      if (i === 0 || parent < 0) return {position, rotation: rotations[i]};
      const inverse = rotations[parent].clone().invert();
      return {position: position.clone().sub(positions[parent]).applyQuaternion(inverse),
              rotation: inverse.multiply(rotations[i])};
    });
  }
  push(p, now, intervalHintMs = null) {
    const previous = this.target;
    const sourceDelta = previous && Number.isFinite(p.source_time) && Number.isFinite(previous.source_time)
      ? (p.source_time - previous.source_time) * 1000 : null;
    const steps = previous ? Math.max(1, p.seq - previous.seq) : 1;
    const interval = intervalHintMs ?? (sourceDelta > 0 ? sourceDelta / steps :
      this.lastArrival === null ? 0 : (now - this.lastArrival) / steps);
    this.lowRate = interval >= 80 && interval <= 500;
    this.lowCount = this.lowRate ? this.lowCount + 1 : 0;
    const gap = this.lastArrival !== null && now - this.lastArrival > 500;
    const rewind = previous && (p.seq <= previous.seq || sourceDelta < 0);
    const shouldBlend = this.enabled && previous && this.lowRate && !gap && !rewind &&
      (intervalHintMs !== null || this.lowCount >= 2);
    const start = shouldBlend ? this.sample(now) : null;
    this.target = p;
    this.lastArrival = now;
    this.active = !!shouldBlend;
    if (this.active) {
      this.start = start;
      this.startBodies = this.localBodies(start);
      this.targetBodies = this.localBodies(p);
      this.startedAt = now;
    }
  }
  sample(now) {
    if (!this.active) return this.target;
    const linear = Math.max(0, Math.min(1, (now - this.startedAt) / this.durationMs));
    if (linear === 1) { this.active = false; return this.target; }
    const t = linear * linear * (3 - 2 * linear);
    const a = this.start, b = this.target, positions = [], rotations = [];
    for (let i = 0; i < this.targetBodies.length; i++) {
      const parent = this.meta.body_parents[i];
      const position = this.startBodies[i].position.clone().lerp(this.targetBodies[i].position, t);
      const rotation = this.startBodies[i].rotation.clone().slerp(this.targetBodies[i].rotation, t);
      if (i > 0 && parent >= 0) {
        position.applyQuaternion(rotations[parent]).add(positions[parent]);
        rotation.premultiply(rotations[parent]);
      }
      positions.push(position); rotations.push(rotation);
    }
    const human = [];
    for (let i = 0; i < b.human_pos.length; i++) {
      const parent = this.meta.human_parents[i];
      if (parent < 0) { human.push(mix(a.human_pos[i], b.human_pos[i], t)); continue; }
      const va = vector(a.human_pos[i]).sub(vector(a.human_pos[parent]));
      const vb = vector(b.human_pos[i]).sub(vector(b.human_pos[parent]));
      const la = va.length(), lb = vb.length();
      if (la > 1e-8 && lb > 1e-8) {
        const swing = new Quaternion().setFromUnitVectors(va.clone().normalize(), vb.clone().normalize());
        va.normalize().applyQuaternion(new Quaternion().slerp(swing, t)).multiplyScalar(la + (lb-la)*t);
      } else va.lerp(vb, t);
      human.push(va.add(vector(human[parent])).toArray());
    }
    const geom_pos = b.geom_pos.slice(), geom_mat = b.geom_mat.slice();
    for (const g of this.geoms) {
      geom_pos[g.index] = vector(g.local_pos).applyQuaternion(rotations[g.body]).add(positions[g.body]).toArray();
      const q = rotations[g.body].clone().multiply(quaternion(g.local_quat));
      const m = new Matrix4().makeRotationFromQuaternion(q).elements;
      geom_mat[g.index] = [m[0],m[4],m[8],m[1],m[5],m[9],m[2],m[6],m[10]];
    }
    return {...b, root_pos: mix(a.root_pos,b.root_pos,t), body_pos: positions.map(v=>v.toArray()),
      body_quat: rotations.map(wxyz), human_pos: human, geom_pos, geom_mat};
  }
}
