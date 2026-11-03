import {PoseSmoother} from './smoothing.js?v=20261014';

// Presentation only. Interpolate received physics states; never extrapolate or
// feed this pose back to the controller, export, fall detection or ground trail.
export class TimedDisplay extends PoseSmoother {
  constructor(geoms, delayMs=30) {
    super(geoms,1);
    this.delayMs=delayMs;
    this.enabled=true;
  }
  reset(meta=this.meta) {
    super.reset(meta);
    this.queue=[];this.clockOffset=null;this.displayTime=-Infinity;
  }
  push(p,now) {
    const stamp=p.simulation_time*1000;
    if(!Number.isFinite(stamp))return;
    const last=this.queue.at(-1);
    if(last && stamp<=last.stamp)this.reset();
    if(this.clockOffset===null || (this.lastArrival!==null && now-this.lastArrival>500)) {
      this.queue=[];this.clockOffset=now-stamp;this.displayTime=stamp;
    }
    this.target=p;this.lastArrival=now;
    this.queue.push({stamp,pose:p,bodies:this.localBodies(p)});
    if(this.queue.length>64)this.queue.shift();
  }
  sample(now) {
    if(!this.target || !this.enabled)return this.target;
    const last=this.queue.at(-1);
    const stamp=Math.min(last.stamp,Math.max(this.displayTime,now-this.clockOffset-this.delayMs));
    this.displayTime=stamp;
    while(this.queue.length>2 && this.queue[1].stamp<=stamp)this.queue.shift();
    const a=this.queue[0],b=this.queue[1];
    if(!b || stamp<=a.stamp)return a.pose;
    if(stamp>=b.stamp)return b.pose;
    this.start=a.pose;this.target=b.pose;
    this.startBodies=a.bodies;this.targetBodies=b.bodies;
    this.startedAt=a.stamp;this.durationMs=b.stamp-a.stamp;this.active=true;
    const result=super.sample(stamp,true);
    this.target=last.pose;
    result.source_time=a.pose.source_time+(b.pose.source_time-a.pose.source_time)*(stamp-a.stamp)/(b.stamp-a.stamp);
    return result;
  }
}
