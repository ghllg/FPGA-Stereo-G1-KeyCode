// WebSocket-compatible facade; rendering and window focus cannot pace uploads.
export class FrameSocket {
  constructor(url,onProgress,onEnded,onFailure){
    this.readyState=0;this.bufferedAmount=0;
    this.worker=new Worker('/static/frame-sender.js');
    this.worker.onmessage=event=>{
      const p=event.data;
      if(p.type==='open'){this.readyState=1;this.onopen?.();}
      else if(p.type==='close'){this.readyState=3;this.onclose?.();this.worker.terminate();}
      else if(p.type==='message')this.onmessage?.({data:p.data});
      else if(p.type==='progress')onProgress(p);
      else if(p.type==='ended')onEnded();
      else if(p.type==='failure')onFailure(p.message);
    };
    this.worker.onerror=event=>onFailure(`动作发送线程失败：${event.message}`);
    this.worker.postMessage({type:'connect',url});
    this.url=url;
  }
  send(data){this.worker.postMessage({type:'send',data});}
  play(clip,base,seq){this.worker.postMessage({type:'play',frames:clip.frames,dt:clip.dt,base,seq,mainOrigin:performance.timeOrigin});}
  stop(){this.worker.postMessage({type:'stop'});}
}
