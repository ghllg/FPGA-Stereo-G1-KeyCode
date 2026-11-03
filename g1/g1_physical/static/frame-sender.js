// Own the input socket and clock outside the rendering/UI thread. The backend
// still receives only the current frame, never a precomputed reference sequence.
let socket, playback=null, timer=null;
function stop(){clearTimeout(timer);timer=null;playback=null;}
function pump(){
  const p=playback;
  if(!p || socket.readyState!==1)return;
  const index=Math.min(p.frames.length-1,p.base+Math.floor((performance.now()-p.start)/1000/p.dt));
  if(index>p.last){
    if(socket.bufferedAmount>262144){stop();postMessage({type:'failure',message:'上传网络积压，请重新开始。'});return;}
    p.skipped+=Math.max(0,index-p.last-1);
    socket.send(JSON.stringify({type:'frame',seq:p.seq++,source_time:index*p.dt,
      client_time:performance.timeOrigin+performance.now()-p.mainOrigin,values:p.frames[index]}));
    p.last=index;
    postMessage({type:'progress',frame:index,seq:p.seq,skipped:p.skipped});
  }
  if(p.base+(performance.now()-p.start)/1000/p.dt>=p.frames.length){
    socket.send(JSON.stringify({type:'end'}));stop();postMessage({type:'ended'});return;
  }
  timer=setTimeout(pump,Math.min(10,p.dt*1000));
}
onmessage=event=>{
  const p=event.data;
  if(p.type==='connect'){
    socket=new WebSocket(p.url);
    socket.onopen=()=>postMessage({type:'open'});
    socket.onclose=()=>{stop();postMessage({type:'close'});};
    socket.onmessage=e=>{
      const packet=JSON.parse(e.data);
      if(packet.type==='error'||packet.type==='reset'||(packet.type==='pose'&&packet.finished))stop();
      postMessage({type:'message',data:e.data});
    };
  }else if(p.type==='send'){
    const packet=JSON.parse(p.data);
    if(['release','init','end'].includes(packet.type))stop();
    if(socket.readyState===1)socket.send(p.data);
  }else if(p.type==='play'){
    stop();playback={...p,start:performance.now(),last:p.base,skipped:0};pump();
  }else if(p.type==='stop')stop();
};
