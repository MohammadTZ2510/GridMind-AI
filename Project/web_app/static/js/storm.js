/* Decorative storm, independent of the simulated weather records. */
(() => {
  'use strict';
  const canvas=document.getElementById('storm-background'),button=document.getElementById('storm-toggle');
  if(!canvas||!button)return;
  const ctx=canvas.getContext('2d');if(!ctx){button.hidden=true;return;}
  const motion=matchMedia('(prefers-reduced-motion: reduce)');
  let enabled=!motion.matches;
  try{const saved=sessionStorage.getItem('gridmind-storm');if(saved!==null)enabled=saved==='on';}catch{}
  let strikeCount=0,w=0,h=0,drops=[],bolts=[],nextStrike=performance.now()+900,last=0,frame=0,dead=false;
  function size(){w=innerWidth;h=innerHeight;const ratio=Math.min(devicePixelRatio||1,1.5);canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);ctx.setTransform(ratio,0,0,ratio,0,0);drops=Array.from({length:w<700?85:180},()=>({x:Math.random()*(w+100),y:Math.random()*h,length:18+Math.random()*38,speed:290+Math.random()*400,depth:.35+Math.random()*.65}));}
  // Successively finer deviations give bolts angular detail instead of a regular zigzag.
  function branch(x,y,endX,endY,spread,levels=6){
    let points=[{x,y},{x:endX,y:endY}];
    for(let level=0;level<levels;level++){
      const refined=[points[0]];
      for(let i=1;i<points.length;i++){const a=points[i-1],b=points[i];refined.push({x:(a.x+b.x)/2+(Math.random()-.5)*spread,y:(a.y+b.y)/2+(Math.random()-.5)*Math.abs(b.y-a.y)*.24},b);}
      points=refined;spread*=.53;
    }
    return points;
  }
  function strike(t){
    strikeCount++;window.GridmindStormAudio?.strike();
    const x=w*(.12+Math.random()*.76),endX=x+(Math.random()-.5)*w*.24;
    const path=branch(x,-15,endX,h*(.65+Math.random()*.3),w*.16),forks=[];
    for(const fraction of [.22,.4,.58,.72]){const p=path[Math.floor(path.length*fraction)],side=Math.random()<.5?-1:1;forks.push(branch(p.x,p.y,p.x+side*w*(.06+Math.random()*.13),Math.min(h,p.y+h*(.12+Math.random()*.23)),w*.055,4));}
    bolts=[{path,forks,born:t,life:1450,x}];nextStrike+=7500;if(nextStrike<=t)nextStrike=t+7500;
  }
  function intensity(b,t){const age=t-b.born;if(age<0)return 0;return Math.min(1,(1-Math.exp(-age/12))*Math.exp(-age/160)+.58*Math.exp(-Math.pow((age-210)/48,2))+.22*Math.exp(-Math.pow((age-390)/70,2)));}
  function stroke(points,opacity,width,blur,white=false){ctx.beginPath();ctx.moveTo(points[0].x,points[0].y);for(const p of points.slice(1))ctx.lineTo(p.x,p.y);ctx.lineWidth=width;ctx.strokeStyle=white?`rgba(255,255,255,${opacity})`:`rgba(255,12,48,${opacity})`;ctx.shadowColor='#ff002f';ctx.shadowBlur=blur;ctx.lineCap='round';ctx.lineJoin='round';ctx.stroke();}
  function illuminate(b,flash){
    ctx.shadowBlur=0;
    // Broad red cloud bounce and a localized pale core behind the glass interface.
    const sky=ctx.createRadialGradient(b.x,h*.08,0,b.x,h*.12,Math.max(w,h)*.95);
    sky.addColorStop(0,`rgba(255,219,228,${flash*.19})`);sky.addColorStop(.26,`rgba(240,15,52,${flash*.15})`);sky.addColorStop(1,'rgba(100,0,15,0)');
    ctx.fillStyle=sky;ctx.fillRect(0,0,w,h);
    const bounce=ctx.createRadialGradient(b.path.at(-1).x,h,0,b.x,h,Math.max(w,h)*.65);
    bounce.addColorStop(0,`rgba(255,36,67,${flash*.15})`);bounce.addColorStop(1,'rgba(70,0,0,0)');ctx.fillStyle=bounce;ctx.fillRect(0,0,w,h);
  }
  function paint(t){
    if(dead)return;frame=requestAnimationFrame(paint);
    if(!enabled||document.hidden){last=t;return;}
    if(t-last<33)return;const dt=Math.min((t-last)/1000,.07);last=t;ctx.clearRect(0,0,w,h);
    if(t>=nextStrike)strike(t);bolts=bolts.filter(b=>t-b.born<b.life);
    let flash=0;for(const b of bolts){const amount=intensity(b,t);flash=Math.max(flash,amount);illuminate(b,amount);}
    ctx.shadowBlur=1.2;ctx.shadowColor='rgba(235,235,255,.28)';ctx.lineWidth=1.15;ctx.strokeStyle='rgba(240,238,245,.58)';ctx.beginPath();
    for(const d of drops){d.y+=d.speed*dt;d.x-=d.speed*.11*dt;if(d.y>h+60){d.y=-60;d.x=Math.random()*(w+120);}if(d.x< -40)d.x=w+40;ctx.moveTo(d.x,d.y);ctx.lineTo(d.x-4*d.depth,d.y+d.length*d.depth);}ctx.stroke();
    if(flash>.015){
      // Stylized reflected/refracted glints: stronger near the bolt, with red rims.
      const x=bolts[0].x;ctx.shadowColor='#ff163c';ctx.shadowBlur=9;ctx.lineWidth=1.8;
      for(let i=0;i<drops.length;i+=2){const d=drops[i],proximity=Math.max(.12,1-Math.abs(d.x-x)/(w*.7));ctx.strokeStyle=`rgba(255,248,250,${flash*proximity*d.depth*.9})`;ctx.beginPath();ctx.moveTo(d.x,d.y);ctx.lineTo(d.x-2,d.y+d.length*d.depth*.8);ctx.stroke();}
    }
    for(const b of bolts){const alpha=intensity(b,t);stroke(b.path,alpha*.18,15,42);stroke(b.path,alpha*.65,4,24);stroke(b.path,alpha,1.6,16,true);for(const p of b.forks){stroke(p,alpha*.35,4,22);stroke(p,alpha*.85,.8,14,true);}}
    ctx.shadowBlur=0;canvas.dataset.frames=String(Number(canvas.dataset.frames||0)+1);canvas.dataset.strikes=String(strikeCount);canvas.dataset.illumination=flash.toFixed(3);
  }
  function apply(){window.GridmindStormAudio?.setStormActive(enabled);button.textContent=enabled?'ϟ Storm effects ON':'ϟ Storm effects OFF';button.setAttribute('aria-pressed',String(enabled));canvas.hidden=!enabled;document.body.classList.toggle('storm-active',enabled);bolts=[];if(!enabled)ctx.clearRect(0,0,w,h);else nextStrike=performance.now()+900;}
  button.onclick=()=>{enabled=!enabled;try{sessionStorage.setItem('gridmind-storm',enabled?'on':'off');}catch{}apply();};
  motion.addEventListener('change',e=>{if(e.matches){enabled=false;apply();}});
  document.addEventListener('visibilitychange',()=>{if(!document.hidden){bolts=[];nextStrike=performance.now()+900;last=performance.now();}});
  addEventListener('resize',size,{passive:true});addEventListener('pagehide',()=>{dead=true;cancelAnimationFrame(frame);});
  size();apply();frame=requestAnimationFrame(paint);
})();
