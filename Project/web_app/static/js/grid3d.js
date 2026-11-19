/* GRIDMIND — data-driven, schematic distribution-grid digital twin.
 * Three.js r128 + OrbitControls. No network calls and no invented feeder states.
 */
(function () {
  'use strict';
  const C = {on:0x64d9ad,off:0xf16e7a,unknown:0xd1b374,cyan:0xdc143c,blue:0x8f0024,muted:0x6d6668,select:0xffffff,diff:0xf4b75b};
  const fmt = v => v == null || !Number.isFinite(Number(v)) ? 'Unavailable' : Number(v).toFixed(2) + ' MW';
  const safe = s => String(s == null ? '' : s).replace(/[&<>"']/g, x=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));
  class Grid3D {
    constructor(container, options = {}) {
      this.container=container; this.options=options; this.nodes=new Map(); this.lines=[]; this.particles=[];
      this.payload=null; this.selected=null; this.compare=false; this.disposed=false; this.frame=0; this.last=0;
      this.container.style.position='relative';
      this.container.style.overflow='hidden';
      this.container.setAttribute('aria-label','Interactive schematic electricity distribution network. Select a feeder to inspect its recorded data.');
      this.tooltip=document.createElement('div');
      Object.assign(this.tooltip.style,{position:'absolute',zIndex:8,display:'none',pointerEvents:'none',background:'rgba(22,19,22,.97)',border:'1px solid #676063',borderRadius:'9px',padding:'12px 14px',color:'#e3cad0',font:'12px/1.7 Inter,Arial,sans-serif',boxShadow:'0 12px 32px #0008',minWidth:'205px'});
      this.container.appendChild(this.tooltip);
      this.hud=document.createElement('div');
      this.hud.style.cssText='position:absolute;left:16px;top:12px;right:120px;z-index:3;pointer-events:none;color:#cac6c7;font:10px/1.7 Inter,Arial,sans-serif;letter-spacing:.6px';
      this.hud.innerHTML='SCHEMATIC TOPOLOGY · SIMULATED DATA<br><span style="letter-spacing:.2px;color:#9b9496">Drag to orbit · Scroll to zoom · Right-drag to pan · Click a feeder</span>';
      this.container.appendChild(this.hud);
      this.resetBtn=document.createElement('button');this.resetBtn.type='button';this.resetBtn.title='Reset 3D camera';this.resetBtn.setAttribute('aria-label','Reset 3D camera');this.resetBtn.textContent='↗ Reset view';
      this.resetBtn.style.cssText='position:absolute;right:16px;top:12px;z-index:5;border:1px solid #595355;background:rgba(0,0,0,.50);color:#e3cad0;border-radius:7px;padding:8px 10px;font:11px Inter,Arial;cursor:pointer';
      this.resetBtn.addEventListener('click',()=>this.resetCamera());this.container.appendChild(this.resetBtn);
      try {
        if(!window.THREE) throw new Error('Three.js unavailable');
        const T=window.THREE;
        this.scene=new T.Scene();this.scene.background=null;this.scene.fog=new T.FogExp2(0x111012,.0095);
        this.camera=new T.PerspectiveCamera(39,1,.1,220);this.camera.position.set(38,34,43);
        this.renderer=new T.WebGLRenderer({antialias:true,alpha:true,powerPreference:'high-performance'});
        this.renderer.setClearColor(0x000000,0);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,1.75));
        this.renderer.outputEncoding=T.sRGBEncoding;this.renderer.toneMapping=T.ACESFilmicToneMapping;this.renderer.toneMappingExposure=1.35;
        this.renderer.domElement.style.cssText='display:block;width:100%;height:100%;touch-action:none';this.renderer.domElement.tabIndex=0;
        this.renderer.domElement.setAttribute('aria-label','3D grid. Feeder details are also available in the adjacent inspector and feeder table.');
        this.container.insertBefore(this.renderer.domElement,this.container.firstChild);
        this.controls=T.OrbitControls ? new T.OrbitControls(this.camera,this.renderer.domElement) : null;
        if(this.controls){this.controls.target.set(0,0,0);this.controls.enableDamping=true;this.controls.dampingFactor=.07;this.controls.maxPolarAngle=Math.PI*.47;this.controls.minDistance=22;this.controls.maxDistance=95;this.controls.enablePan=true;}
        else this.camera.lookAt(0,0,0);
        this.raycaster=new T.Raycaster();this.pointer=new T.Vector2();this.clickables=[];
        this.scene.add(new T.HemisphereLight(0xc8bbc0,0x121112,.9));
        const sun=new T.DirectionalLight(0xe4d9d6,1.4);sun.position.set(15,28,12);this.scene.add(sun);
        const fill=new T.DirectionalLight(0xa74661,.7);fill.position.set(-20,14,-15);this.scene.add(fill);
        const centralLight=new T.PointLight(0xd95077,1.6,35);centralLight.position.set(0,8,0);this.scene.add(centralLight);
        this.glowTexture=this._glowTexture();
        this.root=new T.Group();this.scene.add(this.root);
        this._buildGround();this._buildStation();
        this._onMove=e=>this._pointerMove(e);this._onLeave=()=>{this.tooltip.style.display='none';this.container.style.cursor='grab';};
        this._onDown=e=>{this.down={x:e.clientX,y:e.clientY};};
        this._onUp=e=>{if(this.down&&Math.hypot(e.clientX-this.down.x,e.clientY-this.down.y)<5){const id=this._hit(e);if(id){this.select(id);if(this.options.onSelect)this.options.onSelect(id);}}};
        this.renderer.domElement.addEventListener('pointermove',this._onMove);this.renderer.domElement.addEventListener('pointerleave',this._onLeave);this.renderer.domElement.addEventListener('pointerdown',this._onDown);this.renderer.domElement.addEventListener('pointerup',this._onUp);
        this.renderer.domElement.addEventListener('webglcontextlost',this._lost=e=>{e.preventDefault();this.fallback=true;this._showFallback('WebGL context unavailable');});
        this.observer=new ResizeObserver(()=>this.resize());this.observer.observe(this.container);this.resize();this._animate(0);
      } catch(error) {this.fallback=true;this._showFallback('3D rendering is unavailable on this device');}
    }
    _mat(color,emissive=0,intensity=0){return new THREE.MeshStandardMaterial({color,emissive,emissiveIntensity:intensity,roughness:.63,metalness:.45});}
    _box(parent,w,h,d,x,y,z,material){const mesh=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),material);mesh.position.set(x,y+h/2,z);parent.add(mesh);return mesh;}
    _outline(parent,w,h,d,x,y,z,color=0x5e585a){const geo=new THREE.EdgesGeometry(new THREE.BoxGeometry(w,h,d));const line=new THREE.LineSegments(geo,new THREE.LineBasicMaterial({color,transparent:true,opacity:.65}));line.position.set(x,y+h/2,z);parent.add(line);return line;}
    _glowTexture(){const c=document.createElement('canvas');c.width=c.height=96;const ctx=c.getContext('2d'),g=ctx.createRadialGradient(48,48,0,48,48,48);g.addColorStop(0,'rgba(255,255,255,.95)');g.addColorStop(.1,'rgba(255,255,255,.55)');g.addColorStop(.35,'rgba(255,255,255,.12)');g.addColorStop(1,'rgba(255,255,255,0)');ctx.fillStyle=g;ctx.fillRect(0,0,96,96);return new THREE.CanvasTexture(c);}
    _glow(parent,color,x,y,z,size=3){const m=new THREE.SpriteMaterial({map:this.glowTexture,color,transparent:true,opacity:.65,depthWrite:false,blending:THREE.AdditiveBlending});const s=new THREE.Sprite(m);s.position.set(x,y,z);s.scale.set(size,size,1);parent.add(s);return s;}
    _label(text,color='#cfa4af',size=3.8){const c=document.createElement('canvas');c.width=512;c.height=96;const ctx=c.getContext('2d');ctx.fillStyle=color;ctx.font='600 29px Arial';ctx.textAlign='center';ctx.textBaseline='middle';ctx.shadowColor='#101010';ctx.shadowBlur=8;ctx.fillText(text,256,48);const tex=new THREE.CanvasTexture(c);const s=new THREE.Sprite(new THREE.SpriteMaterial({map:tex,transparent:true,depthTest:false,depthWrite:false}));s.scale.set(size,size*96/512,1);s.userData.labelText=text;return s;}
    _ring(parent,radius,y,color,opacity=1){const geo=new THREE.BufferGeometry();const pts=[];for(let i=0;i<=100;i++){let a=i/100*Math.PI*2;pts.push(Math.cos(a)*radius,y,Math.sin(a)*radius);}geo.setAttribute('position',new THREE.Float32BufferAttribute(pts,3));const line=new THREE.Line(geo,new THREE.LineBasicMaterial({color,transparent:true,opacity}));parent.add(line);return line;}
    _buildGround(){
      const T=THREE,base=this._mat(0x17171a);this._box(this.root,49,.8,42,0,-1,0,base);this._outline(this.root,49,.8,42,0,-1,0,0x7a7274);
      const grid=new T.GridHelper(48,48,0x2f2a2b,0x262123);grid.position.y=-.15;grid.material.transparent=true;grid.material.opacity=.7;this.root.add(grid);
      const frame=new T.Group();this._ring(frame,3.7,.08,0x7c7476,.7);this._ring(frame,4.2,.08,0x373133,.6);this.root.add(frame);
      // Discrete road markings and perimeter illumination add depth without claiming map accuracy.
      for(let i=-22;i<=22;i+=2){this._box(this.root,.7,.035,.035,i,-.08,19.8,this._mat(0x7d7477,0x7d7477,.5));}
      for(const x of [-23.7,23.7])for(const z of [-19.5,19.5])this._glow(this.root,0xc64166,x,.1,z,2);
      for(let i=0;i<60;i++){const x=-24+(i*13.73)%48,z=-20+(i*9.37)%40;const m=new T.Mesh(new T.SphereGeometry(.018,4,4),new T.MeshBasicMaterial({color:0x847c7e}));m.position.set(x,.03,z);this.root.add(m);}
    }
    _buildStation(){
      const T=THREE,g=new T.Group();this.station=g;this.root.add(g);
      this._box(g,6.9,.3,5.2,0,0,0,this._mat(0x262224));this._outline(g,6.9,.3,5.2,0,0,0,0x8b8285);
      this._box(g,3.5,1.8,2.5,-.6,.3,-.3,this._mat(0x332e30));this._outline(g,3.5,1.8,2.5,-.6,.3,-.3,0x9c9597);
      this._box(g,3.8,.12,2.8,-.6,2.1,-.3,this._mat(0x7c7376,0x363032,.5));
      for(let i=0;i<4;i++)this._box(g,.42,.42,1.5,-1.8+i*.8,2.22,-.3,this._mat(0x746c6e));
      for(let i=0;i<3;i++){
        const x=-1.9+i*1.3;
        this._box(g,.75,.85,.7,x,.3,1.8,this._mat(0x635c5e));
        for(let k=0;k<5;k++)this._box(g,.85,.035,.85,x,.4+k*.11,1.8,this._mat(0x8a8184));
        const ins=new T.Mesh(new T.CylinderGeometry(.12,.15,.52,10),this._mat(0x999194));ins.position.set(x,1.38,1.8);g.add(ins);
      }
      this._tower(g,2.35,-.65,5.7);
      const label=this._label('CENTRAL POWER STATION','#d9c7cb',6.2);label.position.set(0,5.3,0);g.add(label);
      this.stationGlow=this._glow(g,0xd74670,0,1.8,0,9);
      this.stationRing=this._ring(g,3.55,.12,C.cyan,.7);
    }
    _tower(group,x,z,height=4){
      const T=THREE,mat=new T.LineBasicMaterial({color:0xada7a8,transparent:true,opacity:.8}),pts=[];
      const beam=(a,b)=>pts.push(...a,...b);
      const half=height*.15;
      for(const sx of [-1,1])for(const sz of [-1,1])beam([x+sx*half,.3,z+sz*half],[x+sx*.15,height,z+sz*.15]);
      for(let y=1;y<height;y+=.8){let r=half*(1-y/(height+1));beam([x-r,y,z-r],[x+r,y+.6,z+r]);beam([x+r,y,z-r],[x-r,y+.6,z+r]);}
      beam([x-1.25,height-.5,z],[x+1.25,height-.5,z]);beam([x-.95,height+.15,z],[x+.95,height+.15,z]);
      const geo=new T.BufferGeometry();geo.setAttribute('position',new T.Float32BufferAttribute(pts,3));group.add(new T.LineSegments(geo,mat));
      for(const dx of [-1.05,0,1.05])this._glow(group,0xbf526e,x+dx,height-.56,z,.4);
    }
    _buildAreas(feeders){
      if(this.areaRoot){this.root.remove(this.areaRoot);this._disposeTree(this.areaRoot);}this.nodes.clear();this.clickables=[];this.lines=[];this.particles=[];
      this.areaRoot=new THREE.Group();this.root.add(this.areaRoot);
      const areas=[...new Set(feeders.map(f=>f.area))];
      const positions=[[-14,-10],[0,-14.5],[14,-9],[-12,11.2],[12,11.2]];
      areas.forEach((name,index)=>{
        const [x,z]=positions[index%positions.length],group=new THREE.Group();group.position.set(x,0,z);this.areaRoot.add(group);
        this._box(group,9.6,.2,7.3,0,-.05,0,this._mat(0x211d1e));this._outline(group,9.6,.2,7.3,0,-.05,0,0x60595b);
        for(const px of [-4.65,4.65])this._box(group,.06,.04,6.8,px,.18,0,this._mat(0x5e585a,0x312b2d,.5));
        const subZ=z<0?2.4:-2.4;
        this._box(group,2,.6,1.2,0,.2,subZ,this._mat(0x585254));this._outline(group,2,.6,1.2,0,.2,subZ,0x857d7f);
        this._tower(group,0,subZ,3.3);
        const label=this._label(name.toUpperCase(),'#cea3ae',7.0);label.position.set(0,.45,z<0?-4.25:4.25);group.add(label);
        const members=feeders.filter(f=>f.area===name);
        members.forEach((f,i)=>{
          const nx=i===0?-2.5:2.5,nz=z<0?-.65:.75;
          this._buildFeeder(group,f,nx,nz);
          const start=new THREE.Vector3(0,1.15,subZ),end=new THREE.Vector3(nx,.6,nz);
          const link=this._cable(group,start,end,.7,C.muted);link.feederId=f.feeder_id;this.lines.push(link);
        });
        const start=new THREE.Vector3(2.35,4.8,-.65),end=new THREE.Vector3(x,3.1,z+subZ);
        const trunk=this._cable(this.areaRoot,start,end,1.1,0x7f7679);trunk.area=name;trunk.trunk=true;this.lines.push(trunk);
        // Two thinner parallel conductors evoke a transmission corridor.
        for(const dx of [-.3,.3]){
          const a=start.clone().add(new THREE.Vector3(dx,0,0)),b=end.clone().add(new THREE.Vector3(dx,0,0));
          this._cable(this.areaRoot,a,b,.9,0x585254,false);
        }
      });
    }
    _buildFeeder(parent,f,x,z){
      const T=THREE,g=new T.Group();g.position.set(x,.2,z);parent.add(g);
      const critical=!!f.is_critical;
      const platform=this._box(g,3.7,.14,3.65,0,0,0,this._mat(0x2d2829));
      const ring=this._ring(g,2.12,.2,C.unknown,.75);
      const statusMat=this._mat(0x676063,C.unknown,.6);
      const h=critical?2.3:1.1+((Number(f.feeder_id.slice(1))%3)*.48);
      this._box(g,1.9,h,1.7,-.35,.15,-.25,this._mat(critical?0x655e60:0x322d2f));
      this._outline(g,1.9,h,1.7,-.35,.15,-.25,0x7f7779);
      this._box(g,2.05,.13,1.85,-.35,h+.15,-.25,statusMat);
      // Lit horizontal window strips, controlled by the feeder's true state.
      const windows=[];
      for(let j=.45;j<h;j+=.45){windows.push(this._box(g,1.5,.08,.025,-.35,j,.615,this._mat(0x7a7275,C.unknown,.35)));windows.push(this._box(g,.025,.08,1.25,.615,j,-.25,this._mat(0x7a7275,C.unknown,.35)));}
      const annex=this._box(g,.85,.65,1.1,1,.15,.45,this._mat(0x5d5759));this._outline(g,.85,.65,1.1,1,.15,.45,0x71696c);
      this._box(g,.9,.07,1.2,1,.8,.45,this._mat(0x756d70));
      if(critical){
        const cross=this._mat(0xe3cad0,0xd0a6b1,.75);this._box(g,.18,.72,.04,-.35,1.05,.63,cross);this._box(g,.65,.18,.045,-.35,1.32,.63,cross);
        this._box(g,.6,.05,.04,-.35,h+.32,-.25,cross);this._box(g,.05,.05,.6,-.35,h+.32,-.25,cross);
      }else{
        for(let k=0;k<2;k++){const tank=new T.Mesh(new T.CylinderGeometry(.22,.22,.55,12),this._mat(0x877e80));tank.position.set(-.8+k*.8,h+.55,-.25);g.add(tank);}
      }
      const beacon=new T.Mesh(new T.SphereGeometry(.15,12,8),new T.MeshBasicMaterial({color:C.unknown}));beacon.position.set(1,.99,.45);g.add(beacon);
      const halo=this._glow(g,C.unknown,1,1.05,.45,2.6);
      const label=this._label(f.feeder_id+(critical?' · HOSPITAL':''),'#e3cad0',critical?4.4:3.3);label.position.set(0,h+1.1,0);g.add(label);
      const selection=this._ring(g,2.35,.22,C.select,.0);
      const diff=this._ring(g,2.55,.23,C.diff,0);
      g.traverse(obj=>{if(obj.isMesh){obj.userData.feederId=f.feeder_id;this.clickables.push(obj);}});
      this.nodes.set(f.feeder_id,{group:g,data:f,ring,selection,diff,beacon,halo,statusMat,windows,color:new T.Color(C.unknown),target:new T.Color(C.unknown),status:null});
    }
    _cable(parent,start,end,sag,color,particles=true){
      const T=THREE,mid=start.clone().lerp(end,.5);mid.y=Math.max(.45,mid.y-sag);
      const curve=new T.CatmullRomCurve3([start,mid,end]);
      const geo=new T.BufferGeometry().setFromPoints(curve.getPoints(38));
      const line=new T.Line(geo,new T.LineBasicMaterial({color,transparent:true,opacity:.72}));parent.add(line);
      const result={line,curve,particles:[],color:new T.Color(color),target:new T.Color(color),enabled:true};
      if(particles)for(let i=0;i<4;i++){
        const p=new T.Mesh(new T.SphereGeometry(.055,5,4),new T.MeshBasicMaterial({color:C.cyan}));parent.add(p);
        result.particles.push({mesh:p,offset:i/4});
      }
      return result;
    }
    update(payload){
      this.payload=payload;if(payload.compare!==undefined)this.compare=!!payload.compare;
      if(this.fallback){this._showFallback();return;}
      const feeders=payload.feeders||[];
      const signature=feeders.map(f=>f.feeder_id+f.area).join('|');
      if(signature!==this.signature){this.signature=signature;this._buildAreas(feeders);}
      feeders.forEach(f=>{const n=this.nodes.get(f.feeder_id);if(!n)return;n.data=f;n.status=f.decision;n.target.setHex(f.decision==='ON'?C.on:f.decision==='OFF'?C.off:C.unknown);n.diff.material.opacity=this.compare&&f.different?.8:0;});
      this.lines.forEach(link=>{
        if(link.trunk){const members=feeders.filter(f=>f.area===link.area);link.enabled=members.some(f=>f.decision==='ON');link.target.setHex(link.enabled?0x999194:C.muted);}
        else{const f=feeders.find(f=>f.feeder_id===link.feederId);link.enabled=!!f&&f.decision==='ON';link.target.setHex(f&&f.decision==='OFF'?0x6a3446:link.enabled?0x467d64:C.muted);}
      });
      const shortage=payload.totals&&payload.totals.forecast_shortage_mw;
      this.stationRing.material.color.setHex(shortage>0?C.diff:C.cyan);
      this.hud.firstChild.textContent=(payload.scenario_id||'')+' · '+String(payload.hour??0).padStart(2,'0')+':00 · '+(payload.method||'CSP')+' · SCHEMATIC';
      this.select(this.selected);
    }
    setComparison(value){this.compare=!!value;if(this.payload)this.update(this.payload);}
    select(id){this.selected=id||null;this.nodes.forEach((n,k)=>{n.selection.material.opacity=k===id?.9:0;});if(this.fallback&&this.payload)this._showFallback();}
    focusFeeder(id){this.select(id);const n=this.nodes.get(id);if(!n||this.fallback)return;const target=new THREE.Vector3();n.group.getWorldPosition(target);this.focusTarget=target;}
    resetCamera(){if(this.camera){this.camera.position.set(38,34,43);if(this.controls){this.controls.target.set(0,0,0);this.controls.update();}else this.camera.lookAt(0,0,0);}this.focusTarget=null;}
    resize(){if(!this.renderer||this.fallback)return;const w=this.container.clientWidth||800,h=this.container.clientHeight||500;this.camera.aspect=w/h;this.camera.updateProjectionMatrix();this.renderer.setSize(w,h,false);}
    _hit(event){if(!this.renderer||!this.raycaster)return null;const rect=this.renderer.domElement.getBoundingClientRect();this.pointer.set(((event.clientX-rect.left)/rect.width)*2-1,-((event.clientY-rect.top)/rect.height)*2+1);this.raycaster.setFromCamera(this.pointer,this.camera);const hits=this.raycaster.intersectObjects(this.clickables,false);return hits.length?hits[0].object.userData.feederId:null;}
    _pointerMove(event){const id=this._hit(event),n=this.nodes.get(id);this.container.style.cursor=id?'pointer':'grab';if(!n){this.tooltip.style.display='none';return;}const f=n.data;this.tooltip.innerHTML='<div style="display:flex;justify-content:space-between;gap:24px"><strong style="color:#e3cad0">'+safe(f.feeder_id)+'</strong><span style="color:'+(f.decision==='ON'?'#64d9ad':f.decision==='OFF'?'#f16e7a':'#d1b374')+'">'+(f.decision||'NO SCHEDULE')+'</span></div><div style="color:#b6b1b3;margin-bottom:5px">'+safe(f.area)+(f.is_critical?' · HOSPITAL':'')+'</div>Actual <b style="float:right">'+fmt(f.actual_demand_mw)+'</b><br>Forecast <b style="float:right">'+fmt(f.regression_prediction_mw)+'</b><br>Served <b style="float:right">'+fmt(f.served_demand_mw)+'</b><br>Unmet <b style="float:right">'+fmt(f.unmet_demand_mw)+'</b><br><span style="color:#b6b1b3">Priority '+safe(f.priority)+' · Click to inspect</span>';const rect=this.container.getBoundingClientRect();this.tooltip.style.display='block';this.tooltip.style.left=Math.min(this.container.clientWidth-235,Math.max(10,event.clientX-rect.left+15))+'px';this.tooltip.style.top=Math.max(10,Math.min(this.container.clientHeight-210,event.clientY-rect.top-45))+'px';}
    _animate(timestamp){
      if(this.disposed)return;this.frame=requestAnimationFrame(t=>this._animate(t));if(this.fallback||document.hidden||timestamp-this.last<25)return;this.last=timestamp;const t=timestamp*.001;
      if(this.controls){if(this.focusTarget){this.controls.target.lerp(this.focusTarget,.045);if(this.controls.target.distanceTo(this.focusTarget)<.02)this.focusTarget=null;}this.controls.update();}
      this.nodes.forEach(n=>{n.color.lerp(n.target,.12);n.ring.material.color.copy(n.color);n.statusMat.emissive.copy(n.color);n.beacon.material.color.copy(n.color);n.halo.material.color.copy(n.color);n.halo.material.opacity=n.status==='ON'?.48+Math.sin(t*1.8)*.08:.23;n.windows.forEach(w=>{w.material.emissive.copy(n.color);w.material.emissiveIntensity=n.status==='ON'?.65:.1;});n.selection.rotation.y=t*.15;});
      this.lines.forEach(l=>{l.color.lerp(l.target,.13);l.line.material.color.copy(l.color);l.particles.forEach(p=>{p.mesh.visible=l.enabled;p.mesh.position.copy(l.curve.getPoint((t*.16+p.offset)%1));});});
      if(this.stationGlow)this.stationGlow.material.opacity=.35+Math.sin(t*.8)*.035;
      this._layoutLabels();this.renderer.render(this.scene,this.camera);
    }
    _layoutLabels(){
      // Suppress labels that overlap in screen space as the user orbits or zooms.
      // The complete feeder names and measurements remain in the inspector/tooltip.
      const labels=[],w=this.container.clientWidth,h=this.container.clientHeight;
      if(!w||!h)return;
      this.root.traverse(o=>{if(o.userData.labelText){const world=new THREE.Vector3();o.getWorldPosition(world);const p=world.clone().project(this.camera);const distance=this.camera.position.distanceTo(world);const pxPerUnit=h/(2*Math.max(distance,.1)*Math.tan(this.camera.fov*Math.PI/360));const width=o.scale.x*pxPerUnit,height=o.scale.y*pxPerUnit;const text=o.userData.labelText;labels.push({sprite:o,x:(p.x+1)*w/2,y:(1-p.y)*h/2,width,height,valid:p.z>-1&&p.z<1,priority:text.startsWith('F')?3:text.startsWith('CENTRAL')?1:2});}});
      labels.sort((a,b)=>b.priority-a.priority);const occupied=[];
      for(const l of labels){const r={left:l.x-l.width/2-4,right:l.x+l.width/2+4,top:l.y-l.height/2-4,bottom:l.y+l.height/2+4};const overlap=occupied.some(b=>r.left<b.right&&r.right>b.left&&r.top<b.bottom&&r.bottom>b.top);l.sprite.visible=l.valid&&!overlap&&r.left>4&&r.right<w-4&&r.top>55&&r.bottom<h-4;if(l.sprite.visible)occupied.push(r);}
    }
    _showFallback(reason){
      if(this.renderer)this.renderer.domElement.style.display='none';if(this.tooltip)this.tooltip.style.display='none';
      if(!this.fallbackEl){this.fallbackEl=document.createElement('div');this.fallbackEl.style.cssText='position:absolute;inset:0;background:radial-gradient(ellipse at center,rgba(35,0,8,.25),rgba(0,0,0,.23) 75%);padding:70px 20px 30px;overflow:auto';this.container.insertBefore(this.fallbackEl,this.container.firstChild);}
      if(!this.payload){this.fallbackEl.innerHTML='<div style="color:#cac6c7;padding:80px 10px;text-align:center">Loading grid topology…</div>';return;}
      const feeders=this.payload.feeders||[],areas=[...new Set(feeders.map(f=>f.area))];
      this.fallbackEl.innerHTML='<div style="text-align:center;font:10px Inter,Arial;letter-spacing:1.8px;color:#cbc8c9;margin-bottom:18px">INTERACTIVE SCHEMATIC · WEBGL FALLBACK</div><div style="margin:0 auto 20px;width:190px;border:1px solid #9a9295;background:rgba(0,0,0,.50);padding:15px;text-align:center;color:#e8c2cd;border-radius:8px;box-shadow:0 0 30px #8e868918">⚡ CENTRAL STATION</div><div style="border-top:1px solid #6e6668;margin:0 10% 24px"></div><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:14px">'+areas.map(area=>'<section style="border:1px solid #595355;border-radius:10px;padding:14px;background:#1e1b1c"><h4 style="font:11px Inter,Arial;letter-spacing:1px;color:#dad7d8;margin:0 0 14px">'+safe(area.toUpperCase())+'</h4>'+feeders.filter(f=>f.area===area).map(f=>{const color=f.decision==='ON'?'#64d9ad':f.decision==='OFF'?'#f16e7a':'#d1b374';return '<button type="button" data-grid-feeder="'+safe(f.feeder_id)+'" style="width:100%;padding:10px;margin:4px 0;border:1px solid '+(f.feeder_id===this.selected?'#e3cad0':this.compare&&f.different?'#f4b75b':'#5c5557')+';border-radius:6px;background:rgba(0,0,0,.50);color:#e3cad0;cursor:pointer;text-align:left"><strong>'+safe(f.feeder_id)+(f.is_critical?' ✚':'')+'</strong><span style="float:right;font-size:10px;color:'+color+'">'+(f.decision||'UNKNOWN')+'</span><span style="display:block;margin-top:6px;color:#bab5b6;font-size:11px">'+fmt(f.actual_demand_mw)+' actual</span></button>';}).join('')+'</section>').join('')+'</div><p style="color:#b2acae;font:11px/1.6 Inter,Arial;margin-top:16px">All feeder states are read from the selected algorithm’s saved schedule. Unknown means no schedule is available.</p>';
      this.fallbackEl.querySelectorAll('[data-grid-feeder]').forEach(btn=>btn.addEventListener('click',()=>{this.select(btn.dataset.gridFeeder);if(this.options.onSelect)this.options.onSelect(btn.dataset.gridFeeder);}));
      this.hud.innerHTML='SCHEMATIC TOPOLOGY · SIMULATED DATA<br><span style="letter-spacing:.2px;color:#9b9496">Select a feeder to inspect · 3D unavailable; all data controls remain functional</span>';
    }
    _disposeTree(root){root.traverse(o=>{if(o.geometry)o.geometry.dispose();if(o.material){const mats=Array.isArray(o.material)?o.material:[o.material];mats.forEach(m=>{if(m.map&&m.map!==this.glowTexture)m.map.dispose();m.dispose();});}});}
    dispose(){this.disposed=true;cancelAnimationFrame(this.frame);if(this.observer)this.observer.disconnect();if(this.controls)this.controls.dispose();if(this.root)this._disposeTree(this.root);if(this.glowTexture)this.glowTexture.dispose();if(this.renderer){this.renderer.domElement.removeEventListener('pointermove',this._onMove);this.renderer.domElement.removeEventListener('pointerleave',this._onLeave);this.renderer.domElement.removeEventListener('pointerdown',this._onDown);this.renderer.domElement.removeEventListener('pointerup',this._onUp);this.renderer.dispose();this.renderer.domElement.remove();}for(const e of [this.tooltip,this.hud,this.resetBtn,this.fallbackEl])if(e)e.remove();}
    destroy(){this.dispose();}
  }
  window.Grid3D=Grid3D;
})();
