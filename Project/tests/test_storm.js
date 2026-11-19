// Run with: node tests/test_storm.js (no browser, network or extra packages).
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
const source=fs.readFileSync(path.join(__dirname,'../web_app/static/js/storm.js'),'utf8');
function setup(reduced=false){
 const fills=[],strokes=[],classes=new Set();let nextFrame=null;
 const ctx={createRadialGradient(){return {addColorStop(){}};},fillRect(){fills.push(this.fillStyle);},setTransform(){},clearRect(){},beginPath(){},moveTo(){},lineTo(){},stroke(){strokes.push({colour:this.strokeStyle,shadow:this.shadowColor,blur:this.shadowBlur,width:this.lineWidth});}};
 const canvas={dataset:{},getContext:()=>ctx,hidden:false};
 const button={setAttribute(k,v){this[k]=v;},hidden:false};
 const sandbox={window:{},document:{addEventListener(){},hidden:false,getElementById:id=>id==='storm-background'?canvas:button,body:{classList:{toggle(name,on){on?classes.add(name):classes.delete(name);}}}},matchMedia:()=>({matches:reduced,addEventListener(){}}),sessionStorage:{getItem:()=>null,setItem(){}},performance:{now:()=>1000},innerWidth:1200,innerHeight:800,devicePixelRatio:1,Math,requestAnimationFrame:f=>(nextFrame=f,1),cancelAnimationFrame(){},addEventListener(){}};
 vm.createContext(sandbox);vm.runInContext(source,sandbox);
 return {canvas,button,strokes,fills,frame:t=>nextFrame(t)};
}
const s=setup();assert.equal(s.button['aria-pressed'],'true');s.frame(1100);s.frame(2050);s.frame(2090);
assert.equal(s.canvas.dataset.strikes,'1');
assert(s.strokes.some(x=>x.colour.startsWith('rgba(255,255,255,')&&x.shadow==='#ff002f'&&x.blur>0),'White lightning cores must have a red shadow');
assert(s.strokes.some(x=>x.colour.startsWith('rgba(255,12,48,')&&x.blur>=20),'Red halo layer must be drawn');
assert(s.strokes.some(x=>x.colour==='rgba(240,238,245,.58)'),'Rain streaks must be drawn');
assert(s.strokes.some(x=>x.colour==='rgba(240,238,245,.58)'&&x.width>=1.1),'Rain must use brighter, thicker streaks');
assert(s.fills.length>=2,'Cloud and ground bounce gradients must render');assert(Number(s.canvas.dataset.illumination)>.1,'Strike must illuminate the sky');assert(s.strokes.some(x=>x.shadow==='#ff163c'),'Raindrops must catch reflected light');s.frame(4000);assert.equal(Number(s.canvas.dataset.illumination),0,'Sky light must fade away');
const before=s.strokes.length;s.button.onclick();assert.equal(s.button['aria-pressed'],'false');assert(s.canvas.hidden);s.frame(2200);assert.equal(s.strokes.length,before,'Off state must stop drawing');
s.button.onclick();assert.equal(s.button['aria-pressed'],'true');s.frame(5000);assert(s.strokes.length>before);
const reduced=setup(true);assert.equal(reduced.button['aria-pressed'],'false');assert(reduced.canvas.hidden);
console.log('PASS: white lightning core, red glow, rain drawing, animation progression, pause/resume, reduced-motion default.');
