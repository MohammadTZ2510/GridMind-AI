// Offline control/audio-graph tests: node tests/test_storm_audio.js
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
const source=fs.readFileSync(path.join(__dirname,'../web_app/static/js/storm_audio.js'),'utf8');
const nodes=[],contexts=[],timers=new Map(),events={};let timerID=0;
class Param{constructor(){this.value=0;}setTargetAtTime(v){this.value=v;}setValueAtTime(v){this.value=v;}linearRampToValueAtTime(v){this.value=v;}exponentialRampToValueAtTime(v){this.value=v;}}
class Node{constructor(type){this.type=type;this.gain=new Param();this.frequency=new Param();this.Q=new Param();this.threshold=new Param();this.knee=new Param();this.ratio=new Param();this.attack=new Param();this.release=new Param();nodes.push(this);}connect(n){this.to=n;}disconnect(){}start(t){this.started=true;this.startTime=t;}stop(t){this.stopped=true;this.stopTime=t;}}
class Audio{constructor(){this.sampleRate=48000;this.currentTime=1;this.destination={};contexts.push(this);}createGain(){return new Node('gain');}createDynamicsCompressor(){return new Node('compressor');}createBiquadFilter(){return new Node('filter');}createBufferSource(){return new Node('source');}createBuffer(ch,n){const data=Array.from({length:ch},()=>new Float32Array(n));return {numberOfChannels:ch,duration:n/this.sampleRate,getChannelData:c=>data[c]};}async resume(){this.resumed=true;}async close(){this.closed=true;}}
const button={setAttribute(k,v){this[k]=v;}},slider={value:'25'},label={};
const document={hidden:false,getElementById:id=>id==='storm-sound'?button:id==='storm-volume'?slider:label,addEventListener(k,f){events[k]=f;}};
const sandbox={window:{AudioContext:Audio},document,sessionStorage:{getItem:()=>null,setItem(){}},Math,setTimeout(f,delay){const id=++timerID;timers.set(id,{f,delay});return id;},clearTimeout(id){timers.delete(id);},addEventListener(k,f){events[k]=f;}};
vm.createContext(sandbox);vm.runInContext(source,sandbox);
(async()=>{
 const api=sandbox.window.GridmindStormAudio;
 assert.equal(contexts.length,0,'No audio context before a user click');assert.equal(button['aria-pressed'],'false');
 api.setStormActive(true);api.strike();assert.equal(timers.size,0,'Muted lightning must schedule no sound');
 await button.onclick();assert.equal(contexts.length,1);assert(contexts[0].resumed);assert.equal(button['aria-pressed'],'true');
 const rain=nodes.find(n=>n.type==='source');assert(rain.loop&&rain.started,'Rain must loop after sound is enabled');
 api.strike();assert.equal(timers.size,1);const entry=[...timers.entries()][0],timer=entry[1];assert(timer.delay>=250&&timer.delay<=850,'Thunder must follow its strike with a short delay');timers.delete(entry[0]);timer.f();
 assert.equal(nodes.filter(n=>n.type==='source'&&n.started).length,3,'Rain plus thunder rumble and crack');
 for(const s of nodes.filter(n=>n.type==='source')){
 assert.equal(s.buffer.numberOfChannels,2,'Stereo weather');let peak=0,sum=0;const a=s.buffer.getChannelData(0),b=s.buffer.getChannelData(1);let different=0;
 for(let i=0;i<a.length;i++){assert(Number.isFinite(a[i]));peak=Math.max(peak,Math.abs(a[i]));sum+=a[i]*a[i];if(a[i]!==b[i])different++;}
 assert(peak<1&&sum>0,'Non-silent waveform without clipped samples');assert(different>a.length*.5,'Independent stereo texture');
 assert(Number.isFinite(s.stopTime)||s.loop,'One-shot sources must stop');
}
assert.equal(nodes.find(n=>n.type==='compressor').threshold.value,-2,'Peak limiter should leave ordinary dynamics intact');
const loudestWindow = buffer => {
   const samples = buffer.getChannelData(0);
   const window = Math.round(contexts[0].sampleRate * .025);
   let highest = 0;
   for (let start = 0; start + window <= samples.length; start += window) {
     let power = 0;
     for (let i = start; i < start + window; i++) power += samples[i] ** 2;
     highest = Math.max(highest, Math.sqrt(power / window));
   }
   return highest;
 };
 const crack=nodes.filter(n=>n.type==='source')[1];
 assert(loudestWindow(crack.buffer)>3*loudestWindow(rain.buffer)*rain.to.to.gain.value,
   'The thunder crack must stand out above even the loudest rain window');
 const master=nodes.find(n=>n.type==='gain');slider.value='99';slider.oninput();assert.equal(master.gain.value,.7,'Volume is capped');
 api.strike();api.setStormActive(false);assert.equal(master.gain.value,0);assert.equal(timers.size,0);api.strike();assert.equal(timers.size,0);
 api.setStormActive(true);document.hidden=true;events.visibilitychange();assert.equal(master.gain.value,0);document.hidden=false;events.visibilitychange();assert.equal(master.gain.value,.7);
 await button.onclick();assert.equal(button['aria-pressed'],'false');assert.equal(master.gain.value,0);events.pagehide();assert(contexts[0].closed);
 console.log('PASS: click-to-enable only, looping rain, strike-linked delayed thunder/crack, volume cap, mute, storm-off, hidden-tab pause and cleanup.');
})().catch(e=>{console.error(e);process.exitCode=1;});
