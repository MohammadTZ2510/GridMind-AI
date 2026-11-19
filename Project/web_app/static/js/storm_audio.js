/* Locally synthesized stereo weather; no audio starts until the user enables it. */
(() => {
  'use strict';
  const button=document.getElementById('storm-sound'),slider=document.getElementById('storm-volume'),label=document.getElementById('storm-volume-label');
  if(!button||!slider)return;
  const Context=window.AudioContext||window.webkitAudioContext;
  let context=null,master=null,rain=null,enabled=false,stormActive=false,volume=.25,closed=false;
  const pending=new Set(),sources=new Set();
  try{const saved=sessionStorage.getItem('gridmind-storm-volume');if(saved!==null&&Number.isFinite(Number(saved)))volume=Math.max(0,Math.min(.7,Number(saved)));}catch{}
  slider.value=Math.round(volume*100);
  function render(){button.textContent=enabled?'Sound ON':'Sound OFF';button.setAttribute('aria-pressed',String(enabled));button.title=enabled?'Mute rain and thunder':'Enable rain and thunder';if(label)label.textContent=Math.round(volume*100)+'%';}
  function clearThunder(){for(const timer of pending)clearTimeout(timer);pending.clear();for(const source of sources){try{source.stop();}catch{}}sources.clear();}
  function update(){if(master)master.gain.setTargetAtTime(enabled&&stormActive&&!document.hidden?volume:0,context.currentTime,.08);}
  function rainBuffer(){
    const seconds = 24;
    const rate = context.sampleRate;
    const frames = Math.ceil(rate * seconds);
    const buffer = context.createBuffer(2, frames, rate);
    const channels = [buffer.getChannelData(0), buffer.getChannelData(1)];

    // Diffuse rain heard from indoors, with occasional drops on the glass.
    const smoothing = 1 - Math.exp(-2 * Math.PI * 1400 / rate);
    for (const data of channels) {
      let wash = 0;
      for (let i = 0; i < frames; i++) {
        wash += smoothing * (Math.random() * 2 - 1 - wash);
        const drift = .86 + .14 * Math.cos(4 * Math.PI * i / frames);
        data[i] = wash * .11 * drift;
      }
      // Smooth the loop seam before adding drops that can wrap across it.
      const seam = Math.round(rate * .025);
      const difference = data[0] - data[frames - 1];
      for (let i = 0; i < seam; i++) {
        data[frames - seam + i] += difference * i / (seam - 1);
      }
    }

    for (let drop = 0; drop < seconds * 24; drop++) {
      const start = Math.floor(Math.random() * frames);
      const duration = .045 + Math.random() * .095;
      const pitch = 460 + Math.random() * 1450;
      const strength = .018 + Math.random() ** 2 * .11;
      const pan = .18 + Math.random() * .64;
      for (let i = 0; i < Math.ceil(rate * duration); i++) {
        const t = i / rate;
        const envelope = Math.min(1, t / .002) * Math.exp(-7 * t / duration);
        const ring = Math.sin(2 * Math.PI * pitch * t * (1 - t * 1.4));
        const splash = (Math.random() * 2 - 1) * Math.exp(-t * 110);
        const sample = strength * envelope * (.38 * ring + .62 * splash);
        const index = (start + i) % frames;
        channels[0][index] += sample * Math.sqrt(1 - pan);
        channels[1][index] += sample * Math.sqrt(pan);
      }
    }
    return buffer;
  }
  function initialise(){
    if(context)return;
    context=new Context();master=context.createGain();master.gain.value=0;
    // Only catches unusual peaks; ordinary storm audio is not heavily compressed.
    const limiter=context.createDynamicsCompressor();limiter.threshold.value=-2;limiter.knee.value=0;limiter.ratio.value=12;limiter.attack.value=.003;limiter.release.value=.12;master.connect(limiter);limiter.connect(context.destination);
    const source=context.createBufferSource();source.buffer=rainBuffer();source.loop=true;
    const high=context.createBiquadFilter();high.type='highpass';high.frequency.value=220;
    const gain=context.createGain();gain.gain.value=.23;source.connect(high);high.connect(gain);gain.connect(master);source.start();rain=source;
  }
  function thunderBuffer(seconds,crack){
    const rate=context.sampleRate,n=Math.ceil(seconds*rate),buffer=context.createBuffer(2,n,rate);
    const echoes=Array.from({length:crack?7:13},(_,k)=>({at:k===0?0:.12+Math.random()*(seconds*.7),amp:(k===0?1:.2+Math.random()*.5),decay:crack?.025+Math.random()*.09:.3+Math.random()*1.05}));
    const pitch=32+Math.random()*22;
    for(let ch=0;ch<2;ch++){
      const data=buffer.getChannelData(ch);let low=0;
      for(let i=0;i<n;i++){
        const t=i/rate,white=Math.random()*2-1;low=.96*low+.04*white;
        let env=0;for(const echo of echoes){const age=t-echo.at-(ch&&echo.at>0?.012:0);if(age>=0)env+=echo.amp*Math.exp(-age/echo.decay)*Math.min(1,age/.0015);}
        const tail=Math.min(1,(seconds-t)/.25);
        const chaotic=1+.25*Math.sin(t*43+Math.sin(t*17)*2);
        data[i]=(crack?white*.32:(low*1.25+white*.025+Math.sin(2*Math.PI*pitch*t)*.018))*env*tail*chaotic;
      }
    }
    let peak=0;for(let ch=0;ch<2;ch++){const data=buffer.getChannelData(ch);for(const sample of data)peak=Math.max(peak,Math.abs(sample));}
    if(peak>.8)for(let ch=0;ch<2;ch++){const data=buffer.getChannelData(ch);for(let i=0;i<data.length;i++)data[i]*=.8/peak;}
    return buffer;
  }
  function layer(buffer,now){
    const source=context.createBufferSource();source.buffer=buffer;source.connect(master);sources.add(source);
    source.onended=()=>{sources.delete(source);source.disconnect();};source.start(now);source.stop(now+buffer.duration);
  }
  function thunder(){
    if(!context||!enabled||!stormActive||document.hidden||closed)return;
    const now=context.currentTime;
    layer(thunderBuffer(.95,true),now); // Wideband, fast crack with irregular fractures.
    layer(thunderBuffer(6.5+Math.random()*1.5,false),now+.025); // Rolling, uneven stereo tail.
  }
  window.GridmindStormAudio={
    setStormActive(active){stormActive=!!active;if(!stormActive)clearThunder();update();},
    strike(){if(!enabled||!stormActive||document.hidden||closed)return;const timer=setTimeout(()=>{pending.delete(timer);thunder();},250+Math.random()*600);pending.add(timer);},
  };
  if(!Context){button.textContent='Sound unavailable';button.disabled=true;slider.disabled=true;return;}
  button.onclick=async()=>{if(button.disabled)return;button.disabled=true;try{if(!enabled){initialise();await context.resume();enabled=true;}else{enabled=false;clearThunder();}update();render();}catch{enabled=false;update();button.textContent='Sound unavailable';button.title='This browser could not start audio.';}finally{button.disabled=false;}};
  slider.oninput=()=>{volume=Math.max(0,Math.min(.7,Number(slider.value)/100));try{sessionStorage.setItem('gridmind-storm-volume',String(volume));}catch{}update();render();};
  document.addEventListener('visibilitychange',()=>{if(document.hidden)clearThunder();update();});
  addEventListener('pagehide',()=>{closed=true;clearThunder();if(rain)try{rain.stop();}catch{}if(context)context.close().catch(()=>{});});
  render();
})();
