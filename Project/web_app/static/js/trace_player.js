/* Replay genuine scheduler observations. No inferred search branches or fake moves. */
(function () {
'use strict';
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const hour = h => String(h).padStart(2,'0')+':00';
const score = v => Number.isFinite(v) ? v.toFixed(5) : '—';
function describe(e) {
  switch(e.event) {
    case 'strategy_start': return `Start CSP strategy ${e.attempt}. Initialize protected and no-shortage slots; grey cells still need a decision.`;
    case 'assign': return `Try ${e.feeder_id} ${e.decision} at ${hour(e.hour)}. This is a tentative search assignment.`;
    case 'backtrack': return `Undo the assignment for ${e.feeder_id} at ${hour(e.hour)} and return that slot to unassigned.`;
    case 'search_limit': return `Search budget reached: ${e.reason}. A further strategy may follow.`;
    case 'csp_start': return `CSP found a valid starting schedule for Hill Climbing (${e.nodes} search nodes).`;
    case 'restart': return `Prepare Hill Climbing start ${Number(e.restart)+1}.`;
    case 'restart_state': return `Load the real schedule for start ${Number(e.restart)+1}; score ${score(e.score)}. ${e.restart===0?'This is the CSP starting schedule.':'This start includes valid seeded perturbations.'}`;
    case 'accepted_move': {
      const on=(e.turn_on||[]).map(c=>`${c.feeder_id} at ${hour(c.hour)}`).join(', ');
      const off=(e.turn_off||[]).map(c=>`${c.feeder_id} at ${hour(c.hour)}`).join(', ');
      return `Accept improvement: ${on?'turn ON '+on:''}${on&&off?'; ':''}${off?'turn OFF '+off:''}. Score ${score(e.previous_score)} → ${score(e.score)}.`;
    }
    case 'rejected_invalid': return 'Reject an improving candidate because it violates a hard constraint. The current schedule stays unchanged.';
    case 'local_optimum': return `No improving valid neighbour found for this start. Score ${score(e.score)} after ${e.iterations} accepted improvements.`;
    case 'infeasible': return `No valid schedule: ${e.reason}`;
    case 'skipped': return `Optimization skipped: ${e.reason}.`;
    case 'complete': return e.best_restart!==undefined ? `Return the best schedule from start ${Number(e.best_restart)+1}, score ${score(e.score)}.` : `CSP completed: ${e.status}. ${e.nodes} nodes explored; ${e.backtracks} backtracks.`;
    default: return String(e.event||'Recorded event').replaceAll('_',' ');
  }
}
class TracePlayer {
  constructor(container,result) {
    this.el=container;this.result=result;this.events=result.trace||[];this.ids=result.feeder_ids||[];
    this.cursor=0;this.timer=null;this.speed=550;this.disposed=false;
    this.el.innerHTML=`<div class="trace-result"><span class="tag ${result.status==='FEASIBLE'?'green':''}">${esc(result.status)}</span><strong>${esc(result.scenario_id)} · ${result.method==='HC'?'Hill Climbing':'CSP Backtracking'}</strong><span>${(result.runtime_s*1000).toFixed(2)} ms execution · ${result.event_count} recorded events</span></div>
      <p class="note">The algorithm has finished. These controls replay its recorded execution; playback speed does not change algorithm runtime. The board shows the search state at the selected event, not a saved dashboard schedule.</p>
      ${result.trace_truncated?'<div class="notice warning">Recording limit reached. Only the first '+this.events.length+' events are replayable; the omitted search is not reconstructed.</div>':''}
      <div class="trace-controls"><button data-trace="reset" aria-label="Reset trace playback">↺ Reset</button><button data-trace="back" aria-label="Previous trace event">← Back</button><button data-trace="play" class="primary-button" aria-label="Play trace">▶ Play</button><button data-trace="next" aria-label="Next trace event">Step →</button><button data-trace="end" aria-label="Last recorded event">End</button><label>Speed <select data-trace="speed" aria-label="Trace playback speed"><option value="1100">Slow</option><option value="550" selected>Normal</option><option value="150">Fast</option></select></label></div>
      <div class="trace-scrub"><input data-trace="seek" type="range" min="0" max="${this.events.length}" value="0" aria-label="Trace event position"><span data-trace="position"></span></div>
      <div class="trace-current" role="status" aria-live="polite"><div class="eyebrow" data-trace="event-type"></div><p data-trace="description"></p><small data-trace="event-info"></small></div>
      <div class="trace-legend"><span><i class="trace-key on"></i>ON</span><span><i class="trace-key off"></i>OFF</span><span><i class="trace-key unknown"></i>Unassigned / unavailable</span><span><i class="trace-key changed"></i>Current event</span></div>
      <div class="table-wrap trace-board-wrap"><table class="trace-board" aria-label="Recorded feeder-hour search state"><thead><tr><th>FEEDER</th>${Array.from({length:24},(_,h)=>'<th>'+String(h).padStart(2,'0')+'</th>').join('')}</tr></thead><tbody>${this.ids.map(id=>'<tr><th>'+esc(id)+(id==='F001'?' ✦':'')+'</th>'+Array.from({length:24},(_,h)=>'<td data-cell="'+id+'-'+h+'" class="trace-cell unknown" title="'+id+' '+hour(h)+' unassigned">·</td>').join('')+'</tr>').join('')}</tbody></table></div>
      ${result.method==='HC'?'<div class="trace-chart" data-trace="chart"></div><p class="note">Each line is one seeded start. Lower scores are better. A new start can begin at a higher score; separate lines prevent it being mistaken for an accepted worsening move.</p>':'<p class="note">CSP may find a solution without backtracking. Only real undo events are shown. Initialization records forced ON slots and no-shortage slots; assignments made during search remain tentative until completion.</p>'}
      <div class="section-heading"><h3>Recorded steps</h3><span class="small muted">Select a step to inspect its state</span></div><ol class="trace-event-list" data-trace="events"></ol>
      <details class="trace-raw"><summary>Technical details · raw JSON</summary><pre class="trace-output"></pre></details>`;
    this.find('reset').onclick=()=>this.seek(0);
    this.find('back').onclick=()=>this.seek(this.cursor-1);
    this.find('next').onclick=()=>this.seek(this.cursor+1);
    this.find('end').onclick=()=>this.seek(this.events.length);
    this.find('play').onclick=()=>this.toggle();
    this.find('speed').onchange=e=>{this.speed=Number(e.target.value);if(this.timer){this.pause();this.toggle();}};
    this.find('seek').oninput=e=>this.seek(Number(e.target.value));
    this.el.querySelector('pre').textContent=JSON.stringify(result,null,2);
    this.draw();
  }
  find(name){return this.el.querySelector('[data-trace="'+name+'"]');}
  pause(){clearInterval(this.timer);this.timer=null;const b=this.find('play');if(b){b.textContent='▶ Play';b.setAttribute('aria-label','Play trace');}}
  toggle(){if(this.timer){this.pause();return;}if(this.cursor>=this.events.length)this.cursor=0;this.find('play').textContent='Ⅱ Pause';this.find('play').setAttribute('aria-label','Pause trace');this.timer=setInterval(()=>{if(!this.el.isConnected||this.disposed){this.pause();return;}if(this.cursor<this.events.length){this.cursor++;this.draw();}if(this.cursor>=this.events.length)this.pause();},this.speed);}
  seek(index){this.pause();this.cursor=Math.max(0,Math.min(this.events.length,index));this.draw();}
  reconstruct(){
    let board=this.ids.map(()=>Array(24).fill(-1));const curves=new Map();let restart=0;
    for(let k=0;k<this.cursor;k++){
      const e=this.events[k];
      if(e.schedule)board=e.schedule.map(r=>r.slice());
      if(e.event==='assign'){const i=this.ids.indexOf(e.feeder_id);if(i>=0)board[i][e.hour]=e.decision==='ON'?1:0;}
      if(e.event==='backtrack'){const i=this.ids.indexOf(e.feeder_id);if(i>=0)board[i][e.hour]=-1;}
      if(e.event==='accepted_move')for(const [key,value] of [['turn_on',1],['turn_off',0]])for(const c of e[key]||[]){const i=this.ids.indexOf(c.feeder_id);if(i>=0)board[i][c.hour]=value;}
      if(e.event==='restart_state'){restart=e.restart;curves.set(restart,{name:'Start '+(restart+1),x:[k+1],y:[e.score]});}
      if(e.event==='accepted_move'&&curves.has(restart)){curves.get(restart).x.push(k+1);curves.get(restart).y.push(e.score);}
    }
    return {board,curves};
  }
  draw(){
    if(this.disposed)return;const {board,curves}=this.reconstruct(),e=this.events[this.cursor-1];
    this.find('seek').value=this.cursor;this.find('position').textContent=`Event ${this.cursor} / ${this.events.length}`;
    this.find('back').disabled=this.cursor===0;this.find('next').disabled=this.cursor===this.events.length;
    this.find('event-type').textContent=e?e.event.replaceAll('_',' ').toUpperCase():'READY TO REPLAY';
    this.find('description').textContent=e?describe(e):'Press Play or Step to watch the actual algorithm decisions appear on the feeder-hour board.';
    this.find('event-info').textContent=e?`Recorded at ${Number(e.elapsed_ms).toFixed(3)} ms${e.node!==undefined?' · search node '+e.node:''}`:'Grey cells have no replayed assignment yet. No state has been invented.';
    const changed=new Set();if(e?.feeder_id)changed.add(e.feeder_id+'-'+e.hour);for(const c of [...(e?.turn_on||[]),...(e?.turn_off||[])])changed.add(c.feeder_id+'-'+c.hour);
    this.ids.forEach((id,i)=>{for(let h=0;h<24;h++){const cell=this.el.querySelector('[data-cell="'+id+'-'+h+'"]'),v=board[i]?.[h],name=v===1?'on':v===0?'off':'unknown';cell.className='trace-cell '+name+(changed.has(id+'-'+h)?' current':'');cell.textContent=v===1?'●':v===0?'×':'·';cell.title=`${id} ${hour(h)}: ${name==='unknown'?'Unassigned / unavailable':name.toUpperCase()}`;}});
    const start=Math.max(0,this.cursor-6),end=Math.min(this.events.length,Math.max(this.cursor+3,8));
    this.find('events').innerHTML=this.events.slice(start,end).map((event,i)=>`<li><button data-event-index="${start+i+1}" class="${start+i+1===this.cursor?'active':''}"><span>${start+i+1}</span>${esc(describe(event))}</button></li>`).join('');
    this.find('events').querySelectorAll('button').forEach(b=>b.onclick=()=>this.seek(Number(b.dataset.eventIndex)));
    const chart=this.find('chart');if(chart&&window.Plotly){Plotly.react(chart,[...curves.values()].map(c=>({...c,type:'scatter',mode:'lines+markers',line:{width:2},marker:{size:5}})),{paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:'#c6c1c3',size:11},margin:{l:60,r:20,t:30,b:45},title:{text:'Objective score during recorded improvements',font:{size:13}},xaxis:{title:'Recorded event',gridcolor:'#312b2d'},yaxis:{title:'Score · lower is better',gridcolor:'#312b2d'},legend:{orientation:'h'},colorway:['#ff2448','#bca6cc','#eabb6f','#64d9ad','#ef7b88']},{responsive:true,displayModeBar:false});}
  }
  dispose(){this.pause();this.disposed=true;const c=this.find('chart');if(c&&window.Plotly)Plotly.purge(c);}
}
window.TracePlayer=TracePlayer;
})();
