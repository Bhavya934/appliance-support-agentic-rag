const chat = document.getElementById('chat');
const form = document.getElementById('chatForm');
const question = document.getElementById('question');
const trace = document.getElementById('trace');
const sourceUsed = document.getElementById('sourceUsed');

function escapeHtml(s=''){return s.replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
function formatText(s=''){return escapeHtml(s).replace(/\n/g,'<br>');}
const SOURCE_LABELS={manuals:'Bosch & Siemens manuals',web_search:'Web search',direct:'Direct reply',insufficient_evidence:'No reliable evidence'};
const sourceLabel=s=>SOURCE_LABELS[s]||s;
// Answers are markdown (steps, tables, bold); render them, sanitised. Falls back to plain text if the CDN is unreachable.
function formatAnswer(s=''){
  if(window.marked&&window.DOMPurify){return `<div class="md">${DOMPurify.sanitize(marked.parse(s))}</div>`;}
  return formatText(s);
}
function addMessage(role, text, source='', citations=[]){
  const wrap=document.createElement('div'); wrap.className=`message ${role}`;
  const citeHtml=citations.length?`<div class="citations"><strong>Sources</strong><br>${citations.map(c=>c.url?`<a href="${escapeHtml(c.url)}" target="_blank" rel="noopener">${escapeHtml(c.title)}</a>`:escapeHtml(c.title)).join('<br>')}</div>`:'';
  const body=role==='assistant'?formatAnswer(text):formatText(text);
  wrap.innerHTML=`<div class="avatar">AI</div><div class="bubble">${body}${source?`<div class="answer-source">Source: ${escapeHtml(sourceLabel(source))}</div>`:''}${citeHtml}</div>`;
  chat.appendChild(wrap); chat.scrollTop=chat.scrollHeight;
}
function renderTrace(items=[]){trace.innerHTML=items.length?items.map(x=>`<div class="trace-item">${escapeHtml(x)}</div>`).join(''):'<div class="empty">No trace.</div>';}
async function askAgent(q){
  addMessage('user',q); question.value=''; renderTrace(['Running LangGraph workflow...']); sourceUsed.textContent='Running';
  const btn=form.querySelector('button'); btn.disabled=true;
  try{
    const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
    const data=await res.json(); if(!res.ok) throw new Error(data.detail||'Request failed');
    addMessage('assistant',data.answer,data.source_used,data.citations||[]); renderTrace(data.trace||[]); sourceUsed.textContent=sourceLabel(data.source_used);
  }catch(e){addMessage('assistant',`Error: ${e.message}`); renderTrace(['Request failed']); sourceUsed.textContent='Error';}
  finally{btn.disabled=false;}
}
form.addEventListener('submit',e=>{e.preventDefault(); const q=question.value.trim(); if(q) askAgent(q);});
// Enter sends, Shift+Enter adds a new line
question.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault(); form.requestSubmit();}});
document.querySelectorAll('.example').forEach(b=>b.addEventListener('click',()=>askAgent(b.textContent.trim())));

const modal=document.getElementById('uploadModal');
const openUpload=document.getElementById('openUpload'); // absent in the public demo
if(openUpload) openUpload.onclick=()=>modal.classList.remove('hidden');
document.getElementById('closeUpload').onclick=()=>modal.classList.add('hidden');
document.getElementById('uploadBtn').onclick=async()=>{
  const file=document.getElementById('fileInput').files[0]; const key=document.getElementById('adminKey').value; const status=document.getElementById('uploadStatus');
  if(!file){status.textContent='Choose a file first.'; return;}
  status.textContent='Indexing document...'; const fd=new FormData(); fd.append('file',file);
  try{const r=await fetch('/api/ingest',{method:'POST',headers:{'X-Admin-Key':key},body:fd}); const d=await r.json(); if(!r.ok) throw new Error(d.detail||'Upload failed'); status.textContent=`Indexed ${d.file}: ${d.chunks} chunks.`;}catch(e){status.textContent=`Error: ${e.message}`;}
};
