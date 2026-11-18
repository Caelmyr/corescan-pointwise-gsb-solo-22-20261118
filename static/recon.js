const $=id=>document.getElementById(id);let project=null,session=null;
const node=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n};
function notify(text,error=false){$('notice').textContent=text;$('notice').className=error?'error':''}
async function api(path,options={}){const response=await fetch(path,{...options,headers:options.body instanceof FormData?{}:{'Content-Type':'application/json',...(options.headers||{})}});const value=await response.json();if(!response.ok)throw Error(value.error||response.statusText);return value}
async function action(fn){const buttons=[...document.querySelectorAll('button')];buttons.forEach(b=>b.disabled=true);try{await fn()}catch(e){notify(e.message,true)}finally{buttons.forEach(b=>b.disabled=false)}}
const spath=s=>'/core/projects/'+project.id+'/recon/sessions/'+session.id+s;
async function post(s,data){const out=await api(spath(s),{method:'POST',body:JSON.stringify({revision:project.revision,...data})});project=out.project;const sid=session.id;session=project.recon_sessions.find(x=>x.id===sid)||null;render();return out.result}
async function listProjects(){const docs=await api('/core/projects');$('projects').replaceChildren();for(const d of docs){const o=node('option',d.name+' · '+(d.recon_sessions?.length||0)+'会话');o.value=d.id;$('projects').append(o)}if(project)$('projects').value=project.id;return docs}
async function loadProject(id){project=await api('/core/projects/'+id);project.recon_sessions=project.recon_sessions||[];session=project.recon_sessions.at(-1)||null;render()}
function img(group,name,params=''){return spath('/image/'+group+'/'+encodeURIComponent(name)+(params?'?'+params:''))}
function render(){if(!project)return;$('subtitle').textContent=project.name+' · '+(project.recon_sessions?.length||0)+' 个重建会话 · 版本 '+project.revision;
$('sessions').replaceChildren();for(const s of (project.recon_sessions||[])){const b=node('button',s.name+' · '+{'curating':'核对中','reconstructed':'待裁决','confirmed':'已确认'}[s.state]);b.className='run'+(session?.id===s.id?' active':'');b.onclick=()=>{session=s;render()};$('sessions').append(b)}
if(!session){$('sessionPanel').replaceChildren(node('div','请先创建或选择重建会话','panel'));return}renderSession()}
function renderSession(){const panel=$('sessionPanel');panel.replaceChildren();
const head=node('div',undefined,'panel');head.append(node('h2',session.name),node('span',{'curating':'核对中','reconstructed':'双方案已重建，待裁决','confirmed':'已确认（输入与设置冻结）'}[session.state],'badge '+session.state));
if(session.confirmation){const f=node('div',undefined,'frozen');f.append(node('p','确认方案：'+session.confirmation.scheme+' · 确认体ID：'+session.confirmation.volume_id));f.append(node('p',session.confirmation.reason||'未填写裁决理由','small'));head.append(f)}
if(session.angle_warnings?.length){const w=node('div',undefined,'warn');w.append(node('p','角度表中未匹配到投影的行：'+session.angle_warnings.join('、')));head.append(w)}
head.append(node('p','视图 '+session.view_names.length+' 张 · 剔除 '+session.excluded.filter(Boolean).length+' 张 · 独立留出 '+session.holdout.length+' 张','small'));
const settings=node('details');settings.append(node('summary','几何与校正设置（确认前可改，改后自动重新做暗空场校正）'));const ta=node('textarea');ta.value=JSON.stringify(session.settings,null,2);settings.append(ta);
const saveSettings=node('button','应用设置并重新校正');saveSettings.onclick=()=>action(async()=>{await post('/curation',{settings:JSON.parse(ta.value)});notify('设置已更新，暗空场校正已重算')});settings.append(saveSettings);head.append(settings);panel.append(head);
panel.append(curationPanel());
if(session.state!=='curating'){panel.append(evidencePanel());panel.append(supportPanel());panel.append(roundtripPanel())}
if(session.state==='reconstructed')panel.append(confirmPanel())}
function curationPanel(){const wrap=node('div',undefined,'panel');wrap.append(node('h2','逐张核对投影与角度配对'));
wrap.append(node('p','修正角度(度)留空表示暂不配对；剔除的整图不参与重建；留出投影不参与任一方案，仅作独立检查。校正后有效射线率为0的视图会被标出，不能把无效计数当零吸收。','small'));
const table=node('table'),thead=node('thead'),tr=node('tr');['#','投影文件','角度(度)','原始图','校正后线积分/有效射线','有效率','剔除','留出'].forEach(h=>tr.append(node('th',h)));thead.append(tr);table.append(thead);const tbody=node('tbody');
const disabled=session.state!=='curating';
session.view_names.forEach((name,v)=>{const row=node('tr');row.append(node('td',String(v)));row.append(node('td',name));
const angleCell=node('td');const inp=node('input');inp.type='number';inp.step='any';inp.value=session.angles[v]??'';inp.disabled=disabled;angleCell.append(inp);row.append(angleCell);
const rawImg=node('img',undefined,'thumb');rawImg.src=img('inputs','raw','view='+v);const cell1=node('td');cell1.append(rawImg);row.append(cell1);
const lineImg=node('img',undefined,'thumb');lineImg.src=img('corrected','line','view='+v);const cell2=node('td');cell2.append(lineImg);row.append(cell2);
const stats=session.correction?.views?.[v];row.append(node('td',stats?(stats.fraction_valid*100).toFixed(1)+'%':'—'));
const ec=node('td');const ex=node('input');ex.type='checkbox';ex.checked=session.excluded[v];ex.disabled=disabled;ec.append(ex);row.append(ec);
const hc=node('td');const hb=node('input');hb.type='checkbox';hb.checked=session.holdout.includes(v);hb.disabled=disabled;hc.append(hb);row.append(hc);
if((stats&&stats.fraction_valid===0)||session.excluded[v])row.style.opacity=.55;
tbody.append(row)});table.append(tbody);wrap.append(table);
if(session.state==='curating'){const btn=node('button',undefined,'primary');btn.textContent='保存核对结果（角度/剔除/留出）';btn.onclick=()=>action(async()=>{const rows=[...tbody.querySelectorAll('tr')];const angles=rows.map(r=>{const val=r.querySelector('td:nth-child(3) input').value;return val===''?null:Number(val)});const excluded=rows.map(r=>r.querySelector('td:nth-child(7) input').checked);const holdout=rows.map((r,i)=>r.querySelector('td:nth-child(8) input').checked?i:null).filter(i=>i!==null);
await post('/curation',{angles,excluded,holdout});notify('核对结果已保存；校正有效射线掩码已同步')});
const run=node('button','运行双方案重建（直接恢复 + 非负平滑逐步拟合）');run.style.marginLeft='10px';run.onclick=()=>action(async()=>{notify('正在重建，CPU实现请稍候…');const r=await post('/reconstruct',{});notify('双方案重建完成，请比较留出投影证据后再裁决');});wrap.append(btn,run)}
return wrap}
function evidencePanel(){const wrap=node('div',undefined,'panel');wrap.append(node('h2','两方案证据比较（独立留出投影不参与重建）'));
const grid=node('div',undefined,'evidence-grid');
for(const [name,label] of [['direct','直接恢复 FBP'],['constrained','非负+平滑逐步拟合']]){const fig=node('figure');fig.append(node('figcaption',label));
const m=session.evidence.metrics[name];const pre=node('pre');pre.textContent=JSON.stringify({'训练视图(参与重建)':m.train,'独立留出视图(未参与)':m.holdout,'全部有效射线':m.all_valid},null,2);fig.append(pre);
const vol=node('img',undefined,'thumb');vol.style.height='230px';vol.src=img('schemes',name+'_volume','axis=z');fig.append(vol);grid.append(fig)}
wrap.append(grid);
const warn=node('div',undefined,'warn');warn.append(node('p',session.evidence.interpretation));if(session.evidence.duplicate_holdout_views?.length)warn.append(node('p','警告：留出视图 #'+session.evidence.duplicate_holdout_views.join('、#')+' 在训练集中存在同角度重复视图，其独立性减弱。'));
wrap.append(warn);
const diff=node('figure');diff.append(node('figcaption','两方案差异（约束−直接）'));const di=node('img',undefined,'thumb');di.src=img('comparison','difference','axis=z');diff.append(di);wrap.append(diff);
const table=node('table'),thead=node('thead'),hr=node('tr');['视图','角色','直接RMSE','约束RMSE','直接MAE','约束MAE'].forEach(h=>hr.append(node('th',h)));thead.append(hr);table.append(thead);const tb=node('tbody');
const dv=session.evidence.metrics.direct.per_view,cv=session.evidence.metrics.constrained.per_view;
dv.forEach((d,i)=>{const c=cv[i];const r=node('tr');[d.view,d.role,d.rmse,c.rmse,d.mae,c.mae].forEach(v=>r.append(node('td',v===null?'—':(typeof v==='number'?v.toFixed(5):String(v)))));if(d.role==='holdout')r.style.background='#243b2e';tb.append(r)});table.append(tb);
const wrap2=node('div',undefined,'table-wrap');wrap2.append(table);wrap.append(wrap2);return wrap}
function supportPanel(){const s=session.evidence.support_summary;const wrap=node('div',undefined,'panel');wrap.append(node('h2','可靠支持范围与未决区域'));
wrap.append(node('p',`最大角间隔 ${s.largest_gap_deg.toFixed(1)}° · 角覆盖 ${s.angular_coverage_deg.toFixed(1)}° · 截断/视场半径 ${s.fov_radius_mm.toFixed(2)} mm · 可靠体素 ${s.reliable_voxels} · 未决 ${s.uncertain_voxels} · 不可恢复 ${s.unsupported_voxels}`,'small'));
wrap.append(node('p','支持图配色：绿=可靠（有效射线充分且角间隔达标），琥珀=未决（保留但未证实），黑=不可恢复；导出确认体仅把可靠区标为有效。','small'));
const fig=node('figure');const im=node('img',undefined,'thumb');im.style.width='320px';im.src=img('support','status','axis=z');fig.append(im,node('figcaption','支持状态（中部Z切面）'));wrap.append(fig);return wrap}
function roundtripPanel(){const wrap=node('div',undefined,'panel');wrap.append(node('h2','恢复体 → 原始角度往返重投影与残差'));
const sel=node('select');[['direct','直接恢复'],['constrained','约束拟合']].forEach(([v,t])=>{const o=node('option',t);o.value=v;if(session.confirmation?.scheme===v)o.selected=true;sel.append(o)});
const btn=node('button','按原顺序/方向重投影并计算残差');btn.onclick=()=>action(async()=>{const r=await post('/reproject',{scheme:sel.value});notify('往返完成：有效射线RMSE '+r.trip.metrics.rmse?.toFixed(5))});wrap.append(sel,btn);
if(session.arrays.roundtrip?.forward){const grid=node('div',undefined,'evidence-grid');
for(const [n,t] of [['forward','重投影（原始角度顺序）'],['residual','残差（仅有效射线；无效/缺角不计零）']]){const fig=node('figure');fig.append(node('figcaption',t));const im=node('img',undefined,'thumb');im.src=img('roundtrip',n,'view='+Math.floor(session.holdout[0]??0));fig.append(im);grid.append(fig)}wrap.append(grid);
const last=session.roundtrips.at(-1);if(last){const pre=node('pre');pre.textContent=JSON.stringify(last.metrics,null,2);wrap.append(pre)}}
return wrap}
function confirmPanel(){const wrap=node('div',undefined,'panel');wrap.append(node('h2','裁决并生成确认体'));
const sel=node('select');[['direct','采用直接恢复'],['constrained','采用非负平滑逐步拟合']].forEach(([v,t])=>{sel.append(new Option(t,v))});
const reason=node('textarea');reason.placeholder='记录裁决理由：留出投影一致性、小孔/薄壁条纹、边缘损失、支持范围的核对结论';
const btn=node('button','确认并生成三维体（带毫米间距与原点）','primary');btn.onclick=()=>action(async()=>{const r=await post('/confirm',{scheme:sel.value,reason:reason.value});notify('确认体已生成（ID '+r.volume_id+'），到三维体检查页即可查看、分区和测量；会话输入已冻结')});
wrap.append(sel,reason,btn);wrap.append(node('p','确认后修改角度或几何不会改动该体；请用“克隆会话”另建试验。确认体可在 / 三维检查页继续正交查看、分区与测量，直接导入NPY/NPZ/ZIP的原流程不受影响。','small'));return wrap}
$('newProject').onclick=()=>action(async()=>{project=await api('/core/projects',{method:'POST',body:JSON.stringify({name:'投影重建 '+new Date().toLocaleString()})});project.recon_sessions=[];session=null;await listProjects();render();notify('新项目已创建')});
$('projects').onchange=()=>action(()=>loadProject($('projects').value));$('reload').onclick=()=>action(async()=>{await listProjects();if(project)await loadProject(project.id)});
$('createSession').onclick=()=>action(async()=>{if(!project)throw Error('请先选择或新建检查项目');const form=new FormData();if(!$('bundle').files[0]||!$('angles').files[0])throw Error('请同时选择投影ZIP和角度表CSV');form.append('bundle',$('bundle').files[0]);form.append('angles',$('angles').files[0]);form.append('revision',project.revision);form.append('settings',$('newSettings').value);form.append('name','投影重建会话 '+new Date().toLocaleTimeString());
const out=await api('/core/projects/'+project.id+'/recon/sessions',{method:'POST',body:form});project=out.project;session=project.recon_sessions.at(-1);render();notify('会话已创建：暗场空场校正完成，请逐张核对配对与有效射线')});
$('downloadSession').onclick=()=>action(()=>{if(session)window.location.href=spath('/download')});
$('cloneSession').onclick=()=>action(async()=>{if(!session)throw Error('请先选择会话');const r=await post('/clone',{});session=project.recon_sessions.find(x=>x.id===r.id);render();notify('已克隆为可编辑的新会话，原确认体保持不变')});
(async()=>{try{const docs=await listProjects();if(docs.length)await loadProject(docs[0])}catch(e){notify(e.message,true)}})();
