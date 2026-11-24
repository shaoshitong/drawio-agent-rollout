from pathlib import Path
import subprocess,json,hashlib,time
script=r'''
from pathlib import Path
from hashlib import sha256
from collections import defaultdict
from copy import deepcopy
import json,sys,time
B=Path('/DATA/disk1/zbw_tmp4/drawio-agent-sft-20261121')
source=B/'checks/thinking-recovery-20261124/source-5174812'
sys.path.insert(0,str(source))
from sft.data import open_store
from sft.export_sft import normalize
from sft.train_unsloth import render_completion
from transformers import AutoProcessor
manifest=B/'checks/dsh-midflight-20261123/curriculum-final.jsonl'
exposure_path=B/'checks/stopped218-supervised-actions-20261124/decisions.jsonl'
native_path=B/'checks/rewritten-supervision-20261124/train/decisions.jsonl'
def lines(p):return [json.loads(l) for l in p.open() if l.strip()]
manifest_rows={r['sample_id']:r for r in lines(manifest)}
exposures=sorted(lines(exposure_path),key=lambda r:r['exposure'])
assert [r['exposure'] for r in exposures]==list(range(6976))
native={r['sample_id']:r for r in lines(native_path)}
processor=AutoProcessor.from_pretrained(B/'models/Qwen3.5-9B',local_files_only=True)
end_id=processor.tokenizer.convert_tokens_to_ids('</think>')
stores={}
def store(root):
 key=str(root)
 if key not in stores:stores[key]=open_store(root)
 return stores[key]
groups=defaultdict(lambda:{'N':0,'Rw':0,'Rn':0,'samples':0})
first_counts=[];format_delta=0;per_sample=[]
for i,e in enumerate(exposures):
 row=manifest_rows[e['sample_id']]; ref=native[e['sample_id']]
 root=manifest.parent/row['snapshot_root']
 msg=deepcopy(store(root).get(row['response_ref'])['choices'][0]['message'])
 rewrite=row.get('reasoning_rewrite')
 if rewrite:
  sidecar=store(rewrite['root']).get(rewrite['ref'])
  assert sidecar['response_ref']==row['response_ref'] and sidecar['request_ref']==row['request_ref']
  if msg.get('reasoning_content'):msg['reasoning_content']=sidecar['current_reasoning']
 comp,_=normalize([msg],decode_images=False)
 comp[0]['content']=comp[0].get('content') or ''
 example={'prompt':[{'role':'user','content':'Audit prompt.'}],'completion':comp,'chat_template_kwargs':{**row.get('chat_template_kwargs',{}),'enable_thinking':True}}
 _,text=render_completion(processor,example)
 assert sha256(text.encode()).hexdigest()==ref['target_sha256'],e['sample_id']
 ids=processor.tokenizer.encode(text,add_special_tokens=False)
 N=len(ids);assert N==ref['target_tokens']==row['completion_tokens'],e['sample_id']
 has=bool((msg.get('reasoning_content') or '').strip())
 Rw=ids.index(end_id) if has else 0
 Rn=ref['partition_tokens'].get('reasoning',0)
 assert Rw>=Rn
 format_delta+=Rw-Rn
 g=groups[e['exposure']//32];g['N']+=N;g['Rw']+=Rw;g['Rn']+=Rn;g['samples']+=1
 if i<32:first_counts.append(N)
 per_sample.append(Rw/N)
 if (i+1)%1000==0:print(json.dumps({'verified':i+1}),file=sys.stderr,flush=True)
assert len(groups)==218 and all(x['samples']==32 for x in groups.values())
def quant(v):
 v=sorted(v)
 return {'mean':sum(v)/len(v),'min':v[0],'p10':v[round((len(v)-1)*.1)],'p50':v[round((len(v)-1)*.5)],'p90':v[round((len(v)-1)*.9)],'max':v[-1]}
ratios=[g['Rw']/g['N'] for _,g in sorted(groups.items())]
r_mass=sum(.7*x for x in ratios);other_mass=sum(1-x for x in ratios)
runtime=B/'runs/causal-cot-20261122/train-v1/train'
counts={p.name:json.loads(p.read_text()) for p in runtime.glob('loss-count-rank-*.json')}
assert set(d['global_accumulated_target_tokens'] for d in counts.values())=={groups[0]['N']}
config=json.loads((runtime/'runtime-config.json').read_text())
report={'at':time.time(),'source_files':{str(p):sha256(p.read_bytes()).hexdigest() for p in (manifest,exposure_path,native_path)},
 'actual_loss_source_commit':'157a493d5c556c81faca941b5bf610e8ca282ea0',
 'actual_runtime':{'full_finetuning':config['full_finetuning'],'thinking_loss_weight':config['thinking_loss_weight'],'batch_size':config['batch_size'],'gradient_accumulation_steps':config['gradient_accumulation_steps'],'loss_count_audits':counts,'adapter_loss_runtime':json.loads((runtime/'adapter-update.json').read_text())['loss_runtime']},
 'label_shift_scope':'Prompt is nonempty and fully masked; labels[...,1:] excludes a masked prompt token, not any current target token. N is the complete target token count.',
 'verification':{'samples':len(exposures),'updates':len(groups),'samples_per_update':32,'all_target_hashes_match_native_audit':True,'all_target_lengths_match_manifest_and_native':True,'first_update_N':groups[0]['N'],'first_update_rank_first_N':first_counts[:4],'actual_first_update_global_count_matches':True},
 'Rw_definition':'Number of target tokens before first native </think> ID when current reasoning is nonempty, exactly token_weights(.7) range. Ending delimiter weight remains1.',
 'native_reasoning_to_weighted_boundary_extra_format_tokens':format_delta,
 'global_token_reasoning_fraction_native':sum(g['Rn'] for g in groups.values())/sum(g['N'] for g in groups.values()),
 'global_token_Rw_fraction':sum(g['Rw'] for g in groups.values())/sum(g['N'] for g in groups.values()),
 'per_sample_Rw_fraction':quant(per_sample),
 'per_update_Rw_fraction':quant(ratios),
 'per_update_target_tokens':quant([g['N'] for g in groups.values()]),
 'mean_reasoning_coefficient_mass':r_mass/len(groups),'mean_other_coefficient_mass':other_mass/len(groups),
 'reasoning_fraction_of_summed_coefficient_mass':r_mass/(r_mass+other_mass),
 'limitations':['These are objective coefficients and token mass, not CE contributions or gradient shares.','Actual per-token CE and per-partition gradients were not saved; cannot infer loss reduction by action category.','Averages across optimizer steps ignore LR and evolving parameter sensitivity.','Runtime denom evidence explicitly records only first update. The same audited full-phase implementation supplies counts for later updates; audit does not replay model forwards.'],
 'gpu_model_forwards':0,'remote_files_written':False}
print(json.dumps(report),flush=True)
'''
command="bash -c 'source /DATA/disk1/zbw_tmp4/drawio-agent-qwen38/runtime-env.sh >/dev/null; exec env CUDA_VISIBLE_DEVICES=\"\" PYTHONDONTWRITEBYTECODE=1 /DATA/disk1/zbw_tmp4/drawio-agent-sft-20261121/env/bin/python -'"
p=subprocess.run(['/Users/shaoshitong/Documents/deepseek-harness/default-workspace/h20b/bssh','-oConnectTimeout=12','-S','/tmp/drawio-sft-h20-20261123.sock',command],input=script,text=True,capture_output=True,timeout=180)
if p.returncode:
 print(json.dumps({'exit':p.returncode,'stderr_tail':p.stderr[-2500:],'stdout_tail':p.stdout[-500:]}));raise SystemExit(1)
d=json.loads(p.stdout)
q=Path('/Users/shaoshitong/Documents/Codex/2026-11-02/chat-thread-codex-threads-01a19ad1-3f77-2/outputs/dsh-trajectory-lab/loss-normalization-audit.json')
q.write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps({k:d[k] for k in ('verification','global_token_Rw_fraction','per_sample_Rw_fraction','per_update_Rw_fraction','mean_reasoning_coefficient_mass','mean_other_coefficient_mass','reasoning_fraction_of_summed_coefficient_mass')}))
