import base64, csv, hashlib, json, os, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

from sft.audit_rollout_gap import RUNS,read_rows
from sft.causal_cot import apply_rewrite
from sft.data import load_manifest,open_store
from sft.history import strip_history_thinking
from sft.summarize_capture import summarize
from transformers import AutoTokenizer

B=Path("/DATA/disk1/zbw_tmp4/drawio-agent-sft-20261121")
out=B/"checks/trajectory-dsh-view-20261124"
manifest=B/"checks/dsh-midflight-20261123/curriculum-final.jsonl"
selected="5e4624820153e82f"
same="ebf9d337d02a5300"
FIELDS=("past_output","past_thinking","past_tools","thinking","output")

def init():
 global tokenizer,stores
 tokenizer=AutoTokenizer.from_pretrained(B/"models/Qwen3.5-9B",local_files_only=True)
 stores={}

@lru_cache(maxsize=4096)
def tokens(s):
 return len(tokenizer.encode(s,add_special_tokens=False))

def text(m):
 value=m.get("content") or ""
 if isinstance(value,list):
  return "\n".join(x.get("text","") for x in value if x.get("type") in {"text","input_text","output_text"})
 return value

def thinking(m):
 return m.get("reasoning_content") or m.get("reasoning") or ""

def call_args(c):
 a=c["function"].get("arguments") or ""
 return a if isinstance(a,str) else json.dumps(a,ensure_ascii=False,separators=(",",":"))

def output(m):
 return tokens(text(m))+sum(tokens(c["function"].get("name") or "")+tokens(call_args(c)) for c in m.get("tool_calls") or [])

def stats(history,target):
 past=[m for m in history if m["role"]=="assistant"]
 return dict(zip(FIELDS, (sum(map(output,past)),sum(tokens(thinking(m)) for m in past),
   sum(len(m.get("tool_calls") or []) for m in past),tokens(thinking(target)),output(target))))

def short(s,limit):
 return s if len(s)<=limit else s[:limit]+"\n…（页面省略 "+str(len(s)-limit)+" 个字符）"

def calls(m,limit):
 return [{"id": c.get("id",""), "name":c["function"].get("name",""), 
          "args":short(call_args(c),limit), "argument_tokens":tokens(call_args(c))}
          for c in m.get("tool_calls") or []]

def history_rows(history):
 return [{"role":m["role"],"id":m.get("tool_call_id",""),
          "text":short(text(m),900),"images":sum(1 for p in m["content"] if p.get("type") in {"image","image_url"})
              if isinstance(m.get("content"),list) else 0,
          "thinking":short(thinking(m),500) if m["role"]=="assistant" else "",
          "calls":calls(m,350)}
          for m in history]

def detail(history,target,values):
 prior=history_rows(history)
 return {**values, "history":prior,"reply":short(text(target),8000),"thought":short(thinking(target),9500),
         "tool_calls":calls(target,9500), "reply_tokens":tokens(text(target)),
         "parameter_tokens":sum(tokens(call_args(c)) for c in target.get("tool_calls") or [])}

def read_store(root):
 if root not in stores:
  stores[root]=open_store(root)
 return stores[root]

def train(row):
 archive=read_store(row["snapshot_root"])
 request=archive.get(row["request_ref"])
 response=archive.get(row["response_ref"])
 rewrite=row.get("reasoning_rewrite")
 original_thought=thinking(response["choices"][0]["message"])
 if rewrite:
  request,response=apply_rewrite(request,response,row,read_store(rewrite["root"]).get(rewrite["ref"]))
 history=strip_history_thinking(request["messages"])
 target=response["choices"][0]["message"]
 values=stats(history,target)
 if row["episode"] not in {selected,same}: return row["split"],values,None
 return row["split"],values,{**detail(history,target,values),
          "id":row["sample_id"],"number":row["call_id"],"stage":row["stage"],
          "input":row["prompt_tokens"],"original_thinking":tokens(original_thought),
          "components":row["final_components"]}

def attach_results(rows):
 found={m["id"]:m["text"] for r in rows for m in r["history"] if m["role"]=="tool" and m["id"]}
 for row in rows:
  for c in row["tool_calls"]:
   c["result"]=found.get(c["id"])

def recorded():
 groups={}
 for label,run in RUNS.items():
  folder=B/"runs/causal-cot-20261122"/run/"evaluation"
  archived=read_rows(folder/"generator-context-requests.jsonl")
  rows=[]
  for i,r in enumerate(summarize(folder/"generator-context-requests-raw",audit=archived),1):
   request=json.loads(Path(r["request"]).read_text()); reply=r["response"]
   target={"content":reply["content_text"],"reasoning_content":reply["reasoning_text"],
           "tool_calls":[{"id":c.get("id",""),"function":{"name":c["name"],"arguments":c.get("arguments","")}}
                         for c in reply["tool_calls"].values()]}
   values=stats(request["messages"],target)
   state=r["state_prefix"]
   rows.append({**detail(request["messages"],target,values),"id":r["request_sha256"],"number":i,
      "input":r["prompt_tokens"],"max_output":r["requested_max_tokens"],"complete":r["complete_stream"],
      "finish":reply.get("finish_reason"),"xml":any(name.endswith(".drawio") for name in state.get("files_sha256",{})),
      "history_sha256":r["request_sha256"],"response_sha256":r["response_sha256"]})
  attach_results(rows)
  groups[label]=rows
  if label=="base9b":
   for file,name in ((folder/"case/reference.png","reference.png"),(folder/"case/round1.png","base-draft.png")):
    if file.is_file(): (out/name).write_bytes(file.read_bytes())
 return groups

def status(p):
 (out/"progress.json").write_text(json.dumps(p)+"\n")
 print(json.dumps(p),flush=True)

def main():
 out.mkdir(exist_ok=True)
 init()
 runs=recorded()
 by_id={r["number"]:r["id"] for r in runs["step170"]}
 for r in runs["step170_resume"]:
  r["copied"]=by_id.get(r["number"])==r["id"]
 train_rows=load_manifest(manifest)
 assert hashlib.sha256(manifest.read_bytes()).hexdigest()=="cf5d343114153f1a3ef50861b93980efc6c80e020f2ffab22777a860651e2d78"
 selected_rows=[r for r in train_rows if r["split"]=="train" or r["episode"]==same]
 totals=Counter()
 scoped={"sft":[],"matched":[]}
 status({"live":{k:len(v) for k,v in runs.items()},"training_rows":len(selected_rows),"processed":0})
 with ProcessPoolExecutor(max_workers=8,initializer=init) as executor:
  for i,(split,values,info) in enumerate(executor.map(train,selected_rows,chunksize=24),1):
   if split=="train":
    totals.update(values)
    totals["n"]+=1
   if info:
    scoped["sft" if split=="train" else "matched"].append(info)
   if i%2000==0: status({"processed":i,"accepted":totals["n"]})
 for k,rows in scoped.items():
  rows.sort(key=lambda r:r["number"])
  for i,r in enumerate(rows,1): r["ordinal"]=i
  attach_results(rows)
 # The training reference is from the exact first request, resolved using the committed archive.
 r=next(r for r in selected_rows if r["episode"]==selected)
 first=read_store(r["snapshot_root"]).get(r["request_ref"])
 for m in first["messages"]:
  c=m.get("content") or []
  if isinstance(c,list):
   image=next((x.get("image_url",{}).get("url") for x in c if x.get("type")=="image_url"),None)
   if image and image.startswith("data:image/"):
    (out/"sft-reference.png").write_bytes(base64.b64decode(image.split(",",1)[1]))
    break
 assert totals["n"]==17839 and len(scoped["matched"])==206 and len(scoped["sft"])==124
 payload={"runs":runs,"sft":scoped["sft"],"matched":scoped["matched"],
          "all_train":{k:totals[k]/totals["n"] for k in FIELDS},"n_train":totals["n"],
          "assets":{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in
                    ("reference.png","base-draft.png","sft-reference.png") if (out/name).is_file()}}
 tmp=out/"report.building.json"
 tmp.write_text(json.dumps(payload,ensure_ascii=False,separators=(",",":")))
 tmp.replace(out/"report.json")
 status({"processed":len(selected_rows),"accepted":totals["n"],"complete":True,"views":{k:len(v) for k,v in scoped.items()},
         "report_sha256":hashlib.sha256((out/"report.json").read_bytes()).hexdigest()})
if __name__=="__main__":
 main()

