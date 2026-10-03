def pareto(rows):
 """Nondominated rows: minimize latency, maximize both AP metrics."""
 def dominates(a,b):
  weak=a['ms']<=b['ms'] and a['ap50']>=b['ap50'] and a['ap95']>=b['ap95']
  strict=a['ms']<b['ms'] or a['ap50']>b['ap50'] or a['ap95']>b['ap95']
  return weak and strict
 return [r for r in rows if not any(dominates(q,r) for q in rows)]

def choose(rows,reference_ids,retention):
 """Exact discrete selection under separable task and metric constraints."""
 selected={}
 for task,candidates in rows.items():
  reference=next(r for r in candidates if r['id']==reference_ids[task])
  feasible=[r for r in candidates if all(r[k]+1e-12>=retention*reference[k] for k in ['ap50','ap95'])]
  if not feasible:raise ValueError('No feasible configuration for '+task)
  selected[task]=min(feasible,key=lambda r:(r['ms'],-r['ap95'],-r['ap50'],r['id']))
 selected['estimated_ms']=sum(selected[k]['ms'] for k in rows)
 selected['retention']=retention
 return selected
