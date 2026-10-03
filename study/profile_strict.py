"""Measure the pre-test-frozen100% sensitivity choice against a same-block control.

Added at final scientific assessment, after the initial21 trials/external study.
No policy, threshold, configuration or test metric is selected anew.
"""
import json,subprocess,sys
from run_study import HERE,RESULT,atomic,state,now,sha,full_profile
selection=json.loads((RESULT/'selection.json').read_text());strict=selection['policies']['1.0']
assert strict['object']['id']=='480_fp32' and strict['behavior']['id']=='640_rect_fp32'
assert (RESULT/'external_airtlab/summary.json').exists()
policies={'strict_100':{'object':strict['object']['id'],'behavior':strict['behavior']['id'],'geometry':'letterbox'},'strict_control_640_fp32':{'object':'640_fp32','behavior':'640_fp32','geometry':'letterbox'}}
protocol=HERE/'STRICT_SENSITIVITY_PROTOCOL.json'
if not protocol.exists():atomic(protocol,{'recorded_utc':now(),'selection_frozen_utc':selection['frozen_utc'],'primary_policy_unchanged':True,'selected_sensitivity_retention':1.0,'policies':policies,'timing_repeats':3,'frames_per_repeat':239,'order_seed':20261004,'reason':'Directly measure the already-frozen strict-retention heterogeneous choice; add a contemporaneous control because original timing occurred in an earlier session. No new policy selection from test or benchmark results. Same full-output function/boundary.','run_script_sha256':sha(HERE/'run_study.py')})
full_profile(selection,policy_overrides=policies,block='strict_sensitivity')
for name,args,log in [('native_predictor.py',['--include-strict'],'native_strict.log'),('analyze.py',[],'analysis_final.log')]:
 with (HERE/log).open('w',encoding='utf-8') as stream:subprocess.run([sys.executable,str(HERE/name),*args],stdout=stream,stderr=subprocess.STDOUT,check=True)
state('strict_sensitivity_complete','Six contemporaneous timing trials; frozen strict policy + control; native object480 check and final aggregation complete')
