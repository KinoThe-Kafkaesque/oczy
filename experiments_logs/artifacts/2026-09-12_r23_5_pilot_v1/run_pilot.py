import json,os,shutil,subprocess,sys,time
from pathlib import Path
repo=Path('/home/nyanpasu/Desktop/code/kinoSoft/oczy')
sys.path.insert(0,str(repo))
from scripts.reproduce_r20_dev import namespace_prefix
from infrastructure.kaggle.runtime_manifest import observe_runtime_manifest,validate_runtime_manifest
root=Path(__file__).parent
pilot=repo/'experiments/r23.5-serialization-dev/pilot_v1'
model=root.parent/'2026-09-11-organ-identity/model'
provenance=Path.home()/'.local/state/oczy/remote-queue/campaigns/r20-int8-dev-calibration-v6/calibration-a8c98d6/results/cal3-d0-t00-03/remote_run_provenance.json'
p=json.loads(provenance.read_text());expected=validate_runtime_manifest(p['job_spec']['runtime_manifest'])
observed=observe_runtime_manifest(model_root=model,logical_model_id=expected['model']['logical_model_id'],resolved_model_convention=expected['model']['resolved_model_convention'],generation_config=expected['greedy_generation'],quantization=expected['model']['quantization'])
assert observed==expected
manifest=json.loads((pilot/'MANIFEST.json').read_text())
assert observed['manifest_sha256']==manifest['runtime_manifest_sha256']
env=dict(os.environ)
env.update({'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OCZY_REMOTE_CPU_ONLY':'1','OCZY_MODEL_DIR':p['model_root'],'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false','HF_HUB_DISABLE_PROGRESS_BARS':'1','TQDM_DISABLE':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(repo/'src')})
records=[]
for phase in ('reference','restore_text','train','restore_soft'):
 prefix=namespace_prefix(model,p['model_root'],root)
 hidden=[pilot/'audit']
 if phase=='train':hidden += [pilot/'probes',root/'reference',root/'restore_text',root/'released_text']
 if phase.startswith('restore_'):hidden += [pilot/'training',root/'reference']
 if phase=='restore_soft':hidden += [root/'train',root/'released_text',root/'restore_text']
 for directory in hidden:prefix += ['--tmpfs',str(directory)]
 prefix += ['--ro-bind','/dev/null',str(repo/'scripts/prepare_serialization_pilot.py'),'--chdir',str(repo),'--']
 forbidden=[]
 for directory in hidden:
  for file in directory.rglob('*') if directory.exists() else []:
   if file.is_file():forbidden.append(str(file))
 check='from pathlib import Path; import json; paths='+repr(forbidden)+'; visible=[p for p in paths if Path(p).is_file()]; print(json.dumps({"forbidden_files_checked":len(paths),"visible_forbidden_files":visible})); assert not visible'
 checked=subprocess.run(prefix+[sys.executable,'-c',check],env=env,text=True,capture_output=True,check=True)
 args=[sys.executable,str(repo/'scripts/serialization_pilot_worker.py'),phase,'--root',str(pilot),'--output',str(root/phase)]
 if phase=='restore_text':args += ['--artifacts',str(root/'released_text')]
 if phase=='restore_soft':args += ['--artifacts',str(root/'released_states')]
 cmd=prefix+args
 print('Starting '+phase+'; '+checked.stdout.strip(),flush=True)
 started=time.monotonic()
 with (root/(phase+'.log')).open('x') as log:
  process=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=7200)
 record={'phase':phase,'command':cmd,'exit_code':process.returncode,'seconds':time.monotonic()-started,'filesystem_firewall':json.loads(checked.stdout),'runtime_manifest_sha256':observed['manifest_sha256'],'manifest_sha256':manifest['manifest_sha256'],'network_disabled':True,'source_state':'local uncommitted diagnostic code'}
 records.append(record)
 (root/'executions.json').write_text(json.dumps(records,indent=2)+'\n')
 print(json.dumps({k:v for k,v in record.items() if k!='command'}),flush=True)
 if process.returncode:
  print((root/(phase+'.log')).read_text()[-4000:],flush=True)
  raise SystemExit(process.returncode)
 result=json.loads((root/phase/'results.json').read_text())
 if 'rows' in result:
  counts={}
  for row in result['rows']:
   key=str(row['pattern'])+'/'+row['condition'];c=counts.setdefault(key,[0,0]);c[0]+=int(row['correct']);c[1]+=1
  print(json.dumps(counts),flush=True)
 if phase=='reference':
  release=root/'released_text';release.mkdir()
  textmeta=json.loads((root/phase/'text_manifest.json').read_text())
  for entry in textmeta['artifacts']:shutil.copyfile(root/phase/entry['path'],release/entry['path'])
  shutil.copyfile(root/phase/'text_manifest.json',release/'text_manifest.json')
 if phase=='train':shutil.copytree(root/'train/states',root/'released_states')
