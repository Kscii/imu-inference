"""Small installation/interface smoke check; scientific evidence is in VALIDATION.json."""
import json
import numpy as np
from step_count import StepCounter

c=StepCounter();x=np.zeros((250,3));x[:,2]=9.80665
assert c.predict(x)['estimated_step_count']==0
assert c.predict(x,valid=np.zeros(250,bool))['estimated_step_count'] is None
assert c.predict(x[:25])['unresolved_valid_s']==1
x[50:100]=np.nan;r=c.predict(x)
assert r['invalid_s']==2 and r['estimated_step_count']==0
print(json.dumps({'status':'PASS','scientific_all_gates_passed':False,'check':'local installation and missing/zero semantics'}))
