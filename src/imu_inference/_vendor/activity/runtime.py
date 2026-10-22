"""Offline classification and accounting of already calibrated 25 Hz chest IMU."""
from pathlib import Path
import json
import hashlib
import numpy as np

LABELS=['lying','sitting','standing','walking_running','posture_transition','other_activity']

class ActivityModel:
    def __init__(self, directory, threads=2):
        self.directory=Path(directory)
        self.metadata=json.loads((self.directory/'model.json').read_text())
        self.required_modalities=self.metadata.get('required_modalities',['acceleration','angular_velocity'])
        if not self.required_modalities or any(k not in ('acceleration','angular_velocity') for k in self.required_modalities):raise ValueError('unknown or empty required modalities')
        self.required_columns=[i for k,columns in (('acceleration',range(3)),('angular_velocity',range(3,6))) if k in self.required_modalities for i in columns]
        self.labels=self.metadata.get('labels',LABELS[:int(self.metadata['class_count'])])
        if len(self.labels)!=int(self.metadata['class_count']) or len(set(self.labels))!=len(self.labels):raise ValueError('invalid model label schema')
        if 'output_groups' in self.metadata:
            groups=self.metadata['output_groups']
            if len(groups)!=len(self.labels) or any(not group for group in groups) or sorted(i for group in groups for i in group)!=list(range(4)):raise ValueError('output groups must partition four native classes')
        elif len(self.labels)!=4:raise ValueError('this runtime requires four native classes or an explicit output mapping')
        if not isinstance(threads,int) or not 1<=threads<=32:raise ValueError('threads must be between 1 and32')
        weights=self.directory/self.metadata['weights_file']
        with weights.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=self.metadata['weights_sha256']:raise ValueError('model weights checksum mismatch')
        from . import engine
        self.engine=engine;self.model=engine.load(weights,self.metadata,threads)

    def predict(self, values, *, valid=None, excluded=None, min_confidence=0., recording_id='recording'):
        """SI units: m/s2 and rad/s. A row covers [i/25,(i+1)/25)."""
        x=np.asarray(values,dtype=np.float32)
        if x.ndim!=2 or x.shape[1]!=6 or len(x)==0:raise ValueError('values must be a nonempty N by6 array')
        if not 0<=min_confidence<=1:raise ValueError('min_confidence must be between0 and1')
        n=len(x);v=np.ones(n,bool) if valid is None else np.asarray(valid)
        e=np.zeros(n,bool) if excluded is None else np.asarray(excluded)
        if v.shape!=(n,) or e.shape!=(n,) or v.dtype!=bool or e.dtype!=bool:raise ValueError('valid and excluded must be boolean arrays of lengthN')
        finite=np.isfinite(x[:,self.required_columns]).all(1);signal=v&finite;task=signal&~e
        clean=np.where(np.isfinite(x),x,0).astype(np.float32)
        p=self.engine.predict(self.model,clean,task,self.metadata)
        classes=int(self.metadata['class_count'])
        if p.shape!=(n,classes) or not np.isfinite(p).all() or not np.allclose(p[task].sum(1),1,atol=1e-5):raise RuntimeError('invalid model output')
        confidence=p.max(1);tag=p.argmax(1).astype(np.int16);tag[~task]=-1;tag[task&(confidence<min_confidence)]=-1
        state=np.full(n,'observed',dtype='<U12');state[~signal]='invalid';state[e]='excluded'
        edges=np.r_[0,np.flatnonzero((tag[1:]!=tag[:-1])|(state[1:]!=state[:-1]))+1,n]
        timeline=[]
        for a,b in zip(edges[:-1],edges[1:]):
            label=self.labels[int(tag[a])] if tag[a]>=0 else ('unclassified' if state[a]=='observed' else None)
            timeline.append(dict(start_s=int(a)/25,end_s=int(b)/25,label=label,status=str(state[a]),mean_model_confidence=float(confidence[a:b].mean()) if state[a]=='observed' else None))
        duration={label:float(np.sum(task&(tag==i))/25) for i,label in enumerate(self.labels)}
        duration['unclassified']=float(np.sum(task&(tag<0))/25);observed=float(task.sum()/25)
        output=dict(schema='imu.activity.recording.v1',recording_id=str(recording_id),model_version=self.metadata['model_version'],active_labels=self.labels,sample_rate_hz=25,samples=n,recording_duration_s=n/25,observed_s=observed,invalid_s=float(np.sum(~signal&~e)/25),excluded_s=float(e.sum()/25),duration_s=duration,fraction_of_observed={k:v/observed if observed else None for k,v in duration.items()},timeline=timeline,min_confidence=min_confidence,confidence_calibrated=False,step_count=None,completed_transition_count=None,fall_count=None,fall_count_source='external_edge_detector',model_acceptance=self.metadata['acceptance_status'])
        assert abs(sum(duration.values())-observed)<1e-7
        assert abs(observed+output['invalid_s']+output['excluded_s']-n/25)<1e-7
        return output

def calendar_segments(result, participant_id, start_time):
    """Attach a caller-verified start time. Never infer it from file timestamps."""
    import datetime as dt
    base=dt.datetime.fromisoformat(start_time.replace('Z','+00:00'))
    if base.tzinfo is None or base.utcoffset() is None:raise ValueError('start_time needs an explicit UTC offset')
    rows=[]
    for i,row in enumerate(result['timeline']):
        rows.append(dict(recording_id=result['recording_id'],segment_id=f"{result['recording_id']}:{i}",start=(base+dt.timedelta(seconds=row['start_s'])).isoformat(),end=(base+dt.timedelta(seconds=row['end_s'])).isoformat(),label=row['label'],status=row['status']))
    return dict(schema='imu.activity.segments.v1',participant_id=participant_id,model_version=result['model_version'],active_labels=result['active_labels'],segments=rows)
