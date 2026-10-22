"""Generic step ESTIMATE add-on. Public hip validation; chest accuracy unmeasured."""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
from .gp2_algorithms import runs,detect_regions
from .gp2_context import features,regions_from_probabilities

ROOT=Path(__file__).resolve().parent


class StepCounter:
    def __init__(self,threads=2):
        if not isinstance(threads,int) or not 1<=threads<=64:raise ValueError('threads must be1-64')
        self.threads=threads
        self.validation=json.loads((ROOT/'VALIDATION.json').read_text())
        path=ROOT/'models/context.joblib'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=self.validation['recipe']['model_sha256']:
            raise ValueError('Step model hash mismatch')
        self.model=joblib.load(path)

    def predict(self,accel_mps2,*,sample_rate_hz=25,valid=None,excluded=None,time_s=None,include_candidates=False):
        """Input is regular25Hz acceleration; masks cut every calculation.

        No step-count extrapolation into unobserved time. The original activity
        timeline is not changed. No fitted or persistent device calibration.
        """
        x=np.asarray(accel_mps2,float)
        if x.ndim!=2 or x.shape[1]!=3 or not len(x):raise ValueError('Nonempty Nx3 acceleration in m/s2 required')
        if not math.isfinite(float(sample_rate_hz)) or float(sample_rate_hz)!=25:raise ValueError('This validated recipe requires regular25Hz input; resample before calling')
        n=len(x);fs=25
        def mask(value,default):
            if value is None:return np.full(n,default,bool)
            value=np.asarray(value)
            if value.shape!=(n,) or value.dtype!=bool:raise ValueError('Masks must be matching boolean arrays')
            return value
        e=mask(excluded,False);v=mask(valid,True)&np.isfinite(x).all(axis=1)&~e
        t=np.arange(n)/fs if time_s is None else np.asarray(time_s,float)
        # Author-resampled OxWalk has timestamp rounding jitter up to30ms while
        # sample positions remain on a25Hz grid. Admit bounded rounding jitter,
        # not a cumulative missing-sample gap or a different sampling rate.
        if t.shape!=(n,) or not np.isfinite(t).all() or (n>1 and (np.any(np.diff(t)<=0) or np.max(np.diff(t))>.081 or np.max(np.abs(t-t[0]-np.arange(n)/fs))>.040001)):
            raise ValueError('Regular finite25Hz clock required; split timestamp gaps before calling')
        events=[];bout_rows=[];analyzed=0.;unresolved=0.;reasons=[];interval_sum=0.;interval_count=0
        for a,b in runs(v):
            if b-a<4*fs:
                unresolved+=(b-a)/fs;reasons.append(dict(start_s=float(t[a]),end_s=float(t[b-1]+1/fs),reason='UNDER_FOUR_SECONDS'))
                continue
            signal=x[a:b]
            with threadpool_limits(limits=self.threads):
                prob=self.model.predict_proba(features(signal))[:,1]
            regions=regions_from_probabilities(prob,.5,len(signal))
            ic,diag=detect_regions(signal,fs,regions,'B2_shin')
            failed=sum((d['end']-d['start'])/fs for d in diag if d['status'] in ['FAILED','TOO_SHORT'])
            analyzed+=(b-a)/fs-failed;unresolved+=failed
            for d in diag:
                if d['status']=='FAILED':reasons.append(dict(start_s=float(t[a]+d['start']/fs),end_s=float(t[a]+d['end']/fs),reason=d['status']))
            times=t[a+ic];events.extend(times.tolist())
            # A gate break or real long gap breaks cadence adjacency.
            for lo,hi in regions:
                ts=times[(ic>=lo)&(ic<hi)]
                for part in np.split(ts,np.flatnonzero(np.diff(ts)>2.)+1):
                    if not len(part):continue
                    eligible=len(part)>=6 and part[-1]-part[0]>=5
                    cadence=float(60*(len(part)-1)/(part[-1]-part[0])) if eligible else None
                    if eligible:interval_sum+=float(part[-1]-part[0]);interval_count+=len(part)-1
                    bout_rows.append(dict(start_s=float(part[0]),end_s=float(part[-1]),estimated_steps=int(len(part)),estimated_cadence_spm=cadence))
        out=dict(schema='imu.steps.estimate.v2',algorithm_version='GP2-G3-B2-frozen',
                 status='GENERIC_ESTIMATE_WITH_LIMITATIONS' if analyzed else 'NO_ANALYZABLE_INPUT',
                 estimated_step_count=int(len(events)) if analyzed else None,
                 estimated_cadence_spm=60*interval_count/interval_sum if interval_sum else None,
                 analyzed_s=analyzed,unresolved_valid_s=unresolved,invalid_s=float((~v&~e).sum()/fs),excluded_s=float(e.sum()/fs),
                 observed_s=float(v.sum()/fs),coverage_of_observed=analyzed/(v.sum()/fs) if v.any() else None,
                 cadence_analyzed_interval_s=interval_sum,bouts=bout_rows,unresolved_reasons=reasons,
                 count_extrapolated=False,step_time_median_s=None,step_time_iqr_s=None,
                 validation_scope='OxWalk15heldoutpeople,hip25Hz; recording-totalMAPE6.14%,10s count error23.40%,negative50.80steps/h; chest accuracy not measured',
                 reference_document='VALIDATION.json')
        if include_candidates:out['candidate_event_times_s']=events
        return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,help='npz: accel_mps2[N,3], optional valid/excluded/time_s');p.add_argument('--output',required=True)
    p.add_argument('--activity-json',help='Optional unchanged activity result, nested under activity in combined output')
    args=p.parse_args()
    with np.load(args.input,allow_pickle=False) as z:
        result=StepCounter().predict(z['accel_mps2'],**{k:z[k] for k in ['valid','excluded','time_s'] if k in z})
    if args.activity_json:result=dict(schema='imu.activity-with-steps.v1',activity=json.loads(Path(args.activity_json).read_text()),steps=result)
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(args.output)


if __name__=='__main__':main()
