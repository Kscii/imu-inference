"""P1-compatible timestamp resampling of already calibrated SI values.

The caller supplies the recording grid and exclusion mask. No pose/calibration
parameters are estimated from activity labels or the recording.
"""
import numpy as np
from scipy.signal import resample_poly

def _spans(mask):
    edge=np.diff(np.r_[False,mask,False].astype(np.int8))
    return zip(np.flatnonzero(edge==1),np.flatnonzero(edge==-1))

def prepare_samples(timestamps_ns, values, *, native_rate_hz, grid_start_ns,
                    sample_count, signal_valid=None, excluded=None):
    t=np.asarray(timestamps_ns);x=np.asarray(values,dtype=np.float64)
    if t.ndim!=1 or not np.issubdtype(t.dtype,np.integer) or len(t)<2:raise ValueError('need at least two integer nanosecond timestamps')
    if x.shape!=(len(t),6) or not np.isfinite(x).all():raise ValueError('raw SI values must be finite N by6; upstream must mark/remove bad raw rows')
    if not np.all(np.diff(t)>0):raise ValueError('timestamps must strictly increase')
    if native_rate_hz not in [25,50]:raise ValueError('supported native rates are25 and50Hz')
    if not isinstance(sample_count,int) or sample_count<1:raise ValueError('sample_count must be positive')
    nominal=1e9/native_rate_hz;median=float(np.median(np.diff(t)))
    if not .9*nominal<=median<=1.1*nominal:raise ValueError('declared rate disagrees with timestamps')
    valid=np.ones(sample_count,bool) if signal_valid is None else np.asarray(signal_valid).copy()
    e=np.zeros(sample_count,bool) if excluded is None else np.asarray(excluded)
    if valid.shape!=(sample_count,) or e.shape!=(sample_count,) or valid.dtype!=bool or e.dtype!=bool:raise ValueError('output validity/exclusion masks must be boolean length sample_count')
    grid=int(grid_start_ns)+np.arange(sample_count,dtype=np.int64)*40_000_000
    valid&=(grid>=int(t[0])-40_000_000)&(grid<=int(t[-1])+40_000_000)
    # No interpolation across unobserved raw gaps exceeding200ms.
    for i in np.flatnonzero(np.diff(t)>200_000_000):valid[(grid>t[i])&(grid<t[i+1])]=False
    task=valid&~e;out=np.zeros((sample_count,6),np.float32);edge=np.zeros(sample_count,bool)
    for a,b in _spans(task):
        lo=int(grid[a]);hi=lo+int(b-a)*40_000_000;keep=(t>=lo)&(t<hi);tt=t[keep];v=x[keep]
        if len(tt)==0:
            valid[a:b]=False;continue
        if native_rate_hz==50:
            relative=np.arange(int(b-a)*40,dtype=np.int64)*1_000_000
            hold=max(int(tt[0]-lo),int(lo+relative[-1]-tt[-1]),0)
            if hold>40_000_000:raise ValueError('raw span needs over40ms endpoint extrapolation; split validity mask')
            dense=np.column_stack([np.interp(relative,tt-lo,v[:,k]) for k in range(6)])
            out[a:b]=resample_poly(dense,1,40,axis=0,window=('kaiser',5.),padtype='line').astype(np.float32)
            edge[a:min(a+11,b)]=True;edge[max(a,b-11):b]=True
        else:
            out[a:b]=np.column_stack([np.interp(grid[a:b]-lo,tt-lo,v[:,k]) for k in range(6)]).astype(np.float32)
    return dict(x=out,valid=valid,excluded=e.copy(),rate_hz=np.array(25),grid_start_ns=np.array(grid_start_ns,dtype=np.int64),filter_edge=edge)
