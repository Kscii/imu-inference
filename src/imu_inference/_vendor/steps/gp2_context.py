"""Public-only small step-context classifier features; acceleration in SI.

One output per second with four seconds of context. Offline/buffered inference.
Features are rotation invariant and do not fit a calibration to any device.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt


def features(x, fs=25, *, centers=None):
    x=np.asarray(x,float)/9.80665
    if fs!=25 or x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all():
        raise ValueError('finite25Hz triaxial acceleration required')
    n=int(np.ceil(len(x)/fs)) if centers is None else len(centers);win=4*fs
    # Reflect signal only for calculation at the recording edge. Original
    # observation masks and scored interval lengths are never extended.
    pad=2*fs
    y=np.pad(x,((pad+fs,pad+2*fs),(0,0)),mode='reflect')
    starts=np.arange(n)*fs+fs if centers is None else np.asarray(centers,int)-fs//2+fs
    w=y[starts[:,None]+np.arange(win)[None,:]]
    norm=np.linalg.norm(w,axis=2)
    gravity=w.mean(axis=1);gravity/=np.maximum(np.linalg.norm(gravity,axis=1,keepdims=True),1e-9)
    vertical=np.sum(w*gravity[:,None,:],axis=2)
    horizontal=np.linalg.norm(w-vertical[:,:,None]*gravity[:,None,:],axis=2)
    blocks=[]
    for v in [norm,vertical,horizontal]:
        q=np.quantile(v,[.05,.25,.5,.75,.95],axis=1).T
        blocks.extend([v.mean(axis=1),v.std(axis=1),v.max(axis=1)-v.min(axis=1),q[:,4]-q[:,0],q[:,3]-q[:,1]])
    demean=norm-norm.mean(axis=1,keepdims=True)
    spec=np.abs(np.fft.rfft(demean*np.hanning(win),axis=1))**2
    freq=np.fft.rfftfreq(win,1/fs);spec[:,0]=0
    total=spec.sum(axis=1)+1e-12;prob=spec/total[:,None]
    for a,b in [(0.5,1),(1,2),(2,3),(3,5),(5,10)]:blocks.append(spec[:,(freq>=a)&(freq<b)].sum(axis=1)/total)
    ix=spec.argmax(axis=1);blocks += [freq[ix],spec[np.arange(n),ix]/total,-np.sum(prob*np.log(prob+1e-12),axis=1)]
    # Autocorrelation features distinguish cyclic gait from many isolated motions.
    fft=np.fft.rfft(demean,n=2*win,axis=1)
    ac=np.fft.irfft(fft*np.conj(fft),n=2*win,axis=1)[:,:win]
    ac/=np.maximum(ac[:,0,None],1e-10)
    lo,hi=int(.28*fs),int(1.6*fs)
    lag=ac[:,lo:hi].argmax(axis=1)+lo
    blocks += [ac[np.arange(n),lag],lag/fs]
    jerk=np.linalg.norm(np.diff(w,axis=1),axis=2)*fs
    blocks += [jerk.mean(axis=1),jerk.std(axis=1),np.quantile(jerk,.95,axis=1)]
    # The center output second is at [2s,3s) of the padded4s context.
    center=w[:,2*fs:3*fs];cn=np.linalg.norm(center,axis=2)
    blocks += [cn.std(axis=1),np.ptp(cn,axis=1),np.linalg.norm(np.diff(center,axis=1),axis=2).mean(axis=1)]
    out=np.column_stack(blocks).astype(np.float32)
    if not np.isfinite(out).all():raise ValueError('Non-finite step-context feature')
    return out


def candidate_features(x, indices, fs=25):
    """Physical signal and local cycle context; no reference or labels used."""
    indices=np.asarray(indices,int)
    if not len(indices):return np.empty((0,41),np.float32)
    f=features(x,fs,centers=indices)
    dt=np.diff(indices)/fs
    before=np.r_[2.,dt];after=np.r_[dt,2.]
    norm=np.linalg.norm(x,axis=1)/9.80665
    local=np.arange(-3,4)
    narrow=norm[np.clip(indices[:,None]+local[None,:],0,len(norm)-1)]
    extra=np.column_stack([np.minimum(before,5),np.minimum(after,5),
        np.minimum(np.abs(before-after),5),np.minimum(before/(after+.04),5),
        narrow[:,3],narrow.std(axis=1),np.ptp(narrow,axis=1),
        np.sum((np.abs(indices[:,None]-indices[None,:])<=2*fs),axis=1) if len(indices)<15000 else np.searchsorted(indices,indices+2*fs,side='right')-np.searchsorted(indices,indices-2*fs)])
    return np.column_stack([f,extra]).astype(np.float32)


def regions_from_probabilities(probability,threshold,n_samples,fs=25):
    from .gp2_algorithms import runs
    return [(int(a*fs),int(min(b*fs,n_samples))) for a,b in runs(np.asarray(probability)>=threshold) if a*fs<n_samples]
