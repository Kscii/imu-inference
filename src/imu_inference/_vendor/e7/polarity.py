"""Longitudinal sign only; not arbitrary orientation calibration."""
import numpy as np

def estimate(x,valid):
    norm=np.linalg.norm(x[:,:3],axis=1);cos=np.divide(x[:,1],norm,out=np.zeros(len(x)),where=norm>0)
    good=valid & np.isfinite(x[:,:3]).all(1)&(norm>=7)&(norm<=12)&(abs(cos)>=.8)
    edge=np.diff(np.r_[False,good,False].astype(np.int8));selected=np.zeros(len(x),bool)
    # Windows must retain one sign; transitions or gaps cannot form an anchor.
    for a,b in zip(np.flatnonzero(edge==1),np.flatnonzero(edge==-1)):
        if b-a>=50 and np.all(np.sign(cos[a:b])==np.sign(cos[a])):selected[a:b]=True
    n=int(selected.sum());positive=int((cos[selected]>0).sum());agreement=max(positive,n-positive)/max(n,1)
    accepted=n>=125 and agreement>=.9
    return dict(accepted=accepted,sign=(1 if positive>=n-positive else -1) if accepted else 1,
        anchor_samples=n,agreement=agreement,fallback='identity' if not accepted else None),selected
