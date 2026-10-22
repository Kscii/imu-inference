"""Frozen parent and posterior decoders; all inputs are signal-derived."""
import numpy as np
from scipy.ndimage import uniform_filter1d, maximum_filter1d
from ..activity.engine import spans
PARENT_RECIPE = dict(decoder="context",threshold=.5,hold=5,back=0)


def legacy_overlay(base, corr, pos, valid, threshold, hold_s):
    """Signal-only offline output: direction correction; bounded carry until strong locomotion."""
    tag = base.argmax(1).astype(np.int8)
    new = corr.argmax(1)
    trigger = (new > 0) & (corr.max(1) >= threshold)
    end, state, previous = -1, -1, -100
    for i, t in enumerate(pos):
        # Reset at physical gaps; cannot carry through excluded time.
        if t-previous > 5 or not valid[t] or not valid[max(0, previous):t+1].all():
            end, state = -1, -1
        previous = t
        if trigger[i]:
            state = int(new[i]-1)
            end = int(t+hold_s*25)
            tag[i] = state
        elif state >= 0 and t <= end:
            if base[i, 3] >= .8:
                state, end = -1, -1
            else:
                tag[i] = state
    return tag

def context_decode(base, q, pos, valid, recipe):
    if recipe["decoder"] == "legacy":
        return legacy_overlay(base, q, pos, valid, recipe["threshold"], recipe["hold"])
    tag = base.argmax(1).astype(np.int8)
    cls = q.argmax(1)
    trigger = (q.max(1) >= recipe["threshold"]) & (cls > 0)
    for lo, hi in spans(valid):
        ids = np.flatnonzero((pos >= lo) & (pos < hi))
        if not len(ids):
            continue
        pp, qq = base[ids], q[ids]
        allow = np.ones((len(ids), 3), bool)
        if recipe["decoder"] == "context":
            sm = uniform_filter1d(pp.astype(float), 11, axis=0, mode="nearest")
            j = np.arange(len(ids))
            pre, post = sm[np.maximum(j-15, 0)], sm[np.minimum(j+15, len(j)-1)]
            # Fixed semantic gates, not chosen using target signals/labels.
            # Approximate previous/next stable posture over [-4,-2]/[2,4] sec.
            allow[:, 0] = (post[:, 0] >= .35) & (post[:, 0] >= pre[:, 0]-.10)
            pre_up, post_up = pre[:, 2:].sum(1), post[:, 2:].sum(1)
            allow[:, 1] = (post[:, 0] < .5) & (post_up <= pre_up+.15)
            allow[:, 2] = ((post_up >= pre_up-.15) | (pre[:, 0] >= .35)) & (post[:, 0] <= pre[:, 0]+.10)
        pulse = np.zeros((len(ids), 3), np.float32)
        for k in range(1, 4):
            eligible = trigger[ids] & (cls[ids] == k) & allow[:, k-1]
            pulse[eligible, k-1] = qq[eligible, k]
        # Strong locomotion and physical gaps are hard temporal barriers.
        # A state cannot resume on the other side of a walking interval.
        for aa, bb in spans(pp[:, 3] < .8):
            left, right = int(recipe["hold"]*5), int(recipe["back"]*5)
            width = left+right+1
            evidence = maximum_filter1d(pulse[aa:bb], width, axis=0,
                                        origin=left-width//2, mode="constant", cval=0)
            peak = evidence.max(1)
            ties = (evidence == peak[:, None]).sum(1)
            change = (peak > 0) & (ties == 1)
            local = evidence.argmax(1).astype(np.int8)
            at = ids[aa:bb][change]
            tag[at] = local[change]
    return tag

def parent_decode(base, probability, pos, valid, recipe):
    parent = context_decode(base, probability[:, :4], pos, valid, PARENT_RECIPE)
    tag = parent.copy()
    q = probability[:, 4:]
    assert q.shape == (len(pos), 3)
    for lo, hi in spans(valid):
        ids = np.flatnonzero((pos >= lo) & (pos < hi))
        if not len(ids):
            continue
        b, p = base[ids], q[ids]
        sm = uniform_filter1d(b.astype(float), 11, axis=0, mode="nearest")
        j = np.arange(len(ids))
        pre, post = sm[np.maximum(j-15, 0)], sm[np.minimum(j+15, len(ids)-1)]
        # These conditions also stop persistence; it cannot resume after lying,
        # locomotion, or a physical gap.
        allowed = ((pre[:, 0] < .3) & (post[:, 0] < .3) &
                   (parent[ids] != 0) & (b[:, 3] < .8))
        cls = p.argmax(1)
        pulse = np.zeros((len(ids), 2), np.float32)
        for k in [1, 2]:
            trigger = allowed & (cls == k) & (p[:, k] >= recipe["threshold"])
            pulse[trigger, k-1] = p[trigger, k]
        for start, stop in spans(allowed):
            left, right = int(recipe["hold"]*5), int(recipe["back"]*5)
            width = left+right+1
            evidence = maximum_filter1d(pulse[start:stop], width, axis=0,
                origin=left-width//2, mode="constant", cval=0)
            maximum = evidence.max(1)
            change = (maximum > 0) & ((evidence == maximum[:, None]).sum(1) == 1)
            tag[ids[start:stop][change]] = evidence[change].argmax(1)+1
    return tag

def expand(pos, p, valid):
    full = np.zeros((len(valid), 4), np.float32)
    for lo, hi in spans(valid):
        mask = (pos >= lo) & (pos < hi)
        q = pos[mask]
        if not len(q):
            raise AssertionError("No prediction support in valid span")
        at = np.arange(lo, hi)
        right = np.clip(np.searchsorted(q, at), 0, len(q)-1)
        left = np.maximum(right-1, 0)
        ix = np.where(abs(at-q[left]) <= abs(at-q[right]), left, right)
        full[lo:hi] = p[mask][ix]
    return full



def index_map(valid,pos):
    """Signal-only center/run mapping. Never accepts annotations."""
    
    ix=np.full(len(valid),-1,int);reset=np.zeros(len(pos),bool)
    for a,b in spans(valid):
        ids=np.flatnonzero((pos>=a)&(pos<b));assert len(ids)
        reset[ids[0]]=True;centers=pos[ids];at=np.arange(a,b)
        right=np.minimum(np.searchsorted(centers,at),len(centers)-1);left=np.maximum(right-1,0)
        nearest=np.where(abs(at-centers[left])<=abs(at-centers[right]),left,right)
        ix[a:b]=ids[nearest]
        assert np.all(np.diff(centers)==5)
    return ix,reset

def state_overlay(parent,base_p,valid,index,posterior,confidence):
    out=parent.copy();where=np.flatnonzero(valid);q=posterior[index[where]]
    winner=q.argmax(1);maximum=q.max(1)
    # Protect all parent lying/locomotion outputs and every existing R15 correction.
    allowed=np.isin(parent[where],[1,2]) & (parent[where]==base_p[where].argmax(1))
    allowed &= (base_p[where,0]<.3)&(base_p[where,3]<.8)&np.isin(winner,[1,2])&(maximum>=confidence)
    allowed &= np.sum(q==maximum[:,None],axis=1)==1
    out[where[allowed]]=winner[allowed]
    assert np.array_equal(out[~valid],parent[~valid])
    assert np.array_equal(out[np.isin(parent,[0,3])],parent[np.isin(parent,[0,3])])
    return out

"""E6: correlated windows share an event evidence budget.
Threshold inherited .85. Rearm by confident keep/opposite direction or signal gap.
No labels, elapsed-time cutoffs, record IDs or fitted private parameters.
"""
import numpy as np

def evidence(q,reset,threshold=.85):
 q=np.asarray(q,float);reset=np.asarray(reset,bool)
 assert q.shape==(len(reset),3) and np.isfinite(q).all() and (q>=0).all()
 assert np.allclose(q.sum(1),1,atol=1e-5)
 direction=np.zeros(len(q),np.int8);mix=np.zeros(len(q));episode=np.zeros(len(q),int)
 active=0;peak=0.;eid=0
 for i,row in enumerate(q):
  if reset[i]:active=0;peak=0.
  winner=int(row.argmax())
  if row[winner]<threshold:continue
  if winner==0:active=0;peak=0.;continue
  if winner!=active:active=winner;peak=0.;eid+=1
  new=min(float(row[winner]),1-1e-9)
  direction[i]=winner;episode[i]=eid
  if new>peak:mix[i]=(new-peak)/(1-peak);peak=new
 return direction,mix,episode

def probabilities(p,q,reset,base,weight=.05,threshold=.85):
 direction,mix,episode=evidence(q,reset,threshold)
 em=np.maximum(np.asarray(p,float),1e-9)**weight;em/=em.sum(1,keepdims=True)
 matrices=np.repeat(np.asarray(base,float)[None,:,:],len(em),axis=0)
 assert (matrices>0).all() and np.allclose(matrices.sum(2),1)
 for i in np.flatnonzero(mix):
  src,dst=(2,1) if direction[i]==1 else (1,2)
  matrices[i,src]*=1-mix[i];matrices[i,src,dst]+=mix[i]
 f=np.empty_like(em);prev=np.full(4,.25)
 for i in range(len(em)):
  v=(np.full(4,.25) if reset[i] else prev@matrices[i])*em[i];v/=v.sum();f[i]=prev=v
 b=np.ones_like(em)
 for i in range(len(em)-2,-1,-1):
  if reset[i+1]:continue
  v=matrices[i+1]@(em[i+1]*b[i+1]);b[i]=v/v.sum()
 post=f*b;post/=post.sum(1,keepdims=True)
 return dict(forward=f,smoothed=post),dict(direction=direction,mix=mix,episode=episode)

