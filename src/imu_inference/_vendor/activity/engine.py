"""Standalone inference for the frozen supported artifacts; no study imports."""
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d,minimum_filter1d,maximum_filter1d
from scipy.signal import butter,sosfiltfilt
from numpy.lib.stride_tricks import sliding_window_view
RATE=25
ACC=[0,1,2,3,4,5,9,10]


def spans(mask):
    import numpy as np
    edge=np.diff(np.r_[False,mask,False].astype(np.int8))
    return list(zip(np.flatnonzero(edge==1).tolist(),np.flatnonzero(edge==-1).tolist()))

def bounds(valid):
    import numpy as np
    lo=np.zeros(len(valid),np.int32);hi=lo.copy()
    for a,b in spans(valid):lo[a:b]=a;hi[a:b]=b
    return lo,hi

def gravity_features(values):
    """Retain raw axes and low-frequency orientation, rather than removing gravity."""
    x = np.asarray(values, dtype=np.float32)
    if len(x) > 24:
        gravity = sosfiltfilt(butter(2, 0.3, fs=RATE, output="sos"), x[:, :3], axis=0)
    else:
        gravity = uniform_filter1d(x[:, :3], size=min(13, len(x)), axis=0, mode="nearest")
    dynamic = x[:, :3] - gravity
    return np.concatenate(
        (
            x,
            gravity,
            dynamic,
            np.linalg.norm(x[:, :3], axis=1)[:, None],
            np.linalg.norm(x[:, 3:], axis=1)[:, None],
        ),
        axis=1,
    ).astype(np.float32)

def representation(x,valid):
    z=np.zeros((len(x),11),np.float32)
    for a,b in spans(valid):
        g=gravity_features(x[a:b]);gravity=g[:,6:9];unit=gravity/np.maximum(np.linalg.norm(gravity,axis=1,keepdims=True),1e-6)
        d=g[:,9:12]/9.80665;w=g[:,3:6];u=np.clip(np.abs(unit[:,1]),0,1)
        vertical=np.sum(d*unit,axis=1);horizontal=np.sqrt(np.maximum(0,np.sum(d*d,axis=1)-vertical**2))
        wv=np.sum(w*unit,axis=1);wh=np.sqrt(np.maximum(0,np.sum(w*w,axis=1)-wv**2))
        orient=np.linalg.norm(np.diff(unit,axis=0,prepend=unit[:1]),axis=1)*2.5
        jerk=np.linalg.norm(np.diff(d,axis=0,prepend=d[:1]),axis=1)
        z[a:b]=np.column_stack([u,np.sqrt(np.maximum(0,1-u*u)),np.arccos(u)/(np.pi/2),np.linalg.norm(d,axis=1),vertical,horizontal,g[:,13],wv,wh,orient,jerk])
    assert np.isfinite(z).all();return z

def stats(z,n):
    mean=uniform_filter1d(z,n,axis=0,mode='nearest');var=np.maximum(0,uniform_filter1d(z*z,n,axis=0,mode='nearest')-mean*mean)
    return np.column_stack([z,mean,np.sqrt(var),minimum_filter1d(z,n,axis=0,mode='nearest'),maximum_filter1d(z,n,axis=0,mode='nearest'),uniform_filter1d(np.abs(np.diff(z,axis=0,prepend=z[:1])),n,axis=0,mode='nearest')])

def feature_scale(acc_only):
    width=8 if acc_only else 11
    return np.r_[np.tile(np.r_[np.zeros(3,bool),np.ones(width-3,bool)],12),np.zeros(10 if acc_only else 15,bool)].astype(bool)

def feature_matrix(z,valid,acc_only=False):
    columns=ACC if acc_only else list(range(11));periodic_columns=[3,4] if acc_only else [3,4,6]
    f=np.zeros((len(z),len(feature_scale(acc_only))),np.float32)
    for a,b in spans(valid):
        part=z[a:b,columns]
        f[a:b]=np.column_stack([stats(part,25),stats(part,101),*[periodic(z[a:b,c]) for c in periodic_columns]])
    assert np.isfinite(f).all();return f

def periodic(x):
    windows=sliding_window_view(np.pad(x,(50,50),mode='edge'),101)
    out=np.zeros((len(x),5),np.float32);freq=np.fft.rfftfreq(128,1/25);band=(freq>=.5)&(freq<=4.);taper=np.hanning(101);lags=np.arange(7,63)
    for a in range(0,len(x),1024):
        z=windows[a:a+1024].astype(np.float64);z-=z.mean(1,keepdims=True);energy=np.sum(z*z,axis=1)
        ac=np.fft.irfft(np.abs(np.fft.rfft(z,n=256,axis=1))**2,n=256,axis=1)[:,lags]
        ac=np.clip(ac/np.maximum(energy[:,None],1e-12),-1,1);best=ac.argmax(1)
        power=np.abs(np.fft.rfft(z*taper,n=128,axis=1))**2;bp=power[:,band];peak=bp.argmax(1)
        values=np.column_stack([ac[np.arange(len(z)),best],25/lags[best],bp.sum(1)/np.maximum(power.sum(1),1e-12),bp[np.arange(len(z)),peak]/np.maximum(bp.sum(1),1e-12),freq[band][peak]])
        values[energy<1e-6]=0;out[a:a+len(z)]=values
    return out

def halo(arm):return 62 if arm=='E' else 254

class SmallTCN(torch.nn.Module):
    def __init__(self,arm):
        super().__init__();self.input=torch.nn.Conv1d(8,64,1);ds=[1,2,4,4,4,8,8] if arm=='E' else [1,2,4,8,16,32,64]
        self.blocks=torch.nn.ModuleList([torch.nn.Conv1d(64,64,5,dilation=d,padding=2*d) for d in ds]);self.head=torch.nn.Conv1d(64,4,1)
    def forward(self,x):
        x=torch.nn.functional.gelu(self.input(x))
        for layer in self.blocks:x=x+torch.nn.functional.gelu(layer(x))
        return self.head(x)

class ConvBiGRU(torch.nn.Module):
    def __init__(self):
        super().__init__();self.convs=torch.nn.ModuleList([torch.nn.Conv1d(8 if i==0 else 64,64,5,padding=2) for i in range(4)]);self.recurrent=torch.nn.GRU(64,64,num_layers=2,bidirectional=True,batch_first=True);self.head=torch.nn.Linear(128,4)
    def forward(self,x):
        for conv in self.convs:x=torch.nn.functional.gelu(conv(x))
        x,_=self.recurrent(x.transpose(1,2));return self.head(x).transpose(1,2)

def build(arm):return ConvBiGRU() if arm=='G' else SmallTCN(arm)

def neural_predict(net,r,z,arm,device='cuda',tile=None):
    h=halo(arm);tile=125 if arm=='G' else (512 if tile is None else tile);p=np.zeros((len(z),6),np.float32);net.eval();pending=[]
    def consume(items):
        length=max(len(x[2]) for x in items);batch=np.stack([np.pad(x[2],((0,length-len(x[2])),(0,0)),mode='edge') for x in items]);out=net(torch.from_numpy(batch).to(device).transpose(1,2)).transpose(1,2).float().softmax(-1).cpu().numpy()
        for i,(a,b,_) in enumerate(items):p[a:b,:4]=out[i,h:h+b-a]
    with torch.inference_mode():
        for a,b in spans(r['valid']):
            for start in range(a,b,tile):
                end=min(start+tile,b);outer_end=start+tile if arm=='G' else end;ids=np.clip(np.arange(start-h,outer_end+h),a,b-1);pending.append((start,end,z[ids]))
                if len(pending)==16:consume(pending);pending=[]
        if pending:consume(pending)
    p[~r['valid']]=1/6;return p

def load(path, metadata, threads):
    torch.set_num_threads(threads)
    if metadata['architecture']=='rf':
        import joblib
        bundle=joblib.load(path)
        models=bundle['models']
        for model in models:model.n_jobs=threads
    else:
        bundle=torch.load(path,map_location='cpu',weights_only=True)
        models=[]
        for i,state in enumerate(bundle['states']):
            arm=metadata.get('members',[{}]*len(bundle['states']))[i].get('network_arm',metadata.get('network_arm','E'))
            net=build(arm);net.load_state_dict(state,strict=True);net.eval();models.append((net,arm))
    if len(models)!=int(metadata['member_count']):raise ValueError('ensemble member mismatch')
    if metadata['architecture']=='locomotion_posture_ensemble':
        from pathlib import Path
        import hashlib
        import xgboost as xgb
        binaries=[]
        for item in metadata['binary_members']:
            p=Path(path).parent/item['file']
            if p.parent.resolve()!=Path(path).parent.resolve():raise ValueError('invalid binary artifact path')
            with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
            if digest!=item['sha256']:raise ValueError('binary model checksum mismatch')
            booster=xgb.Booster();booster.load_model(p);booster.set_param({'device':'cpu','nthread':threads});binaries.append(booster)
        if len(binaries)!=metadata['binary_member_count']:raise ValueError('binary member mismatch')
        for recipe in metadata['hierarchy_recipes']:
            if not recipe['posture_groups'] or not recipe['binary_indices']:raise ValueError('empty hierarchy')
            if any(not group or any(i<0 or i>=len(models) for i in group) for group in recipe['posture_groups']):raise ValueError('invalid posture group')
            if any(i<0 or i>=len(binaries) for i in recipe['binary_indices']):raise ValueError('invalid binary index')
        return {'neural':models,'binary':binaries}
    return models

def predict(models, values, valid, metadata):
    z=representation(values,valid);p=np.zeros((len(values),4),np.float32)
    if metadata['architecture']=='locomotion_posture_ensemble':
        cols=np.r_[np.array([i for i in range(96) if i%8>=3]),np.arange(96,106)].astype(int)
        from collections import Counter
        features=feature_matrix(z,valid,True)[:,cols].copy()
        group_use=Counter(tuple(ids) for recipe in metadata['hierarchy_recipes'] for ids in recipe['posture_groups'])
        binary_use=Counter(i for recipe in metadata['hierarchy_recipes'] for i in recipe['binary_indices'])
        posture_cache={};walk_cache={}
        def posture(ids):
            key=tuple(ids)
            if key in posture_cache:return posture_cache[key]
            mean=np.zeros((len(values),6),np.float32)
            for i in ids:
                net,arm=models['neural'][i]
                mean+=neural_predict(net,{'valid':valid},z[:,ACC].copy(),arm,device='cpu')/len(ids)
            static=np.maximum(mean[:,:3],1e-8);static=static/static.sum(1,keepdims=True)
            if group_use[key]>1:posture_cache[key]=static
            return static
        def walking(i):
            if i not in walk_cache:
                value=models['binary'][i].inplace_predict(features)
                if binary_use[i]>1:walk_cache[i]=value
                return value
            return walk_cache[i]
        for recipe in metadata['hierarchy_recipes']:
            static_groups=[posture(ids) for ids in recipe['posture_groups']]
            group=np.zeros((len(values),4),np.float32)
            for i in recipe['binary_indices']:
                walk=walking(i)
                for static in static_groups:
                    q=np.zeros((len(values),4),np.float32);q[:,:3]=static*(1-walk[:,None]);q[:,3]=walk;group+=q
            group/=len(recipe['binary_indices'])*len(static_groups)
            p+=group
        p/=len(metadata['hierarchy_recipes'])
    elif metadata['architecture']=='rf':
        f=feature_matrix(z,valid,True)
        for model in models:
            for a in range(0,len(f),16384):p[a:a+16384]+=model.predict_proba(f[a:a+16384])/len(models)
    else:
        z=z[:,ACC].copy()
        for net,arm in models:p+=neural_predict(net,{'valid':valid},z,arm,device='cpu')[:,:4]/len(models)
    if 'output_groups' in metadata:
        p=np.column_stack([p[:,group].sum(axis=1) for group in metadata['output_groups']]).astype(np.float32)
    p[~valid]=1/p.shape[1]
    return p

