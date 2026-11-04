"""C2/C3: training-fitted, physical-unit inference layers. No held-out data access."""
import numpy as np
from . import baselines


def fit(P, config):
    ncu=P.nc+P.ne;cols=P.info['columns'];mu=np.array(P.info['mu']);sd=np.array(P.info['sd'])
    raw=np.concatenate([x[normal] for x,normal in P.train_raw]).astype('float64')
    lo,hi=raw.min(0),raw.max(0)
    transitions=[(x[:-1][normal[:-1]&normal[1:]],x[1:][normal[:-1]&normal[1:]]) for x,normal in P.train_raw]
    before=np.concatenate([a for a,b in transitions]).astype('float64')
    after=np.concatenate([b for a,b in transitions]).astype('float64')
    rates=np.quantile(np.abs(after[:,ncu:]-before[:,ncu:]),.999,axis=0)
    valves=[]
    for j,name in enumerate(P.resp_names):
        if name in config['exclude']:continue
        if name.endswith('Z') and name[:-1]+'D' in cols[:ncu] and ('CV' in name):
            u=cols.index(name[:-1]+'D');dy=after[:,ncu+j]-before[:,ncu+j]
            toward=(after[:,u]-before[:,ncu+j])*dy>=0
            moving=np.abs(dy)>0
            def quant(mask):return float(np.quantile(np.abs(dy[mask]),.999)) if mask.any() else 0.
            toward_rate=quant(toward&moving);away_rate=min(toward_rate,quant(~toward&moving))
            valves.append({'response':j,'input':u,'toward':toward_rate,'away':away_rate})
    motors=[]
    if P.name=='qappd':
        for k in range(2):
            v=cols.index(f'Voltage{k}');i=cols.index(f'Current{k}');w=cols.index(f'MotorSpeed{k}')
            phi=np.column_stack([raw[:,i],raw[:,w],np.ones(len(raw))])
            coef=np.linalg.lstsq(phi,raw[:,v],rcond=None)[0]
            if coef[0]<=0 or coef[1]<=0:raise ValueError(f'QAPPD motor {k}: unexpected resistance/back-EMF signs {coef}')
            motors.append({'voltage':v,'current':i-ncu,'speed':w-ncu,'R':float(coef[0]),'k':float(coef[1]),'c':float(coef[2])})
    result={'mu':mu.tolist(),'sd':sd.tolist(),'lo':lo.tolist(),'hi':hi.tolist(),'rates':rates.tolist(),
            'valves':valves,'motors':motors,'config':config,'ncu':ncu,'nc':P.nc,'W':P.W}
    # Audit local range/rate rules on true adjacent training samples, not accumulated generated paths.
    actual=after[:,ncu:];clipped=local_limit(result,before[:,ncu:],actual,after[:,:ncu],limits_only=True)
    changed=np.abs(actual-clipped)>1e-5
    result['training_limit_change_per_tag']=dict(zip(P.resp_names,changed.mean(0).tolist()))
    if changed.mean(0).max()>=.05:raise AssertionError('C2 range/rate rules change >=5% of a real training tag')
    return result


def local_limit(c,prev,y,u,limits_only=False):
    y=y.copy();n=c['ncu'];lo=np.array(c['lo'][n:]);hi=np.array(c['hi'][n:]);rate=np.array(c['rates'])
    valve={p['response']:p for p in c['valves']}
    for j in range(y.shape[-1]):
        if j in valve:
            y[...,j]=np.clip(y[...,j],*c['config']['valve_range'])
        elif c['config']['clip_nonvalve']:
            y[...,j]=np.clip(y[...,j],lo[j],hi[j])
        if j in valve and not limits_only:
            p=valve[j];dy=y[...,j]-prev[...,j]
            toward=(u[...,p['input']]-prev[...,j])*dy>=0
            cap=np.where(toward,p['toward'],p['away'])
        else:cap=rate[j]
        y[...,j]=prev[...,j]+np.clip(y[...,j]-prev[...,j],-cap,cap)
    if not limits_only:
        # Algebraic relation takes precedence over generic rate/range clips for current.
        for p in c['motors']:
            y[...,p['current']]=(u[...,p['voltage']]-p['k']*y[...,p['speed']]-p['c'])/p['R']
    return y


def coverage(c,X):
    nc=c['nc'];lo=np.array(c['lo'][:nc]);hi=np.array(c['hi'][:nc])
    u=X[:,c['W']:,:nc]*np.array(c['sd'][:nc])+np.array(c['mu'][:nc])
    # Float32 z-score round trips can differ from physical extrema by a few ULPs.
    tol=1e-6*np.maximum(1,hi-lo)
    excess=np.maximum(np.maximum(lo-u,u-hi)-tol,0)
    return np.clip((excess/(c['config']['c3_frac']*np.maximum(hi-lo,1e-8))).max(-1),0,1)


def apply(c,grey,X,pred,variant='Full'):
    if variant=='A5':return pred.copy()
    n=c['ncu'];mu=np.array(c['mu']);sd=np.array(c['sd']);W=c['W']
    y=pred.astype('float64').copy()
    if variant not in ('A2','A5'):
        w=coverage(c,X)[...,None]
        y=(1-w)*y+w*baselines.grey_sim(grey,X,W,n)
    if variant=='A1':return y.astype('float32')
    physical=y*sd[n:]+mu[n:];prev=X[:,W-1,n:]*sd[n:]+mu[n:]
    u=X[:,W:,:n]*sd[:n]+mu[:n]
    for t in range(y.shape[1]):
        prev=local_limit(c,prev,physical[:,t],u[:,t],limits_only=variant=='A4')
        physical[:,t]=prev
    return ((physical-mu[n:])/sd[n:]).astype('float32')
