"""K scan for the prototype table: grasp/hold stage groups, descriptor B."""
import json, glob, os, itertools
import numpy as np

FRUIT_DIR="/home/robot/Downloads/DamageVLA_training_post_process_20260911"
CART_DIR ="/home/robot/Downloads/DamageVLA_training_post_process_20260916_carton"
SG={"grasp":"grasp","lift":"hold","translate":"hold","place":"hold"}
ZERO_N=15
OUTLIERS={"pi0_train_20260916_015938_carton_02_full"}

def load(p): return [json.loads(l) for l in open(p) if l.strip()]
def feats(ep,cond):
    rows=load(ep)
    zi=[i for i,r in enumerate(rows) if r.get("stage")=="prepare"
        and max(abs(float(x)) for x in (r.get("cmd_speed_l") or [0]*6))<1e-6][:ZERO_N]
    if len(zi)<5: return None
    L=np.array([r["tactile_voltage_signals"]["left_data"] for r in rows],float)
    R=np.array([r["tactile_voltage_signals"]["right_data"] for r in rows],float)
    bL,bR=L[zi].mean(0),R[zi].mean(0); noise=float(np.std(np.concatenate([L[zi]-bL,R[zi]-bR])))
    Lz,Rz=L-bL,R-bR; grip=0.5*(Lz.sum(1)+Rz.sum(1))
    w=np.array([float(r["gripper_width"]) for r in rows])
    sg=np.array([SG.get(r.get("stage"),"") for r in rows]); thr=max(5*noise,1e-4); ridx=np.arange(25)//5
    o={"condition":cond,"episode":os.path.basename(os.path.dirname(ep)),"noise":noise}
    for s in ("grasp","hold"):
        m=sg==s
        if m.sum()==0: continue
        o[f"mu_{s}"]=float(grip[m].mean()); o[f"sd_{s}"]=float(grip[m].std())
        o[f"area_{s}"]=float((((Lz[m]>thr).sum(1)+(Rz[m]>thr).sum(1))/2).mean())
        v=Lz[m].clip(min=0)+Rz[m].clip(min=0); t=v.sum(1); ok=t>thr
        o[f"cop_{s}"]=float((v[ok]@ridx/t[ok]).mean()) if ok.sum() else np.nan
    g=sg=="grasp"
    if g.sum()>=4 and np.ptp(w[g].max()-w[g])>1e-4:
        x=w[g].max()-w[g]; A=np.vstack([x,np.ones_like(x)]).T
        o["stiffness"]=float(np.linalg.lstsq(A,grip[g],rcond=None)[0][0])
    return o

recs=[]
for ep in sorted(glob.glob(FRUIT_DIR+"/*/observations.jsonl")):
    p=load(ep)[0].get("prompt","").lower()
    c=next((k for k in ("apple","kiwi","tomato","cucumber") if k in p),"unknown")
    f=feats(ep,c);  recs.append(f) if f else None
for ep in sorted(glob.glob(CART_DIR+"/*/observations.jsonl")):
    n=os.path.basename(os.path.dirname(ep))
    if n in OUTLIERS: continue
    f=feats(ep,"carton_empty" if "empty" in n else "carton_full"); recs.append(f) if f else None
NOISE_FLOOR=float(np.median([r["noise"] for r in recs]))*np.sqrt(50)
print(f"episodes {len(recs)}, grip noise floor ~{NOISE_FLOOR:.4f} (sigma clamp)")

def descriptors(rs):
    conds=sorted({r["condition"] for r in rs}); keys=[]; D=[]
    for c in conds:
        e=[r for r in rs if r["condition"]==c]
        st=np.nanmean([r.get("stiffness",np.nan) for r in e])
        for s in ("grasp","hold"):
            mu=np.nanmean([r.get(f"mu_{s}",np.nan) for r in e])
            sd=max(np.nanmean([r.get(f"sd_{s}",np.nan) for r in e]), NOISE_FLOOR)
            ar=np.nanmean([r.get(f"area_{s}",np.nan) for r in e]); cp=np.nanmean([r.get(f"cop_{s}",np.nan) for r in e])
            keys.append((c,s)); D.append([mu,np.log(sd),ar,cp,st])
    return keys,np.array(D)

def robust(x):
    med=np.median(x,0); iqr=np.percentile(x,75,0)-np.percentile(x,25,0); return (x-med)/(iqr+1e-6)
def kmeans(x,k,seed=0,n_init=200):
    rng=np.random.default_rng(seed); best=None
    for _ in range(n_init):
        C=x[rng.choice(len(x),k,replace=False)]
        for _ in range(200):
            a=((x[:,None]-C[None])**2).sum(-1).argmin(1)
            Cn=np.array([x[a==j].mean(0) if (a==j).any() else C[j] for j in range(k)])
            if np.allclose(Cn,C): break
            C=Cn
        i=((x-C[a])**2).sum()
        if best is None or i<best[0]: best=(i,a.copy())
    return best[1]
def silhouette(x,a):
    n=len(x); Dm=np.sqrt(((x[:,None]-x[None])**2).sum(-1)); s=[]
    for i in range(n):
        same=(a==a[i]); same[i]=False
        if same.sum()==0: s.append(0.0); continue
        ai=Dm[i,same].mean()
        bi=min(Dm[i,a==j].mean() for j in set(a) if j!=a[i])
        s.append((bi-ai)/max(ai,bi))
    return float(np.mean(s))
def ari(a,b):
    n=len(a); ca=sorted(set(a)); cb=sorted(set(b))
    M=np.array([[np.sum((a==i)&(b==j)) for j in cb] for i in ca])
    su=lambda v: (v*(v-1)/2).sum()
    idx=su(M); ea=su(M.sum(1)); eb=su(M.sum(0)); tot=n*(n-1)/2
    exp=ea*eb/tot; mx=(ea+eb)/2
    return (idx-exp)/(mx-exp) if mx!=exp else 1.0

keys,D=descriptors(recs); X=robust(D)
print(f"groups: {len(keys)} ({len({k[0] for k in keys})} conditions x 2 stage groups)\n")
print(f"{'K':>3}{'silhouette':>12}{'LOO stability':>15}{'混条件簇':>12}{'簇大小':>22}")
res={}
for K in range(2,11):
    a=kmeans(X,K)
    sil=silhouette(X,a)
    aris=[]
    for r in recs:                                   # leave-one-episode-out
        sub=[q for q in recs if q["episode"]!=r["episode"]]
        _,D2=descriptors(sub)
        aris.append(ari(a,kmeans(robust(D2),K)))
    mix=sum(len({keys[i][0] for i in range(len(a)) if a[i]==j})>1 for j in set(a))
    sizes=np.bincount(a,minlength=K)
    res[K]=(sil,np.mean(aris),mix)
    print(f"{K:>3}{sil:>12.3f}{np.mean(aris):>15.3f}{mix:>12}   {str(sizes.tolist()):>20}")

best=max(res,key=lambda k:(res[k][1],res[k][0]))
print(f"\n按稳定性优先: K={best}  (silhouette {res[best][0]:.3f}, LOO-ARI {res[best][1]:.3f})")
for K in (4,5,6):
    a=kmeans(X,K); print(f"\n--- K={K} 分组 ---")
    for j in sorted(set(a)):
        print("   ", ", ".join(f"{keys[i][0]}/{keys[i][1]}" for i in range(len(a)) if a[i]==j))
