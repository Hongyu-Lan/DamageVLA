import json, glob, os, collections
import numpy as np
exec(open("descriptor_analysis.py").read().split("recs = []")[0])

def ep_feats(ep, cond):
    rows = load(ep); zi = zero_window(rows)
    if len(zi) < 5: return None
    L = np.array([r["tactile_voltage_signals"]["left_data"] for r in rows], float)
    R = np.array([r["tactile_voltage_signals"]["right_data"] for r in rows], float)
    bL,bR = L[zi].mean(0), R[zi].mean(0); noise = float(np.std(np.concatenate([L[zi]-bL,R[zi]-bR])))
    Lz,Rz = L-bL, R-bR; grip = 0.5*(Lz.sum(1)+Rz.sum(1))
    w = np.array([float(r["gripper_width"]) for r in rows]); stage = np.array([r.get("stage") for r in rows])
    thr = max(5*noise,1e-4); rowsidx = np.arange(25)//5
    out = {"condition":cond, "episode":os.path.basename(os.path.dirname(ep))}
    for s in STAGES:                                   # per-stage features
        m = stage==s
        if m.sum()==0: continue
        out[f"mu_{s}"]=float(grip[m].mean()); out[f"sd_{s}"]=float(grip[m].std())
        out[f"area_{s}"]=float((((Lz[m]>thr).sum(1)+(Rz[m]>thr).sum(1))/2).mean())
        v = Lz[m].clip(min=0)+Rz[m].clip(min=0); tot=v.sum(1); ok=tot>thr
        out[f"cop_{s}"]=float((v[ok]@rowsidx/tot[ok]).mean()) if ok.sum() else np.nan
    g = stage=="grasp"                                  # condition-level: stiffness from closing
    if g.sum()>=4 and np.ptp(w[g].max()-w[g])>1e-4:
        x=w[g].max()-w[g]; y=grip[g]; A=np.vstack([x,np.ones_like(x)]).T
        co=np.linalg.lstsq(A,y,rcond=None)[0]; out["stiffness"]=float(co[0])
    return out

recs=[]
for ep in sorted(glob.glob(FRUIT_DIR+"/*/observations.jsonl")):
    p=load(ep)[0].get("prompt","").lower()
    c=next((k for k in ("apple","kiwi","tomato","cucumber","potato","banana") if k in p),"unknown")
    f=ep_feats(ep,c);  recs.append(f) if f else None
for ep in sorted(glob.glob(CART_DIR+"/*/observations.jsonl")):
    n=os.path.basename(os.path.dirname(ep)); f=ep_feats(ep,"carton_empty" if "empty" in n else "carton_full")
    recs.append(f) if f else None

OUTLIERS = {"pi0_train_20260916_015938_carton_02_full",          # 5-10x the other full cartons
            "pi0_train_20260916_023258_carton_01_empty",          # no contact signal at all
            "pi0_train_20260916_023845_carton_01_empty",
            "pi0_train_20260916_024043_carton_01_empty"}
def fratio(rs, key):
    conds = sorted({r["condition"] for r in rs}); means=[]; wi=[]
    for c in conds:
        v=np.array([r[key] for r in rs if r["condition"]==c and key in r],float); v=v[~np.isnan(v)]
        if len(v)>=2: means.append(v.mean()); wi.append(v.var(ddof=1))
    return np.var(means,ddof=1)/np.mean(wi) if len(means)>=2 and np.mean(wi)>0 else np.nan

kept=[r for r in recs if r["episode"] not in OUTLIERS]
print(f"episodes: all {len(recs)}, after dropping {len(OUTLIERS)} outliers {len(kept)}")
print("\n=== F-ratio, all episodes vs outliers removed ===")
print(f"{'feature':<14}{'all':>8}{'clean':>8}")
for k in ["mu_grasp","sd_grasp","mu_lift","area_grasp","cop_grasp","stiffness"]:
    print(f"{k:<14}{fratio(recs,k):>8.2f}{fratio(kept,k):>8.2f}")

conds=sorted({r["condition"] for r in kept})
gkeys=[]; rowsD=[]
for c in conds:
    rs=[r for r in kept if r["condition"]==c]
    st=np.nanmean([r.get("stiffness",np.nan) for r in rs])
    for s in STAGES:
        mu=np.nanmean([r.get(f"mu_{s}",np.nan) for r in rs]); sd=np.nanmean([r.get(f"sd_{s}",np.nan) for r in rs])
        ar=np.nanmean([r.get(f"area_{s}",np.nan) for r in rs]); cp=np.nanmean([r.get(f"cop_{s}",np.nan) for r in rs])
        gkeys.append((c,s)); rowsD.append([mu,np.log(max(sd,1e-4)),ar,cp,st])
G=np.array(rowsD)
def robust(x):
    med=np.median(x,0); iqr=np.percentile(x,75,0)-np.percentile(x,25,0); return (x-med)/(iqr+1e-6)
def kmeans(x,k,seed=0,n_init=50):
    rng=np.random.default_rng(seed); best=None
    for _ in range(n_init):
        C=x[rng.choice(len(x),k,replace=False)]
        for _ in range(100):
            a=((x[:,None]-C[None])**2).sum(-1).argmin(1)
            Cn=np.array([x[a==j].mean(0) if (a==j).any() else C[j] for j in range(k)])
            if np.allclose(Cn,C): break
            C=Cn
        i=((x-C[a])**2).sum()
        if best is None or i<best[0]: best=(i,a.copy())
    return best[1]
for name,cols in [("A: [mu, log sd]",[0,1]),
                  ("B: [mu, log sd, area(stage), cop(stage), stiffness(cond)]",[0,1,2,3,4])]:
    a=kmeans(robust(G[:,cols]),4)
    print(f"\n=== clustering {name} ===")
    for j in range(4):
        mem=[f"{c}/{s}" for (c,s),aa in zip(gkeys,a) if aa==j]
        print(f"  cluster {j}: {', '.join(mem) if mem else '(empty)'}")
    bycond=len({j for j in range(4) if len({m.split('/')[0] for m in [f'{c}/{s}' for (c,s),aa in zip(gkeys,a) if aa==j]})>1})
    bystage=sum(len({m.split('/')[1] for m in [f'{c}/{s}' for (c,s),aa in zip(gkeys,a) if aa==j]})>1 for j in range(4))
    print(f"  clusters mixing conditions: {bycond}/4   clusters mixing stages: {bystage}/4")
