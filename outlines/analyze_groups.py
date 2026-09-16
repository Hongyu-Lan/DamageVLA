"""DraftVLA 组级分析：逐条件统计、方差分解、K 扫描、异常 episode 标记。

用法（在有 post-process 目录的机器上）：
    python3 analyze_groups.py <fruit_dir> [<fruit_dir> ...] --carton <carton_dir> [...]

输出：
  1) 每个条件的 mu/sigma/刚度/接触面积（按 grasp 和 hold 两个阶段组）
  2) 方差分解 F_condition / F_stage，用来判断哪些特征进原型描述子
  3) K 扫描（silhouette + 留一稳定性 ARI），用来定原型数 K
  4) 按排除规则标记的 episode
"""
import argparse, json, glob, os
import numpy as np

FRUITS = ("apple", "kiwi", "tomato", "cucumber", "potato", "banana")
SG = {"grasp": "grasp", "lift": "hold", "translate": "hold", "place": "hold"}
ZERO_N = 15


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def episode_features(ep, cond):
    rows = load(ep)
    zi = [i for i, r in enumerate(rows) if r.get("stage") == "prepare"
          and max(abs(float(x)) for x in (r.get("cmd_speed_l") or [0] * 6)) < 1e-6][:ZERO_N]
    if len(zi) < 5:
        return None
    L = np.array([r["tactile_voltage_signals"]["left_data"] for r in rows], float)
    R = np.array([r["tactile_voltage_signals"]["right_data"] for r in rows], float)
    bL, bR = L[zi].mean(0), R[zi].mean(0)
    noise = float(np.std(np.concatenate([L[zi] - bL, R[zi] - bR])))
    Lz, Rz = L - bL, R - bR
    grip = 0.5 * (Lz.sum(1) + Rz.sum(1))                 # 目标标量：两指归零电压总和的均值
    w = np.array([float(r["gripper_width"]) for r in rows])
    sg = np.array([SG.get(r.get("stage"), "") for r in rows])
    thr = max(5 * noise, 1e-4)
    ridx = np.arange(25) // 5
    o = {"condition": cond, "episode": os.path.basename(os.path.dirname(ep)), "noise": noise}
    for s in ("grasp", "hold"):
        m = sg == s
        if m.sum() == 0:
            continue
        o[f"mu_{s}"] = float(grip[m].mean())
        o[f"sd_{s}"] = float(grip[m].std())
        o[f"area_{s}"] = float((((Lz[m] > thr).sum(1) + (Rz[m] > thr).sum(1)) / 2).mean())
        v = Lz[m].clip(min=0) + Rz[m].clip(min=0)
        t = v.sum(1); ok = t > thr
        o[f"cop_{s}"] = float((v[ok] @ ridx / t[ok]).mean()) if ok.sum() else np.nan
    g = sg == "grasp"
    if g.sum() >= 4 and np.ptp(w[g].max() - w[g]) > 1e-4:
        x = w[g].max() - w[g]
        A = np.vstack([x, np.ones_like(x)]).T
        o["stiffness"] = float(np.linalg.lstsq(A, grip[g], rcond=None)[0][0])   # dF/d(闭合量)
        o["engaged"] = int(max((Lz[g].max(0) > thr).sum(), (Rz[g].max(0) > thr).sum()))
        o["w_final"] = float(w[g].min())
    return o


def robust(x):
    med = np.median(x, 0)
    iqr = np.percentile(x, 75, 0) - np.percentile(x, 25, 0)
    return (x - med) / (iqr + 1e-6)


def kmeans(x, k, seed=0, n_init=200):
    rng = np.random.default_rng(seed); best = None
    for _ in range(n_init):
        C = x[rng.choice(len(x), k, replace=False)]
        for _ in range(200):
            a = ((x[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            Cn = np.array([x[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
            if np.allclose(Cn, C):
                break
            C = Cn
        inertia = ((x - C[a]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, a.copy())
    return best[1]


def silhouette(x, a):
    D = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1)); s = []
    for i in range(len(x)):
        m = (a == a[i]).copy(); m[i] = False
        if m.sum() == 0:
            s.append(0.0); continue
        ai = D[i, m].mean()
        bi = min(D[i, a == j].mean() for j in set(a) if j != a[i])
        s.append((bi - ai) / max(ai, bi))
    return float(np.mean(s))


def ari(a, b):
    n = len(a); ca = sorted(set(a)); cb = sorted(set(b))
    M = np.array([[np.sum((a == i) & (b == j)) for j in cb] for i in ca])
    su = lambda v: (v * (v - 1) / 2).sum()
    idx, ea, eb = su(M), su(M.sum(1)), su(M.sum(0)); tot = n * (n - 1) / 2
    exp = ea * eb / tot; mx = (ea + eb) / 2
    return (idx - exp) / (mx - exp) if mx != exp else 1.0


def descriptors(recs, noise_floor):
    conds = sorted({r["condition"] for r in recs}); keys = []; D = []
    for c in conds:
        e = [r for r in recs if r["condition"] == c]
        st = np.nanmean([r.get("stiffness", np.nan) for r in e])
        for s in ("grasp", "hold"):
            mu = np.nanmean([r.get(f"mu_{s}", np.nan) for r in e])
            sd = max(np.nanmean([r.get(f"sd_{s}", np.nan) for r in e]), noise_floor)
            ar = np.nanmean([r.get(f"area_{s}", np.nan) for r in e])
            cp = np.nanmean([r.get(f"cop_{s}", np.nan) for r in e])
            keys.append((c, s)); D.append([mu, np.log(sd), ar, cp, st])
    return keys, np.array(D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fruit_dirs", nargs="*")
    ap.add_argument("--carton", action="append", default=[])
    args = ap.parse_args()

    recs = []
    for d in args.fruit_dirs:
        for ep in sorted(glob.glob(d + "/*/observations.jsonl")):
            p = load(ep)[0].get("prompt", "").lower()
            c = next((k for k in FRUITS if k in p), "unknown")
            f = episode_features(ep, c)
            if f: recs.append(f)
    for d in args.carton:
        for ep in sorted(glob.glob(d + "/*/observations.jsonl")):
            n = os.path.basename(os.path.dirname(ep))
            f = episode_features(ep, "carton_empty" if "empty" in n else "carton_full")
            if f: recs.append(f)

    conds = sorted({r["condition"] for r in recs})
    NF = float(np.median([r["noise"] for r in recs])) * np.sqrt(50)
    print(f"episodes {len(recs)} | conditions {len(conds)} | grip noise floor {NF:.4f} (sigma 下限)\n")

    print(f"{'condition':<14}{'n':>3}{'mu_grasp':>17}{'mu_hold':>17}{'stiffness':>14}{'area':>8}")
    for c in conds:
        rs = [r for r in recs if r["condition"] == c]
        f = lambda k: np.array([r.get(k, np.nan) for r in rs], float)
        g, h, st, ar = f("mu_grasp"), f("mu_hold"), f("stiffness"), f("area_grasp")
        print(f"{c:<14}{len(rs):>3}{np.nanmean(g):>10.3f}+-{np.nanstd(g):<6.3f}"
              f"{np.nanmean(h):>10.3f}+-{np.nanstd(h):<6.3f}{np.nanmean(st):>9.1f}{np.nanmean(ar):>8.1f}")

    print("\n=== 方差分解 (F = 效应方差 / 组内 episode 方差) ===")
    print(f"{'feature':<10}{'F_condition':>13}{'F_stage':>10}")
    for nm, tpl in [("mu", "mu_{s}"), ("log sd", "sd_{s}"), ("area", "area_{s}"), ("cop", "cop_{s}")]:
        M = {}; W = []
        for c in conds:
            rs = [r for r in recs if r["condition"] == c]
            for s in ("grasp", "hold"):
                v = np.array([r.get(tpl.format(s=s), np.nan) for r in rs], float); v = v[~np.isnan(v)]
                if len(v) >= 2:
                    M[(c, s)] = v.mean(); W.append(v.var(ddof=1))
        w = np.mean(W)
        fc = np.mean([np.var([M[(c, s)] for c in conds if (c, s) in M], ddof=1) for s in ("grasp", "hold")]) / w
        fs = np.mean([np.var([M[(c, s)] for s in ("grasp", "hold") if (c, s) in M], ddof=1) for c in conds]) / w
        print(f"{nm:<10}{fc:>13.2f}{fs:>10.2f}")

    keys, D = descriptors(recs, NF); X = robust(D)
    print(f"\n=== K 扫描 ({len(keys)} 组) ===")
    print(f"{'K':>3}{'silhouette':>12}{'LOO-ARI':>10}{'mixed-cond':>12}   sizes")
    for K in range(2, 11):
        a = kmeans(X, K); s = silhouette(X, a)
        st = np.mean([ari(a, kmeans(robust(descriptors([q for q in recs if q["episode"] != r["episode"]], NF)[1]), K))
                      for r in recs])
        mix = sum(len({keys[i][0] for i in range(len(a)) if a[i] == j}) > 1 for j in set(a))
        print(f"{K:>3}{s:>12.3f}{st:>10.3f}{mix:>12}   {np.bincount(a, minlength=K).tolist()}")
    for K in (4, 5, 6):
        a = kmeans(X, K); print(f"\n--- K={K} ---")
        for j in sorted(set(a)):
            print("   ", ", ".join(f"{keys[i][0]}/{keys[i][1]}" for i in range(len(a)) if a[i] == j))

    print("\n=== 按排除规则标记的 episode ===")
    for c in conds:
        rs = [r for r in recs if r["condition"] == c]
        v = np.array([r.get("mu_grasp", np.nan) for r in rs], float)
        med = np.median(v); mad = np.median(np.abs(v - med))
        for r in sorted(rs, key=lambda r: r.get("mu_grasp", 0)):
            tags = []
            if mad > 0 and abs(r.get("mu_grasp", med) - med) > 3 * 1.4826 * mad:
                tags.append("3xMAD 离群(抓过头)")
            if r.get("engaged", 9) < 3 and c != "carton_empty":
                tags.append(f"接触 taxel 仅 {r.get('engaged')} 个")
            if c not in ("carton_empty",) and r.get("mu_hold", 9) < 10 * NF:
                tags.append("持握力近噪声(疑似没夹到)")
            if tags:
                print(f"  {c:<14}{r['episode']:<42}mu={r.get('mu_grasp', float('nan')):6.3f} -> {', '.join(tags)}")


if __name__ == "__main__":
    main()
