"""Fit the same small probe on every arm's frozen features; score it on the held-out episodes.

Input: the features.npz files written by probe_features.py, one per arm. For each arm and each
feature key (z_phy, fused_force_token, force_token_raw, vl_prefix_mean, vl_prefix_last):

  0. equal capacity: standardise on TRAIN frames, then PCA (fit on train) to at most 128 components,
     so a 2048-D token and the 128-D z_phy are probed with the same budget;
  1. a linear Gaussian head  x -> (mu_hat, log sigma_hat), fitted as ridge regression onto the
     frozen group targets (mu_g, log sigma_g) with the ridge strength chosen by episode-grouped
     cross-validation on TRAIN frames (closed form, so it cannot diverge); scored on VAL frames with
     the forward KL  KL(N(mu_g, sigma_g) || N(mu_hat, sigma_hat))  -- PiVLA's L_dist, same floor;
     a mu-only variant  (mu_g - mu_hat)^2 / (2 sigma_g^2)  is reported next to it because the KL is
     dominated by sigma_hat whenever the head is over-confident;
  2. linear classifiers (cross-entropy, L2): condition (6-way here), and carton empty-vs-full on
     (a) frames AFTER the grasp annotation -- contact available -- and (b) frames BEFORE it -- no
     contact, so any accuracy above chance is appearance / instance leakage. (a) minus (b) is what
     contact adds; this is the paper's Q5 probe.

Baseline for the KL: the constant train-set prior (mean of mus, median of sigmas), the same
"predict the global mean" predictor as loss_dist_baseline.

Usage:
  uv run examples/force/probe_train_eval.py \
      --arm full=examples/force/probe_20260921/full/features.npz \
      --arm forcevla=examples/force/probe_20260921/forcevla/features.npz \
      --arm noforce=examples/force/probe_20260921/noforce/features.npz \
      --out examples/force/probe_20260921/probe_results.json
"""

import argparse
import json
import pathlib

import numpy as np
import torch

SIGMA_FLOOR = 0.0051  # draftvla_contact.GRIP_SIGMA_FLOOR: the label sigma floor, also the head's
KL_FLOOR = 1e-4
CARTONS = ("carton_empty", "carton_full")
PCA_DIM = 128
RIDGE_GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def kl_forward(mu_gt, sigma_gt, mu_pred, sigma_pred):
    sigma_gt = np.maximum(sigma_gt, KL_FLOOR)
    sigma_pred = np.maximum(sigma_pred, KL_FLOOR)
    return np.log(sigma_pred / sigma_gt) + (sigma_gt**2 + (mu_gt - mu_pred) ** 2) / (2 * sigma_pred**2) - 0.5


class Projector:
    """Standardise on train, then PCA to <= PCA_DIM (identity when the feature is already small)."""

    def __init__(self, x_train: np.ndarray):
        self.m = x_train.mean(0, keepdims=True)
        self.s = x_train.std(0, keepdims=True) + 1e-6
        z = (x_train - self.m) / self.s
        if z.shape[1] > PCA_DIM:
            # thin SVD on the centred, standardised train matrix
            _, _, vt = np.linalg.svd(z - z.mean(0, keepdims=True), full_matrices=False)
            self.components = vt[:PCA_DIM].T  # [D, k]
        else:
            self.components = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.m) / self.s
        return z @ self.components if self.components is not None else z


def ridge_fit(x, y, lam):
    """Closed-form ridge with intercept. x [n, d], y [n, k]."""
    xb = np.concatenate([x, np.ones((x.shape[0], 1))], 1)
    reg = lam * np.eye(xb.shape[1])
    reg[-1, -1] = 0.0  # do not penalise the intercept
    return np.linalg.solve(xb.T @ xb + reg, xb.T @ y)


def ridge_predict(w, x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], 1) @ w


def fit_gaussian_head(x, mu, sigma, groups, seed=0):
    """Ridge onto [mu, log sigma]; lambda by 5-fold episode-grouped CV on the train KL."""
    y = np.stack([mu, np.log(np.maximum(sigma, SIGMA_FLOOR))], 1)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, min(5, len(uniq)))
    best_lam, best = None, np.inf
    for lam in RIDGE_GRID:
        score = []
        for f in folds:
            va = np.isin(groups, f)
            w = ridge_fit(x[~va], y[~va], lam)
            pred = ridge_predict(w, x[va])
            score.append(kl_forward(mu[va], sigma[va], pred[:, 0], np.exp(pred[:, 1]) + SIGMA_FLOOR).mean())
        s = float(np.mean(score))
        if s < best:
            best, best_lam = s, lam
    return ridge_fit(x, y, best_lam), best_lam, best


def eval_gaussian_head(w, x, mu, sigma):
    pred = ridge_predict(w, x)
    mu_hat, sigma_hat = pred[:, 0], np.exp(pred[:, 1]) + SIGMA_FLOOR
    kl = kl_forward(mu, sigma, mu_hat, sigma_hat)
    kl_mu_only = (mu - mu_hat) ** 2 / (2 * np.maximum(sigma, KL_FLOOR) ** 2)
    return kl, kl_mu_only, mu_hat, sigma_hat


def fit_linear_classifier(x, y, n_classes, *, l2=1e-2, steps=400, seed=0):
    torch.manual_seed(seed)
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    clf = torch.nn.Linear(xt.shape[1], n_classes)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(clf(xt), yt) + l2 * (clf.weight**2).sum()
        loss.backward()
        opt.step()
    return clf


def accuracy(clf, x, y):
    if len(y) == 0:
        return None
    with torch.no_grad():
        pred = clf(torch.as_tensor(x, dtype=torch.float32)).argmax(1).numpy()
    return float((pred == y).mean())


def _mean(v):
    return float(v.mean()) if len(v) else None


def evaluate_arm(name: str, path: pathlib.Path, out_rows: list, log):
    d = np.load(path, allow_pickle=False)
    keys = [str(k) for k in d["feature_keys"]]
    split, cond, sg, ep = d["split"], d["condition"], d["stage_group"], d["episode"]
    mu_t, sig_t = d["mu_target"].astype(np.float64), d["sigma_target"].astype(np.float64)
    labelled = sg != ""
    tr, va = (split == "train") & labelled, (split == "val") & labelled
    conditions = sorted(set(cond.tolist()))
    cond_idx = np.asarray([conditions.index(c) for c in cond])
    is_carton = np.isin(cond, CARTONS)
    post, pre = d["t_from_grasp_start"] >= 0, d["t_from_grasp_start"] < 0
    prior_mu, prior_sigma = float(mu_t[tr].mean()), float(np.median(sig_t[tr]))
    base_kl = kl_forward(mu_t[va], sig_t[va], np.full(va.sum(), prior_mu), np.full(va.sum(), prior_sigma))
    base_mu = (mu_t[va] - prior_mu) ** 2 / (2 * sig_t[va] ** 2)

    log(f"\n=== {name}: {path.name}  train {tr.sum()} / val {va.sum()} labelled frames; conditions {conditions}")
    log(f"  constant-prior baseline on val: KL {base_kl.mean():.3f} (grasp {base_kl[sg[va]=='grasp'].mean():.3f} hold {base_kl[sg[va]=='hold'].mean():.3f} | carton {base_kl[is_carton[va]].mean():.3f} produce {base_kl[~is_carton[va]].mean():.3f})  mu-only {base_mu.mean():.3f}")
    for key in keys:
        x = d[key].astype(np.float64)
        proj = Projector(x[tr])
        x_tr, x_va = proj(x[tr]), proj(x[va])
        w, lam, cv = fit_gaussian_head(x_tr, mu_t[tr], sig_t[tr], ep[tr])
        kl, kl_mu, mu_hat, sigma_hat = eval_gaussian_head(w, x_va, mu_t[va], sig_t[va])
        row = {
            "arm": name, "feature": key, "dim_raw": int(x.shape[1]), "dim_probe": int(x_tr.shape[1]), "ridge_lambda": lam, "cv_train_kl": cv,
            "kl_val": float(kl.mean()), "kl_val_grasp": _mean(kl[sg[va] == "grasp"]), "kl_val_hold": _mean(kl[sg[va] == "hold"]),
            "kl_val_carton": _mean(kl[is_carton[va]]), "kl_val_produce": _mean(kl[~is_carton[va]]),
            "kl_mu_only_val": float(kl_mu.mean()), "kl_mu_only_carton": _mean(kl_mu[is_carton[va]]),
            "kl_baseline_val": float(base_kl.mean()), "kl_baseline_carton": _mean(base_kl[is_carton[va]]), "kl_mu_only_baseline": float(base_mu.mean()),
            "sigma_hat_val_median": float(np.median(sigma_hat)),
        }
        # Q5 probes on the same projected features.
        proj_all = Projector(x[split == "train"])
        xa_tr, xa_va = proj_all(x[split == "train"]), proj_all(x[split == "val"])
        clf = fit_linear_classifier(xa_tr, cond_idx[split == "train"], len(conditions))
        row["condition_acc_val"], row["condition_chance"] = accuracy(clf, xa_va, cond_idx[split == "val"]), 1.0 / len(conditions)
        for tag, mask in (("post_grasp", post), ("pre_grasp", pre)):
            c_tr, c_va = (split == "train") & is_carton & mask, (split == "val") & is_carton & mask
            if c_tr.sum() > 10 and c_va.sum() > 10:
                pc = Projector(x[c_tr])
                y_tr, y_va = (cond[c_tr] == "carton_full").astype(int), (cond[c_va] == "carton_full").astype(int)
                row[f"carton_ef_acc_{tag}"] = accuracy(fit_linear_classifier(pc(x[c_tr]), y_tr, 2), pc(x[c_va]), y_va)
                row[f"carton_ef_n_val_{tag}"] = int(c_va.sum())
        cv_mask = is_carton[va] & post[va]
        row["carton_mu_hat_empty_post"] = _mean(mu_hat[cv_mask & (cond[va] == "carton_empty")])
        row["carton_mu_hat_full_post"] = _mean(mu_hat[cv_mask & (cond[va] == "carton_full")])
        out_rows.append(row)
        f = lambda v, p=3: ("--" if v is None else f"{v:.{p}f}")  # noqa: E731
        log(
            f"  {key:18s} {x.shape[1]:4d}->{x_tr.shape[1]:3d}  lam {lam:>6g}  KL val {f(row['kl_val'])} (grasp {f(row['kl_val_grasp'])} hold {f(row['kl_val_hold'])} | carton {f(row['kl_val_carton'])} produce {f(row['kl_val_produce'])})"
            f"  mu-only {f(row['kl_mu_only_val'])}  sigma_hat~{f(row['sigma_hat_val_median'])}"
            f"  | cond-acc {f(row['condition_acc_val'],2)}  carton E/F post {f(row.get('carton_ef_acc_post_grasp'),2)} pre {f(row.get('carton_ef_acc_pre_grasp'),2)}"
            f"  mu_hat E/F post {f(row['carton_mu_hat_empty_post'])}/{f(row['carton_mu_hat_full_post'])}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", required=True, metavar="NAME=features.npz")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_file = (out.parent / "probe_results.log").open("w")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")

    rows: list = []
    for spec in args.arm:
        name, path = spec.split("=", 1)
        evaluate_arm(name, pathlib.Path(path), rows, log)
    out.write_text(json.dumps({"rows": rows, "note": "linear probes on frozen features (standardise + PCA<=128, ridge with episode-grouped CV); KL = forward Gaussian KL to the train-only group targets on held-out frames; baseline = constant train-set prior; carton E/F pre-grasp accuracy = appearance/instance leakage control"}, indent=2))
    log(f"\nwrote {out}")


if __name__ == "__main__":
    main()
