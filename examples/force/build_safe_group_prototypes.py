"""Rebuild the group-level safe wrench distributions and the K-means safe-interaction prototypes.

This recreates the annotation script described by ``DATASET_README_zh.md`` §7/§8 (the original is not
on disk). It is the prerequisite for the K/tau ablations and for the train-only label rebuild that
closes the label-side leak (plan §4.3.1 / §8.1).

Recipe (plan §4.2, verified against the shipped labels):
  group   = (task=pick_place, fruit, stage) over stages {grasp, lift, translate, place} -> 12 groups.
  signal  = per-dim mean of ``tactile_estimated_wrenches.left_estimated`` and ``.right_estimated``.
  mu_g    = mean over all frames of the group; sigma_g = population std (ddof=0).
  safe_distribution_g = concat(mu_g, sigma_g)                                            [12]
  q_g     = concat(norm(mu_g), norm(log(clamp(sigma_g, 1e-4))))                          [12]
            where norm is per-dim robust (x - median) / (IQR + 1e-6) over the groups in use.
  C       = K-means(Q, K=4, seed=0)                                                      [K, 12]
  Y       = softmax(-mean_sq_dist(Q, C) / tau_q)                                         [12, K]

Usage:
  # Regression gate: reproduce the shipped labels from the 22 contributing episodes.
  uv run examples/force/build_safe_group_prototypes.py --verify-parity

  # Rebuild the shipped tables.
  uv run examples/force/build_safe_group_prototypes.py --output-dir /tmp/proto --write

  # Train-only rebuild (plan §8.1): additionally hold out the 4 validation episodes and emit
  # per-episode ``labels.jsonl`` sidecars for the converter's --labels-dir.
  uv run examples/force/build_safe_group_prototypes.py \
      --val-episodes <potato-validation-episode> --val-episodes <pear-validation-episode> \
      --val-episodes <banana-validation-episode> \
      --output-dir prototype_metadata_trainonly --write

Nothing is written unless --write (or --verify-parity, which only reads) is given.
"""

import argparse
import dataclasses
import json
import pathlib

import numpy as np

# Resolve the complete uploaded dataset next to the repository.
DEFAULT_DATA_DIR = pathlib.Path(__file__).resolve().parents[3] / "DamageVLA_training_post_process_20260821"

# The 4 failed episodes are excluded from both group statistics and training conversion.
DEFAULT_EXCLUDE = (
    "pi0_train_20260821_152043",  # pear, failed grasp/place
    "pi0_train_20260821_152329",  # pear, failed grasp/place
    "pi0_train_20260821_155925",  # banana, failed grasp/place
    "pi0_train_20260821_161238",  # banana, failed grasp/place
)

TASK = "pick_place"
FRUITS = ("banana", "pear", "potato")
STAGES = ("grasp", "lift", "translate", "place")
WRENCH_LAYOUT = ("fx", "fy", "fz", "tx", "ty", "tz")
SIGMA_CLAMP = 1e-4
IQR_EPS = 1e-6

# group_id = fruit_index * len(STAGES) + stage_index; verified against every labeled frame on disk.
GROUP_KEYS = tuple((TASK, fruit, stage) for fruit in FRUITS for stage in STAGES)
GROUP_INDEX = {key: i for i, key in enumerate(GROUP_KEYS)}


def _load_records(episode_dir: pathlib.Path) -> list[dict]:
    with (episode_dir / "observations.jsonl").open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    records.sort(key=lambda r: r["index"])
    return records


def _two_finger_mean(record: dict) -> np.ndarray:
    """Per-dim mean of the left and right estimated tactile wrenches -> [6]."""
    wrenches = record["tactile_estimated_wrenches"]
    left = np.asarray(wrenches["left_estimated"], dtype=np.float64)
    right = np.asarray(wrenches["right_estimated"], dtype=np.float64)
    return 0.5 * (left + right)


def _robust_norm(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-dim robust normalization over the group axis: (x - median) / (IQR + eps)."""
    median = np.median(x, axis=0)
    iqr = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
    return (x - median) / (iqr + IQR_EPS), median, iqr


def _kmeans_plusplus_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic k-means++ seeding (Arthur & Vassilvitskii 2007)."""
    n = x.shape[0]
    centers = np.empty((k, x.shape[1]), dtype=x.dtype)
    centers[0] = x[rng.integers(n)]
    closest_sq = ((x - centers[0]) ** 2).sum(-1)
    for i in range(1, k):
        total = closest_sq.sum()
        if total <= 0:  # All points already coincide with a center.
            centers[i] = x[rng.integers(n)]
        else:
            centers[i] = x[rng.choice(n, p=closest_sq / total)]
        closest_sq = np.minimum(closest_sq, ((x - centers[i]) ** 2).sum(-1))
    return centers


def _kmeans(x: np.ndarray, k: int, *, seed: int = 0, n_init: int = 50, max_iter: int = 300) -> np.ndarray:
    """Lloyd's algorithm with k-means++ seeding. Deterministic given ``seed``; returns centers [k, D]."""
    best_centers, best_inertia = None, np.inf
    for run in range(n_init):
        rng = np.random.default_rng(seed + run)
        centers = _kmeans_plusplus_init(x, k, rng)
        labels = np.zeros(x.shape[0], dtype=np.int64)
        for _ in range(max_iter):
            dist_sq = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            new_labels = dist_sq.argmin(-1)
            new_centers = centers.copy()
            for j in range(k):
                members = x[new_labels == j]
                if len(members):
                    new_centers[j] = members.mean(0)
            if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
                centers, labels = new_centers, new_labels
                break
            centers, labels = new_centers, new_labels
        inertia = ((x - centers[labels]) ** 2).sum()
        if inertia < best_inertia - 1e-12:
            best_centers, best_inertia = centers, inertia
    # Canonical ordering so the run is reproducible regardless of the init draw.
    order = np.lexsort(best_centers.T[::-1])
    return best_centers[order]


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


@dataclasses.dataclass
class Tables:
    safe_distribution_all: np.ndarray  # [12, 12]
    descriptors: np.ndarray  # [12, 12]
    prototype_centers: np.ndarray  # [K, 12]
    soft_prototype_targets_all: np.ndarray  # [12, K]
    normalizer_stats: dict[str, np.ndarray]
    group_frame_counts: np.ndarray  # [16]
    group_episode_counts: np.ndarray  # [16]


def _collect_group_frames(episode_dirs: list[pathlib.Path]) -> tuple[dict[int, list[np.ndarray]], dict[int, set[str]]]:
    frames: dict[int, list[np.ndarray]] = {g: [] for g in range(len(GROUP_KEYS))}
    episodes: dict[int, set[str]] = {g: set() for g in range(len(GROUP_KEYS))}
    for episode_dir in episode_dirs:
        for r in _load_records(episode_dir):
            stage = r.get("stage")
            if stage not in STAGES:
                continue
            fruit = r.get("group_fruit")
            key = (r.get("group_task") or TASK, fruit, stage)
            if key not in GROUP_INDEX:
                continue
            group_id = GROUP_INDEX[key]
            # group_id in the jsonl is authoritative for group identity (plan §4.2).
            if r.get("group_id") is not None and r["group_id"] >= 0 and r["group_id"] != group_id:
                raise ValueError(f"{episode_dir.name} frame {r['index']}: group_id {r['group_id']} != {group_id} {key}")
            frames[group_id].append(_two_finger_mean(r))
            episodes[group_id].add(episode_dir.name)
    return frames, episodes


def _build_tables(episode_dirs: list[pathlib.Path], *, k: int, tau_q: float) -> Tables:
    frames, episodes = _collect_group_frames(episode_dirs)
    empty = [GROUP_KEYS[g] for g, v in frames.items() if not v]
    if empty:
        raise ValueError(f"No frames for group(s) {empty} — cannot build a [12, 12] table.")

    safe = np.zeros((len(GROUP_KEYS), 12), dtype=np.float64)
    counts = np.zeros(len(GROUP_KEYS), dtype=np.int64)
    for g in range(len(GROUP_KEYS)):
        x = np.stack(frames[g])  # [n, 6]
        safe[g] = np.concatenate([x.mean(0), x.std(0, ddof=0)])
        counts[g] = len(x)

    mu, sigma = safe[:, :6], safe[:, 6:]
    log_sigma = np.log(np.clip(sigma, SIGMA_CLAMP, None))
    mu_norm, mu_median, mu_iqr = _robust_norm(mu)
    log_sigma_norm, log_sigma_median, log_sigma_iqr = _robust_norm(log_sigma)
    descriptors = np.concatenate([mu_norm, log_sigma_norm], axis=1)  # [12, 12]

    centers = _kmeans(descriptors, k, seed=0)
    dist_sq = ((descriptors[:, None, :] - centers[None, :, :]) ** 2).mean(-1)  # [16, K]
    targets = _softmax(-dist_sq / tau_q, axis=-1)

    return Tables(
        safe_distribution_all=safe,
        descriptors=descriptors,
        prototype_centers=centers,
        soft_prototype_targets_all=targets,
        normalizer_stats={
            "mu_median": mu_median,
            "mu_iqr": mu_iqr,
            "log_sigma_median": log_sigma_median,
            "log_sigma_iqr": log_sigma_iqr,
        },
        group_frame_counts=counts,
        group_episode_counts=np.asarray([len(episodes[g]) for g in range(len(GROUP_KEYS))], dtype=np.int64),
    )


def _resolve_episodes(root: pathlib.Path, exclude: set[str]) -> tuple[list[pathlib.Path], list[str]]:
    all_dirs = sorted(p.parent for p in root.glob("*/observations.jsonl"))
    if not all_dirs:
        raise FileNotFoundError(f"No <episode>/observations.jsonl under {root}")
    unknown = exclude - {d.name for d in all_dirs}
    if unknown:
        raise ValueError(f"Excluded episode(s) not found under {root}: {sorted(unknown)}")
    included = [d for d in all_dirs if d.name not in exclude]
    return included, [d.name for d in all_dirs if d.name in exclude]


def _verify_parity(root: pathlib.Path, *, k: int, tau_q: float, tol: float = 1e-9) -> bool:
    """Assert the recomputed safe distributions reproduce the SHIPPED per-frame labels (plan §10.10)."""
    included, excluded = _resolve_episodes(root, set(DEFAULT_EXCLUDE))
    print(f"[parity] {len(included)} contributing episode(s), {len(excluded)} excluded: {excluded}")
    tables = _build_tables(included, k=k, tau_q=tau_q)

    # Shipped per-frame gt_safe_distribution, one representative row per group (constant within a group).
    shipped: dict[int, np.ndarray] = {}
    for episode_dir in included:
        for r in _load_records(episode_dir):
            g, dist = r.get("group_id"), r.get("gt_safe_distribution")
            if g is None or g < 0 or dist is None:
                continue
            dist = np.asarray(dist, dtype=np.float64)
            if g in shipped and not np.allclose(shipped[g], dist, atol=0, rtol=0):
                raise ValueError(f"group {g}: shipped gt_safe_distribution is not constant within the group")
            shipped[g] = dist
    missing = set(range(len(GROUP_KEYS))) - set(shipped)
    if missing:
        raise ValueError(f"No shipped label found for group(s) {sorted(missing)}")

    print(f"\n[parity] recomputed vs shipped gt_safe_distribution, tol={tol:g}")
    print(f"{'gid':>3}  {'group':<28} {'frames':>6}  {'max |abs err|':>13}")
    ok = True
    for g, key in enumerate(GROUP_KEYS):
        err = float(np.abs(tables.safe_distribution_all[g] - shipped[g]).max())
        ok &= err <= tol
        flag = "" if err <= tol else "   <-- FAIL"
        print(f"{g:>3}  {'/'.join(key[1:]):<28} {tables.group_frame_counts[g]:>6}  {err:>13.3e}{flag}")

    known = {  # Shipped 20260821 table, banana/grasp.
        "mu": [-1.993, -0.295, 0.975, 8.42, 7.026, 26.706],
        "sigma": [2.2939, 1.0844, 1.5062, 13.1031, 23.3223, 27.9388],
    }
    g = GROUP_INDEX[(TASK, "banana", "grasp")]
    mu, sigma = tables.safe_distribution_all[g, :6], tables.safe_distribution_all[g, 6:]
    print(f"\n[parity] known-good check, group {g} (banana, grasp):")
    print(f"  mu    = {np.round(mu, 3).tolist()}   expected {known['mu']}")
    print(f"  sigma = {np.round(sigma, 4).tolist()}   expected {known['sigma']}")
    known_ok = np.allclose(np.round(mu, 3), known["mu"], atol=0) and np.allclose(
        np.round(sigma, 4), known["sigma"], atol=0
    )
    ok &= known_ok
    print(f"  -> {'MATCH' if known_ok else 'MISMATCH'}")

    # Not part of the gate (K-means cluster ids are only defined up to a permutation), but reported.
    hard_recomputed = tables.soft_prototype_targets_all.argmax(-1)
    shipped_targets = {}
    for episode_dir in included:
        for r in _load_records(episode_dir):
            g, y = r.get("group_id"), r.get("soft_prototype_target")
            if g is not None and g >= 0 and y is not None:
                shipped_targets[g] = np.asarray(y, dtype=np.float64)
    if len(shipped_targets) == len(GROUP_KEYS):
        shipped_hard = np.asarray([shipped_targets[g].argmax() for g in range(len(GROUP_KEYS))])
        same_partition = _same_partition(hard_recomputed, shipped_hard)
        print("\n[parity] prototype partition vs shipped (informational, not gated): ", end="")
        print("IDENTICAL up to cluster relabeling" if same_partition else "DIFFERENT")
        for cluster in range(k):
            members = [
                f"{GROUP_KEYS[g][1]}/{GROUP_KEYS[g][2]}"
                for g in range(len(GROUP_KEYS))
                if hard_recomputed[g] == cluster
            ]
            print(f"  cluster {cluster}: {members}")

    print(f"\n[parity] {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def _same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """True if two label vectors induce the same partition (ignoring cluster id naming)."""
    mapping: dict[int, int] = {}
    inverse: dict[int, int] = {}
    for x, y in zip(a.tolist(), b.tolist(), strict=True):
        if mapping.setdefault(x, y) != y or inverse.setdefault(y, x) != x:
            return False
    return True


def _write_label_sidecars(
    root: pathlib.Path, output_dir: pathlib.Path, tables: Tables, *, stats_episodes: set[str]
) -> None:
    """Write per-episode labels.jsonl (never touching observations.jsonl) for the converter's --labels-dir."""
    n_written = 0
    for episode_dir in sorted(p.parent for p in root.glob("*/observations.jsonl")):
        rows = []
        for r in _load_records(episode_dir):
            stage = r.get("stage")
            key = (r.get("group_task") or TASK, r.get("group_fruit"), stage)
            group_id = GROUP_INDEX.get(key, -1) if stage in STAGES else -1
            valid = group_id >= 0
            rows.append(
                {
                    "index": r["index"],
                    "group_id": int(group_id),
                    "gt_safe_distribution": tables.safe_distribution_all[group_id].tolist() if valid else None,
                    "soft_prototype_target": tables.soft_prototype_targets_all[group_id].tolist() if valid else None,
                    "prototype_supervision_valid": bool(valid),
                }
            )
        out = output_dir / episode_dir.name / "labels.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        n_written += 1
        tag = "stats" if episode_dir.name in stats_episodes else "held-out/excluded"
        print(f"  labels.jsonl  {episode_dir.name}  {len(rows):>5} frames  ({tag})")
    print(f"Wrote {n_written} labels.jsonl sidecar(s) under {output_dir}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Root holding pi0_train_*/ dirs")
    p.add_argument(
        "--exclude",
        action="append",
        metavar="EP",
        help="Repeatable; excluded from the statistics. Defaults to the 4 failed episodes in the 20260821 batch.",
    )
    p.add_argument(
        "--val-episodes",
        action="append",
        metavar="EP",
        help="Repeatable; additionally held out of the statistics (train-only rebuild, plan §8.1). "
        "When given, per-episode labels.jsonl sidecars are written under --output-dir.",
    )
    p.add_argument("--tau-q", type=float, default=0.1, help="Softmax temperature for the soft prototype targets")
    p.add_argument("--k", type=int, default=4, help="Number of K-means prototypes")
    p.add_argument("--output-dir", default="prototype_metadata", help="Where the tables are written")
    p.add_argument("--write", action="store_true", help="Actually write the outputs (default is a dry run)")
    p.add_argument("--verify-parity", action="store_true", help="Run the plan §10.10 regression gate and exit")
    return p.parse_args()


def main(
    data_dir: str = str(DEFAULT_DATA_DIR),
    *,
    exclude: list[str] | None = None,
    val_episodes: list[str] | None = None,
    tau_q: float = 0.1,
    k: int = 4,
    output_dir: str = "prototype_metadata",
    write: bool = False,
    verify_parity: bool = False,
) -> None:
    """Rebuild the group safe distributions + prototype tables."""
    exclude = list(DEFAULT_EXCLUDE) if exclude is None else exclude
    val_episodes = val_episodes or []
    root = pathlib.Path(data_dir)

    if verify_parity:
        if not _verify_parity(root, k=k, tau_q=tau_q):
            raise SystemExit("Label parity gate FAILED")
        return

    stats_exclude = set(exclude) | set(val_episodes)
    included, excluded_names = _resolve_episodes(root, stats_exclude)
    print(f"Data root:        {root}")
    print(f"Contributing:     {len(included)} episode(s)")
    print(f"Excluded (denylist + val): {len(excluded_names)} -> {excluded_names}")
    print(f"Hyperparams:      K={k}  tau_q={tau_q}  stages={list(STAGES)}  N_group={len(GROUP_KEYS)}")

    tables = _build_tables(included, k=k, tau_q=tau_q)

    print(f"\n{'gid':>3}  {'group':<20} {'eps':>3} {'frames':>6}  mu[:3]                     sigma[:3]")
    for g, key in enumerate(GROUP_KEYS):
        mu, sigma = tables.safe_distribution_all[g, :3], tables.safe_distribution_all[g, 6:9]
        print(
            f"{g:>3}  {'/'.join(key[1:]):<20} {tables.group_episode_counts[g]:>3} "
            f"{tables.group_frame_counts[g]:>6}  {np.round(mu, 3).tolist()!s:<26} {np.round(sigma, 3).tolist()}"
        )
    hard = tables.soft_prototype_targets_all.argmax(-1)
    print("\nPrototype membership:")
    for cluster in range(k):
        members = [f"{GROUP_KEYS[g][1]}/{GROUP_KEYS[g][2]}" for g in range(len(GROUP_KEYS)) if hard[g] == cluster]
        print(f"  cluster {cluster}: {members}")

    if not write:
        print(f"\nDry run — nothing written. Re-run with --write to populate {output_dir}.")
        return

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "safe_distribution_all.npy", tables.safe_distribution_all.astype(np.float64))
    np.save(out / "prototype_centers.npy", tables.prototype_centers.astype(np.float64))
    np.save(out / "soft_prototype_targets_all.npy", tables.soft_prototype_targets_all.astype(np.float64))
    np.save(out / "descriptors.npy", tables.descriptors.astype(np.float64))
    np.savez(out / "normalizer_stats.npz", **tables.normalizer_stats)

    metadata = {
        "task": TASK,
        "fruits": list(FRUITS),
        "stages": list(STAGES),
        "n_group": len(GROUP_KEYS),
        "group_layout": "group_id = fruit_index * len(stages) + stage_index",
        "groups": [
            {
                "group_id": g,
                "task": key[0],
                "fruit": key[1],
                "stage": key[2],
                "n_frames": int(tables.group_frame_counts[g]),
                "n_episodes": int(tables.group_episode_counts[g]),
                "prototype": int(hard[g]),
            }
            for g, key in enumerate(GROUP_KEYS)
        ],
        "included_episodes": [d.name for d in included],
        "excluded_episodes": excluded_names,
        "denylist_episodes": sorted(set(exclude)),
        "val_episodes": sorted(set(val_episodes)),
        "hyperparams": {
            "k": k,
            "tau_q": tau_q,
            "kmeans_seed": 0,
            "sigma_clamp": SIGMA_CLAMP,
            "iqr_eps": IQR_EPS,
            "sigma_is_std_ddof": 0,
        },
        "signal": "tactile_estimated_wrench_two_finger_mean",
        "gt_safe_distribution_layout": [f"mu_{d}" for d in WRENCH_LAYOUT] + [f"sigma_{d}" for d in WRENCH_LAYOUT],
    }
    (out / "group_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out / "prototype_tables.json").write_text(
        json.dumps(
            {
                "safe_distribution_all": tables.safe_distribution_all.tolist(),
                "prototype_centers": tables.prototype_centers.tolist(),
                "soft_prototype_targets_all": tables.soft_prototype_targets_all.tolist(),
                "descriptors": tables.descriptors.tolist(),
                "normalizer_stats": {k_: v.tolist() for k_, v in tables.normalizer_stats.items()},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nWrote tables to {out}")

    if val_episodes:
        print("\nval_episodes given -> writing per-episode label sidecars (shipped observations.jsonl untouched):")
        _write_label_sidecars(root, out, tables, stats_episodes={d.name for d in included})


if __name__ == "__main__":
    args = _parse_args()
    main(
        data_dir=args.data_dir,
        exclude=args.exclude,
        val_episodes=args.val_episodes,
        tau_q=args.tau_q,
        k=args.k,
        output_dir=args.output_dir,
        write=args.write,
        verify_parity=args.verify_parity,
    )
