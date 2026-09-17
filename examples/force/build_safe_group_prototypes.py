"""Build the 16-group safe-GRIP distributions and K-means prototypes (contract of 2026-09-16, Plan B).

Replaces the 12-D wrench recipe (kept as build_safe_group_prototypes_legacy_wrench12.py; the GitHub
repo history has it too). The new recipe, per outlines/todo_training_contract.md §B/§D/§E/§F and
data_analysis_20260916.md:

  group    = (task=pick_place, condition, stage_group); 8 conditions x {grasp, hold} -> 16 groups.
             condition = fruit name (6) or carton_{empty,full} (2); hold = lift+translate+place.
  signal   = scalar grip_t = 0.5*(sum left_data_zeroed + sum right_data_zeroed), per-episode zeroed
             over the leading still-open prepare window (draftvla_contact.tactile_zeroing).
  mu_g     = mean over the EPISODE stage-group means of the contributing (train) episodes.
  sigma_g  = population std (ddof=0) over those episode means, clamped to >= 0.0051.
             EPISODE-level on purpose: sigma_g is "variation across safe demonstrations"
             (data_analysis §3); frame-level pooling would mostly measure the closing ramp
             inside the grasp stage (5x inflation there), which is trajectory structure.
  gt_safe_distribution_g = [mu_g, sigma_g]                                     (2 numbers)
  descriptor_g = robust-normed [mu, log sd, contact_area, cop_row, stiffness]  (5 features)
             Formulas and aggregation are 1:1 with outlines/analyze_groups.py (the K-scan script):
             the descriptor's sd is the MEAN over episodes of the within-episode frame-level std
             (unlike the LABEL sigma, which is the std over episode means); stiffness is the
             condition-level MEAN over episodes, shared by the condition's two stage groups.
  C        = K-means(descriptors, K=6, seed=0, 50 restarts);  Y = softmax(-mean_sq_dist / tau_q).

split-first (contract §F): --val-episodes/--val-file hold episodes out of ALL statistics; sidecar
labels.jsonl files are written for every episode (train, val, denylisted alike) by looking up the
train-fitted table -- the converter's --labels-dir consumes them and the denylist drops what must
never be trained on.

Usage:
  uv run examples/force/build_safe_group_prototypes.py \
      --val-file examples/force/val_episodes_20260917.txt \
      --output-dir prototype_metadata_task12_trainonly --write
"""

import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import draftvla_contact as contact  # noqa: E402

DEFAULT_DATA_DIR = pathlib.Path(__file__).resolve().parents[3] / "DamageVLA_training_post_process_20260916"

IQR_EPS = 1e-6
DESCRIPTOR_FEATURES = ("mu", "log_sigma", "contact_area", "cop_offset", "stiffness")


def _robust_norm(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.median(x, axis=0)
    iqr = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
    return (x - median) / (iqr + IQR_EPS), median, iqr


def _kmeans_plusplus_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[0]
    centers = np.empty((k, x.shape[1]), dtype=x.dtype)
    centers[0] = x[rng.integers(n)]
    closest_sq = ((x - centers[0]) ** 2).sum(-1)
    for i in range(1, k):
        total = closest_sq.sum()
        centers[i] = x[rng.integers(n)] if total <= 0 else x[rng.choice(n, p=closest_sq / total)]
        closest_sq = np.minimum(closest_sq, ((x - centers[i]) ** 2).sum(-1))
    return centers


def _kmeans(x: np.ndarray, k: int, *, seed: int = 0, n_init: int = 50, max_iter: int = 300) -> np.ndarray:
    best_centers, best_inertia = None, np.inf
    for run in range(n_init):
        rng = np.random.default_rng(seed + run)
        centers = _kmeans_plusplus_init(x, k, rng)
        labels = np.zeros(x.shape[0], dtype=np.int64)
        for _ in range(max_iter):
            new_labels = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1).argmin(-1)
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
    order = np.lexsort(best_centers.T[::-1])
    return best_centers[order]


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


@dataclasses.dataclass
class EpisodeFeatures:
    name: str
    condition: str
    zero_window: int
    n_frames: int
    stage_group_mean: dict[str, float]  # stage_group -> episode mean grip
    stage_group_std: dict[str, float]  # within-episode frame-level std (descriptor sd feature)
    stage_group_frames: dict[str, int]
    area_mean: dict[str, float]
    cop_mean: dict[str, float | None]
    stiffness: float | None


def _episode_features(episode_dir: pathlib.Path) -> EpisodeFeatures:
    records = contact.load_records(episode_dir)
    name = episode_dir.name
    condition = contact.episode_condition(records, name)
    zeroing = contact.tactile_zeroing(records, name)
    thr = zeroing.contact_threshold

    grips, stages, widths = [], [], []
    areas, cops = [], []
    for r in records:
        left, right = contact.zeroed_tactile(r, zeroing)
        grips.append(contact.grip_scalar(left, right))
        stages.append(r.get("stage") or "")
        widths.append(float(r["gripper_width"]))
        areas.append(contact.contact_area(left, right, thr))
        cops.append(contact.cop_row(left, right, thr))
    grips = np.asarray(grips)
    stages_arr = np.asarray(stages)
    widths_arr = np.asarray(widths)

    sg_mean, sg_std, sg_frames, sg_area, sg_cop = {}, {}, {}, {}, {}
    for sg in contact.STAGE_GROUPS:
        mask = np.isin(stages_arr, [s for s, g in contact.STAGE_TO_GROUP.items() if g == sg])
        sg_frames[sg] = int(mask.sum())
        sg_mean[sg] = float(grips[mask].mean()) if mask.any() else float("nan")
        sg_std[sg] = float(grips[mask].std()) if mask.any() else float("nan")
        sg_area[sg] = float(np.asarray(areas)[mask].mean()) if mask.any() else float("nan")
        valid_cops = [c for c, m in zip(cops, mask) if m and c is not None]
        sg_cop[sg] = float(np.mean(valid_cops)) if valid_cops else None

    return EpisodeFeatures(
        name=name,
        condition=condition,
        zero_window=zeroing.window_size,
        n_frames=len(records),
        stage_group_mean=sg_mean,
        stage_group_std=sg_std,
        stage_group_frames=sg_frames,
        area_mean=sg_area,
        cop_mean=sg_cop,
        stiffness=contact.episode_stiffness(grips, widths_arr, stages_arr),
    )


@dataclasses.dataclass
class Tables:
    safe_distribution_all: np.ndarray  # [16, 2] = [mu, sigma]
    descriptors: np.ndarray  # [16, 5] robust-normed
    descriptors_raw: np.ndarray  # [16, 5] before norm
    prototype_centers: np.ndarray  # [K, 5]
    soft_prototype_targets_all: np.ndarray  # [16, K]
    normalizer_stats: dict[str, np.ndarray]
    group_frame_counts: np.ndarray
    group_episode_counts: np.ndarray
    episode_means: dict[str, dict[str, float]]  # audit: episode -> stage_group -> mean


def _build_tables(train_feats: list[EpisodeFeatures], *, k: int, tau_q: float) -> Tables:
    n_group = len(contact.GROUP_KEYS)
    by_group_means: dict[int, list[float]] = {g: [] for g in range(n_group)}
    by_group_stds: dict[int, list[float]] = {g: [] for g in range(n_group)}
    by_group_frames: dict[int, int] = {g: 0 for g in range(n_group)}
    by_group_eps: dict[int, int] = {g: 0 for g in range(n_group)}
    by_group_area: dict[int, list[float]] = {g: [] for g in range(n_group)}
    by_group_cop: dict[int, list[float]] = {g: [] for g in range(n_group)}
    by_cond_stiff: dict[str, list[float]] = {c: [] for c in contact.CONDITIONS}

    for f in train_feats:
        for sg in contact.STAGE_GROUPS:
            g = contact.GROUP_INDEX[(contact.TASK, f.condition, sg)]
            if f.stage_group_frames[sg] == 0 or not np.isfinite(f.stage_group_mean[sg]):
                raise ValueError(f"{f.name}: no frames in stage group {sg!r}")
            by_group_means[g].append(f.stage_group_mean[sg])
            by_group_stds[g].append(f.stage_group_std[sg])
            by_group_frames[g] += f.stage_group_frames[sg]
            by_group_eps[g] += 1
            by_group_area[g].append(f.area_mean[sg])
            if f.cop_mean[sg] is not None:
                by_group_cop[g].append(f.cop_mean[sg])
        if f.stiffness is not None:
            by_cond_stiff[f.condition].append(f.stiffness)

    empty = [contact.GROUP_KEYS[g] for g in range(n_group) if not by_group_means[g]]
    if empty:
        raise ValueError(f"No training episodes for group(s): {empty}")

    safe = np.zeros((n_group, 2))
    raw_desc = np.zeros((n_group, len(DESCRIPTOR_FEATURES)))
    for g, (task, cond, sg) in enumerate(contact.GROUP_KEYS):
        means = np.asarray(by_group_means[g])
        # LABEL: episode-level statistics (decided 2026-09-17) -- sigma over episode means.
        mu = float(means.mean())
        sigma = max(float(means.std(ddof=0)), contact.GRIP_SIGMA_FLOOR)
        safe[g] = [mu, sigma]
        # DESCRIPTOR (analyze_groups.py): sd = mean over episodes of the within-episode frame std;
        # stiffness = condition-level mean over episodes.
        desc_sd = max(float(np.mean(by_group_stds[g])), contact.GRIP_SIGMA_FLOOR)
        cops = by_group_cop[g]
        stiffs = by_cond_stiff[cond]
        if not stiffs:
            raise ValueError(f"No stiffness estimate for condition {cond!r}")
        raw_desc[g] = [
            mu,
            np.log(desc_sd),
            float(np.mean(by_group_area[g])),
            float(np.mean(cops)) if cops else 0.0,
            float(np.mean(stiffs)),
        ]

    desc, median, iqr = _robust_norm(raw_desc)
    centers = _kmeans(desc, k, seed=0)
    dist_sq = ((desc[:, None, :] - centers[None, :, :]) ** 2).mean(-1)
    targets = _softmax(-dist_sq / tau_q, axis=-1)

    return Tables(
        safe_distribution_all=safe,
        descriptors=desc,
        descriptors_raw=raw_desc,
        prototype_centers=centers,
        soft_prototype_targets_all=targets,
        normalizer_stats={"median": median, "iqr": iqr},
        group_frame_counts=np.asarray([by_group_frames[g] for g in range(n_group)], dtype=np.int64),
        group_episode_counts=np.asarray([by_group_eps[g] for g in range(n_group)], dtype=np.int64),
        episode_means={f.name: dict(f.stage_group_mean) for f in train_feats},
    )


def _write_label_sidecars(
    all_dirs: list[pathlib.Path], output_dir: pathlib.Path, tables: Tables, *, train_names: set[str], deny: set[str]
) -> None:
    """labels.jsonl per episode (observations.jsonl is never touched). Every episode gets one --
    train and val alike look up the SAME train-fitted table; the converter's denylist decides
    what is never trained on."""
    for episode_dir in all_dirs:
        records = contact.load_records(episode_dir)
        condition = contact.episode_condition(records, episode_dir.name)
        rows = []
        for r in records:
            stage = r.get("stage")
            if stage in contact.CONTACT_STAGES:
                sg = contact.STAGE_TO_GROUP[stage]
                g = contact.GROUP_INDEX[(contact.TASK, condition, sg)]
                rows.append(
                    {
                        "index": r["index"],
                        "group_id": int(g),
                        "gt_safe_distribution": tables.safe_distribution_all[g].tolist(),
                        "soft_prototype_target": tables.soft_prototype_targets_all[g].tolist(),
                        "prototype_supervision_valid": True,
                    }
                )
            else:
                rows.append(
                    {
                        "index": r["index"],
                        "group_id": -1,
                        "gt_safe_distribution": None,
                        "soft_prototype_target": None,
                        "prototype_supervision_valid": False,
                    }
                )
        out = output_dir / episode_dir.name / "labels.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        tag = "denylist" if episode_dir.name in deny else ("train" if episode_dir.name in train_names else "val")
        print(f"  labels.jsonl  {episode_dir.name}  {len(rows):>5} frames  ({tag})")


def _read_val_file(path: str | None) -> list[str]:
    if not path:
        return []
    lines = pathlib.Path(path).read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def main(
    data_dir: str = str(DEFAULT_DATA_DIR),
    *,
    exclude: list[str] | None = None,
    val_episodes: list[str] | None = None,
    val_file: str | None = None,
    tau_q: float = 0.1,
    k: int = 6,
    output_dir: str = "prototype_metadata_task12_trainonly",
    write: bool = False,
) -> None:
    exclude = list(contact.DEFAULT_EXCLUDE) if exclude is None else exclude
    val = sorted(set(val_episodes or []) | set(_read_val_file(val_file)))
    root = pathlib.Path(data_dir)

    all_dirs = sorted(p.parent for p in root.glob("*/observations.jsonl"))
    if not all_dirs:
        raise FileNotFoundError(f"No <episode>/observations.jsonl under {root}")
    names = {d.name for d in all_dirs}
    for label, subset in (("denylist", exclude), ("val", val)):
        unknown = set(subset) - names
        if unknown:
            raise ValueError(f"{label} episode(s) not found under {root}: {sorted(unknown)}")
    overlap = set(exclude) & set(val)
    if overlap:
        raise ValueError(f"Episode(s) in both denylist and val: {sorted(overlap)}")

    deny = set(exclude)
    train_dirs = [d for d in all_dirs if d.name not in deny and d.name not in set(val)]
    print(f"Data root:   {root}")
    print(f"Episodes:    {len(all_dirs)} total = {len(train_dirs)} train + {len(val)} val + {len(deny)} denylisted")
    print(f"Hyperparams: K={k}  tau_q={tau_q}  sigma_floor={contact.GRIP_SIGMA_FLOOR}  N_group={len(contact.GROUP_KEYS)}\n")

    train_feats = [_episode_features(d) for d in train_dirs]
    tables = _build_tables(train_feats, k=k, tau_q=tau_q)

    print(f"{'gid':>3}  {'group':<24} {'eps':>3} {'frames':>6} {'mu':>8} {'sigma':>8}   raw desc [area cop stiff]")
    for g, (task, cond, sg) in enumerate(contact.GROUP_KEYS):
        mu, sigma = tables.safe_distribution_all[g]
        a, c_, s_ = tables.descriptors_raw[g, 2:]
        print(
            f"{g:>3}  {cond + '/' + sg:<24} {tables.group_episode_counts[g]:>3} "
            f"{tables.group_frame_counts[g]:>6} {mu:>8.4f} {sigma:>8.4f}   [{a:6.1f} {c_:5.2f} {s_:7.1f}]"
        )
    hard = tables.soft_prototype_targets_all.argmax(-1)
    print("\nPrototype membership:")
    for cluster in range(k):
        members = [
            f"{contact.GROUP_KEYS[g][1]}/{contact.GROUP_KEYS[g][2]}"
            for g in range(len(contact.GROUP_KEYS))
            if hard[g] == cluster
        ]
        print(f"  cluster {cluster}: {members}")

    # The model-config normalizer for the 1-D label (contract §B): mean of the group mus, median of
    # the group sigmas -- same construction as the legacy 6-D one.
    label_mean = float(tables.safe_distribution_all[:, 0].mean())
    label_scale = float(np.median(tables.safe_distribution_all[:, 1]))
    print(f"\nphy_label_mean  = ({label_mean:.6f},)")
    print(f"phy_label_scale = ({label_scale:.6f},)")

    if not write:
        print(f"\nDry run -- nothing written. Re-run with --write to populate {output_dir}.")
        return

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "safe_distribution_all.npy", tables.safe_distribution_all)
    np.save(out / "prototype_centers.npy", tables.prototype_centers)
    np.save(out / "soft_prototype_targets_all.npy", tables.soft_prototype_targets_all)
    np.save(out / "descriptors.npy", tables.descriptors)
    np.save(out / "descriptors_raw.npy", tables.descriptors_raw)
    np.savez(out / "normalizer_stats.npz", **tables.normalizer_stats)

    metadata = {
        "contract": "todo_training_contract.md 2026-09-16 (Plan B): 1-D grip target, 16 groups",
        "task": contact.TASK,
        "conditions": list(contact.CONDITIONS),
        "stage_groups": list(contact.STAGE_GROUPS),
        "stage_to_group": contact.STAGE_TO_GROUP,
        "n_group": len(contact.GROUP_KEYS),
        "group_layout": "group_id = condition_index * 2 + stage_group_index",
        "signal": contact.CONTACT_INPUT_SIGNAL,
        "grip_definition": "0.5 * (sum(left_data_zeroed) + sum(right_data_zeroed)); per-episode mean-zeroed",
        "sigma_estimator": "population std ddof=0 over EPISODE stage-group means; floor 0.0051",
        "descriptor_features": list(DESCRIPTOR_FEATURES),
        "descriptor_note": "area/cop/stiffness formulas pending reconciliation with analyze_groups.py",
        "gt_safe_distribution_layout": ["mu_grip", "sigma_grip"],
        "groups": [
            {
                "group_id": g,
                "condition": key[1],
                "stage_group": key[2],
                "mu": float(tables.safe_distribution_all[g, 0]),
                "sigma": float(tables.safe_distribution_all[g, 1]),
                "n_frames": int(tables.group_frame_counts[g]),
                "n_episodes": int(tables.group_episode_counts[g]),
                "prototype": int(hard[g]),
            }
            for g, key in enumerate(contact.GROUP_KEYS)
        ],
        "train_episodes": [d.name for d in train_dirs],
        "val_episodes": val,
        "denylist_episodes": sorted(deny),
        "episode_stage_group_means": tables.episode_means,
        "phy_label_mean": [label_mean],
        "phy_label_scale": [label_scale],
        "hyperparams": {"k": k, "tau_q": tau_q, "kmeans_seed": 0, "kmeans_restarts": 50,
                        "sigma_floor": contact.GRIP_SIGMA_FLOOR, "iqr_eps": IQR_EPS},
    }
    (out / "group_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\nWrote tables to {out}\n\nSidecars:")
    _write_label_sidecars(all_dirs, out, tables, train_names={d.name for d in train_dirs}, deny=deny)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--exclude", action="append", metavar="EP",
                   help="Repeatable denylist; defaults to the 6-episode contract denylist")
    p.add_argument("--val-episodes", action="append", metavar="EP", help="Repeatable; held out of all statistics")
    p.add_argument("--val-file", default=None, help="File with one held-out episode name per line")
    p.add_argument("--tau-q", type=float, default=0.1)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--output-dir", default="prototype_metadata_task12_trainonly")
    p.add_argument("--write", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    main(data_dir=a.data_dir, exclude=a.exclude, val_episodes=a.val_episodes, val_file=a.val_file,
         tau_q=a.tau_q, k=a.k, output_dir=a.output_dir, write=a.write)
