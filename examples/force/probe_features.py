"""Extract frozen internal features of one task12 arm on demonstration frames, for offline probing.

Answers, for EVERY arm, the question the distribution head answers only for PiVLA: how much of the
demonstrated safe-interaction statistics is decodable from the model's representation? The model is
not changed and nothing is sampled; `Pi0.probe_features` runs the frozen prefix (and, on force-aware
arms, the FVLMoE force path) and this script stores the resulting vectors together with the frame's
labels. `probe_train_eval.py` then fits the same small head on every arm's features (train episodes)
and scores it on the held-out episodes -- one ruler for PiVLA, ForceVLA and no-force pi0.

Frames come from the post-processed records + the original 1080p images exactly as in
g1_gate_offline.py (same crop, same zeroing, same contact_input); labels come from the train-only
group table (mu_g, sigma_g, prototype). Every episode is tagged train/val from the split files.

Usage (one arm per call, ~0.3 s per frame on a 2080 Ti):
  uv run examples/force/probe_features.py --config-name pi0_draftvla_task12 \
      --checkpoint-dir checkpoints/pi0_draftvla_task12/task12_full/9999 \
      --out examples/force/probe_20260921/full/features.npz
Default frame selection: every 3rd frame of the contact stages plus the 30 prepare frames before the
first grasp frame, for every episode that has local frames (--stride, --prepare-frames, --episodes).
"""

import argparse
import json
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import draftvla_contact as contact  # noqa: E402
from g1_gate_offline import _DEFAULT_IMAGES_ROOT, _DEFAULT_LABELS_DIR, _DEFAULT_RECORDS_ROOT, _crop_resize, _read_episode_list  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.shared import nnx_utils  # noqa: E402
import openpi.training.config as _config  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent
CONTACT_STAGES = ("grasp", "lift", "translate", "place")


def select_frames(records, prepare_frames: int, stride: int) -> list[int]:
    stages = [r.get("stage") for r in records]
    contact_idx = [i for i, s in enumerate(stages) if s in CONTACT_STAGES]
    if not contact_idx:
        return []
    first = contact_idx[0]
    lead = [i for i in range(max(0, first - prepare_frames), first) if stages[i] == "prepare"]
    return lead[::stride] + contact_idx[::stride]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--records-root", default=str(_DEFAULT_RECORDS_ROOT))
    p.add_argument("--images-root", default=str(_DEFAULT_IMAGES_ROOT))
    p.add_argument("--labels-dir", default=str(_DEFAULT_LABELS_DIR))
    p.add_argument("--train-list", default=str(_HERE / "train_episodes_20260917.txt"))
    p.add_argument("--val-list", default=str(_HERE / "val_episodes_20260917.txt"))
    p.add_argument("--episodes", nargs="*", default=None, help="subset; default = every train+val episode with local frames")
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--prepare-frames", type=int, default=30)
    p.add_argument("--max-frames-per-episode", type=int, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    records_root, images_root = pathlib.Path(args.records_root), pathlib.Path(args.images_root)
    train = _read_episode_list(pathlib.Path(args.train_list))
    val = _read_episode_list(pathlib.Path(args.val_list))
    split = {**{e: "train" for e in train}, **{e: "val" for e in val}}
    episodes = args.episodes or [e for e in train + val if (images_root / e / "rgb").exists()]
    skipped = [e for e in train + val if e not in episodes]
    groups = {(g["condition"], g["stage_group"]): g for g in json.loads((pathlib.Path(args.labels_dir) / "group_metadata.json").read_text())["groups"]}

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = (out_path.parent / "extract.log").open("a")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"config {args.config_name}\ncheckpoint {args.checkpoint_dir}\nepisodes {len(episodes)} (skipped without frames: {len(skipped)}: {skipped})")

    config = _config.get_config(args.config_name)
    t0 = time.monotonic()
    policy = _policy_config.create_trained_policy(config, args.checkpoint_dir)
    model = policy._model  # noqa: SLF001 - the transforms and the loaded weights are exactly the serving ones
    probe = nnx_utils.module_jit(model.probe_features)
    log(f"policy loaded in {time.monotonic() - t0:.0f}s")

    feats: dict[str, list[np.ndarray]] = {}
    meta: dict[str, list] = {k: [] for k in ("episode", "condition", "split", "index", "stage", "stage_group", "group_id", "mu_target", "sigma_target", "prototype", "grip", "width", "t_from_grasp_start")}
    n_done = 0
    for ep in episodes:
        records = contact.load_records(records_root / ep)
        condition = contact.episode_condition(records, ep)
        zeroing = contact.tactile_zeroing(records, ep)
        chosen = select_frames(records, args.prepare_frames, args.stride)
        if args.max_frames_per_episode:
            chosen = chosen[: args.max_frames_per_episode]
        first_grasp = next(i for i, r in enumerate(records) if r.get("stage") in CONTACT_STAGES)
        t_ep = time.monotonic()
        for i in chosen:
            r = records[i]
            tcp = r["tcp_pose"]
            obs = {
                "observation/image": _crop_resize(images_root / ep / r["image_path"], r["image_crop_box"]),
                "observation/wrist_image": _crop_resize(images_root / ep / r["wrist_image_path"], r["wrist_image_crop_box"]),
                "observation/state": np.asarray([*tcp["position_xyz"], *tcp["rotation_vector"], r["gripper_width"]], dtype=np.float32),
                "observation/contact_input": contact.contact_input_57(r, zeroing, ep),
                "prompt": r["prompt"],
            }
            inputs = policy._input_transform(jax.tree.map(lambda x: x, obs))  # noqa: SLF001
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            out = probe(_model.Observation.from_dict(inputs))
            for k, v in out.items():
                feats.setdefault(k, []).append(np.asarray(v[0], dtype=np.float32))
            sg = contact.STAGE_TO_GROUP.get(r.get("stage"))
            g = groups.get((condition, sg)) if sg else None
            left, right = contact.zeroed_tactile(r, zeroing)
            meta["episode"].append(ep)
            meta["condition"].append(condition)
            meta["split"].append(split.get(ep, "unknown"))
            meta["index"].append(int(r["index"]))
            meta["stage"].append(r.get("stage") or "")
            meta["stage_group"].append(sg or "")
            meta["group_id"].append(g["group_id"] if g else -1)
            meta["mu_target"].append(g["mu"] if g else np.nan)
            meta["sigma_target"].append(g["sigma"] if g else np.nan)
            meta["prototype"].append(g["prototype"] if g else -1)
            meta["grip"].append(contact.grip_scalar(left, right))
            meta["width"].append(float(r["gripper_width"]))
            meta["t_from_grasp_start"].append(i - first_grasp)
            n_done += 1
        log(f"  {ep} [{condition}, {split.get(ep)}] {len(chosen)} frames in {time.monotonic() - t_ep:.0f}s (total {n_done})")

    arrays = {k: np.stack(v) for k, v in feats.items()}
    arrays.update({k: np.asarray(v) for k, v in meta.items()})
    arrays["feature_keys"] = np.asarray(sorted(feats))
    arrays["config_name"] = np.asarray(args.config_name)
    arrays["checkpoint"] = np.asarray(args.checkpoint_dir)
    np.savez_compressed(out_path, **arrays)
    log(f"wrote {out_path}: {n_done} frames, features {[(k, arrays[k].shape[1]) for k in sorted(feats)]}  ({time.monotonic() - t0:.0f}s)")


if __name__ == "__main__":
    main()
