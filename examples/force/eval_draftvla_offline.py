"""Offline validation-set evaluation for DraftVLA checkpoints (contract §F).

Loads a trained checkpoint, runs `compute_train_losses` over the *validation* LeRobot repo (no
gradients), and reports the metrics that decide checkpoint selection and diagnose the physical
branch -- aggregated AND split by stage group (grasp / hold), the split the supervision groups use.

The stage split works by masking: group_id = condition_index * 2 + stage_group_index, so parity
selects the stage group; `supervision_valid` is ANDed with the parity mask before the loss call, so
the masked means inside the model become stage-group-specific. loss_flow is unmasked by design and
is only reported in the "all" row.

Read the val loss_dist AGAINST val loss_dist_baseline: below baseline = the head reads physical
state out of unseen episodes' tactile input; at baseline = the train-set fit was memorization.

Usage (typically every 500-1000 steps during/after training; pick the checkpoint by val action fit):
  uv run examples/force/eval_draftvla_offline.py \
      --config-name pi0_draftvla_task12 \
      --checkpoint-dir checkpoints/pi0_draftvla_task12/<exp>/<step> \
      --val-repo-id draftvla/task12_tactile_val
"""

import argparse
import dataclasses
import json
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

# Metrics worth printing, in report order. Anything else the model returns is kept in the JSON.
REPORT_KEYS = (
    "loss_flow",
    "loss_dist",
    "loss_dist_baseline",
    "kl_grip",
    "loss_proto",
    "loss_proto_excess",
    "proto_acc",
    "proto_entropy",
    "g_phy_rel",
    "phy_alpha",
    "z_phy_cos",
    "frac_valid",
    "num_valid",
)

STAGE_SPLITS = {"all": None, "grasp": 0, "hold": 1}  # group_id parity; None = no extra mask


def _stage_masked_aux(aux: dict, parity: int | None) -> dict:
    if parity is None or "supervision_valid" not in aux:
        return aux
    group_id = jnp.asarray(aux["group_id"]).reshape(-1)
    valid = jnp.asarray(aux["supervision_valid"]).reshape(-1)
    mask = (group_id >= 0) & (group_id % 2 == parity)
    out = dict(aux)
    out["supervision_valid"] = valid & mask
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", required=True, help="Registered TrainConfig name (e.g. pi0_draftvla_task12)")
    p.add_argument("--checkpoint-dir", required=True, help="Step dir holding params/ (and assets/)")
    p.add_argument("--val-repo-id", default="draftvla/task12_tactile_val")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-batches", type=int, default=None, help="Cap for a quick look; default = full val set")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", default=None, help="Also dump the metrics table to this file")
    args = p.parse_args()

    ckpt = pathlib.Path(args.checkpoint_dir)
    config = _config.get_config(args.config_name)
    # Same transforms/norm-stats machinery as training, pointed at the val repo. Norm stats must
    # stay the TRAIN repo's (that is correct and deliberate; never run compute_norm_stats on the
    # val repo) -- they are keyed by asset_id, which defaults to repo_id, so pin it explicitly to
    # the train repo id or the loader would look for (nonexistent) val stats.
    train_repo_id = config.data.repo_id
    config = dataclasses.replace(
        config,
        batch_size=args.batch_size,
        data=dataclasses.replace(
            config.data,
            repo_id=args.val_repo_id,
            assets=_config.AssetsConfig(asset_id=train_repo_id),
        ),
    )

    model = config.model.load(_model.restore_params(ckpt / "params", dtype=jnp.float32))
    model.eval()

    # JIT once; without this the per-op dispatch is ~100x slower and a single checkpoint blows
    # through a 2h walltime. All splits share one trace (same shapes, different mask values).
    @nnx.jit
    def eval_step(model, rng, observation, actions, aux):
        return model.compute_train_losses(rng, observation, actions, train=False, aux=aux, step=0)

    loader = _data_loader.create_data_loader(
        config, sharding=None, shuffle=False, num_batches=args.max_batches, skip_norm_stats=False
    )

    rng = jax.random.key(args.seed)
    sums: dict[str, dict[str, float]] = {name: {} for name in STAGE_SPLITS}
    weights: dict[str, dict[str, float]] = {name: {} for name in STAGE_SPLITS}
    n_batches = 0
    stage_splits = None  # decided from the first batch: phy-less arms have no masked metrics
    for batch in loader:
        observation, actions, aux = batch
        rng, step_rng = jax.random.split(rng)
        if stage_splits is None:
            _, probe = eval_step(model, step_rng, observation, actions, aux)
            stage_splits = dict(STAGE_SPLITS) if "loss_dist" in probe else {"all": None}
        for split_name, parity in stage_splits.items():
            loss, metrics = eval_step(model, step_rng, observation, actions, _stage_masked_aux(aux, parity))
            metrics = {"loss_total": loss, **metrics}
            nv = float(metrics.get("num_valid", 1.0))
            for k, v in metrics.items():
                v = float(v)
                # Masked metrics are means over valid frames; weight by num_valid so batches with
                # few labeled frames do not count as much as full ones.
                w = nv if k in _MASKED else 1.0
                sums[split_name][k] = sums[split_name].get(k, 0.0) + v * w
                weights[split_name][k] = weights[split_name].get(k, 0.0) + w
        n_batches += 1

    table: dict[str, dict[str, float]] = {}
    for split_name in stage_splits or {}:
        table[split_name] = {
            k: sums[split_name][k] / max(weights[split_name][k], 1e-9) for k in sums[split_name]
        }

    print(f"\ncheckpoint: {ckpt}\nval repo:   {args.val_repo_id}   batches: {n_batches} x {args.batch_size}\n")
    header = f"{'metric':<22}" + "".join(f"{s:>14}" for s in table)
    print(header)
    print("-" * len(header))
    for k in REPORT_KEYS:
        if not any(k in table[s] for s in table):
            continue
        row = f"{k:<22}"
        for s in table:
            v = table[s].get(k)
            if k == "loss_flow" and s != "all":
                row += f"{'--':>14}"  # unmasked by design; identical in every split
            else:
                row += f"{v:>14.4f}" if v is not None else f"{'--':>14}"
        print(row)

    dist, base = table["all"].get("loss_dist"), table["all"].get("loss_dist_baseline")
    if dist is not None and base is not None:
        verdict = "reads physical state from unseen episodes" if dist < 0.8 * base else (
            "NO better than a constant predictor -- train-set fit was memorization")
        print(f"\nval loss_dist / baseline = {dist:.4f} / {base:.4f}  -> {verdict}")

    if args.output_json:
        pathlib.Path(args.output_json).write_text(json.dumps({"checkpoint": str(ckpt), "table": table}, indent=2))
        print(f"wrote {args.output_json}")


_MASKED = frozenset(
    {"loss_dist", "loss_dist_baseline", "loss_proto", "loss_proto_excess", "proto_acc", "proto_entropy", "kl_grip"}
)

if __name__ == "__main__":
    main()
