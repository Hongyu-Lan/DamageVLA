# Force-aware π₀ (ForceVLA-style) and DraftVLA

This example extends **π₀** (flow-matching VLA) with a 6-axis **force/torque** modality, following
the [ForceVLA](../../forceVLA.pdf) design: force is **late-fused** *after* the frozen vision-language
backbone and used to guide the flow-matching action head.

**DraftVLA** (Damage-Aware ForceVLA) builds on the M2 path with a **Physical Interaction Token**
`z_phy` that drives additive action guidance plus two supervised heads — a stage-level safe force
distribution and a soft prototype classifier. Spec: [`outlines/draftvla_plan.md`](../../outlines/draftvla_plan.md) (v2.1).

| Config | Dataset | Description |
|---|---|---|
| `pi0_draftvla` | `draftvla/fruits_tactile` | **DraftVLA**: tactile wrench + FVLMoE + `z_phy` → G_phy + `L_dist` + `L_proto`. |
| `pi0_draftvla_forcevla` | `draftvla/fruits_tactile` | Ablation `forcevla`: same data + freeze, physical branch OFF. |
| `pi0_draftvla_noforce` | `draftvla/fruits_tactile` | Ablation `no-force`: vanilla π₀. |
| `pi0_force_baseline` | `force/banana` | Vanilla π₀ (force off), VLM frozen — the ablation control. |
| `pi0_force_token` | `force/banana` | **M1**: force projected to a single conditioning **token** in the action-expert suffix. |
| `pi0_force_fvlmoe` | `force/banana` | **M2**: faithful **FVLMoE** — force token + transformer encoder + sparse 4-expert top-1 MoE, added into the action guidance. |

## ⚠️ DraftVLA deployment contract: use the estimated gripper wrench

The current DraftVLA training run does **not** use the UR flange topic
`/force_torque_sensor_broadcaster/wrench`. Its per-frame 12-D wrench input is:

```python
mean = 0.5 * (left_estimated_wrench + right_estimated_wrench)
difference = 0.5 * (left_estimated_wrench - right_estimated_wrench)
gripper_wrench = np.concatenate([mean, difference])
```

where both inputs use `[fx, fy, fz, tx, ty, tz]`. In the uploaded data they come from
`/tactile/left_finger/estimated_wrench` and `/tactile/right_finger/estimated_wrench`. A robot client
must use the same estimator, component order, coordinate convention and units, then send:

```python
left = read_left_estimated_wrench()    # (6,)
right = read_right_estimated_wrench()  # (6,)
obs["observation/gripper_wrench"] = np.concatenate([0.5 * (left + right), 0.5 * (left - right)])
```

Do not send the flange wrench under this key. The raw dataset does not record a ROS `frame_id` or an
explicit units field for the tactile estimates, so deployment must inherit those conventions from the
estimator that produced the training topics; verify them before commanding the real robot.

All three **freeze the PaliGemma VLM** and train only the action expert (+ force modules), initialized
from the released `pi0_base` checkpoint.

> ⚠️ **Read this before training — on a multi-GPU machine, train on a DISPLAY-FREE GPU.** Two traps:
> 1. **GPU count.** `fsdp_devices=1` does *not* limit the number of GPUs — `scripts/train.py` meshes
>    over `jax.device_count()` = *every visible* GPU (replicating the model on each). Pin one GPU.
> 2. **Which GPU.** Training on the GPU that drives your monitors can hang its driver during the
>    heavy `pi0_base` load + JIT-compile and **crash the whole desktop back to the login screen —
>    before step 0** (this is a display/driver crash, not an out-of-memory error). Pick a GPU with
>    **no display attached**.
>
> `run_smoke_repro.sh` handles both automatically: it picks a display-free GPU (via `select_gpu.py`),
> orders indices like `nvidia-smi` (`CUDA_DEVICE_ORDER=PCI_BUS_ID`), and aborts unless exactly one GPU
> is visible. List your GPUs anytime with `uv run examples/force/select_gpu.py`.

## Data

The model consumes, per timestep:

- `image` (base camera) and `wrist_image`, 224×224×3
- `state` (7) = TCP position `xyz` (3) + TCP `rotation_vector` (3) + `gripper_width` (1)
- wrench input: DraftVLA uses 12-D `[mean_fx..mean_tz, difference_fx..difference_tz]` (z-score
  normalized), while legacy `pi0_force_*` configs use the 6-D flange `force_torque`.
- `actions` (7) = `action_7` = TCP velocity (6) + gripper target (1) — already relative, **no delta conversion**
- `prompt` (language instruction)

The raw dataset is a folder containing `observations.jsonl` + `rgb/*.jpg` + `wrist/*.jpg`
(see `data/pi0_train_20260611_204757`). The converter turns **every** `observations.jsonl` found
under `--data_dir` into one LeRobot episode, so adding more episode folders later needs no code change.

### DraftVLA data (`draftvla/fruits_tactile`)

`../DamageVLA_training_post_process_20260821` — 26 episodes, 20,344 frames, 3 fruits
(potato/pear/banana), 10 Hz. Same model inputs as above, plus per-frame supervision:
The model input `gripper_wrench` is the per-frame 12-D `[two-finger mean, signed half-difference]`.
`gt_safe_distribution` deliberately remains 12-D: `[mu_mean×6, sigma_mean×6]` of the aggregate mean
wrench for this (fruit, stage) group. It is accompanied by `soft_prototype_target` (4),
`supervision_valid` (bool), and `group_id` (int32).

Only `supervision_valid` frames contribute to `L_dist` / `L_proto`. Note that **all 26 episodes carry
non-null group labels in the supervised stages**, including the 4 failed episodes — so
masking on `supervision_valid` does *not* exclude them. The converter's **episode denylist** is the
only thing that does, and it is on by default.

```bash
# Validate the uploaded JSONL and every referenced 224x224 image.
uv run examples/force/validate_draftvla_data.py

# Rebuild the group labels + prototypes (optional; verifies the shipped labels).
uv run examples/force/build_safe_group_prototypes.py --verify-parity

# Convert (22 successful episodes; the 4 failed ones are skipped by default).
uv run examples/force/convert_draftvla_data_to_lerobot.py --repo-id draftvla/fruits_tactile

# Train (reads train_config.conf; MODE=draftvla).
bash run_train.sh
```

Images in `rgb/` and `wrist/` are **already 224×224** — the converter must not re-crop them.
Originals live in `rgb_full/` / `wrist_full/`.

## Setup

```bash
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## Reproduce the training smoke test (and its loss image)

The published loss curve `training_loss_smoke.jpg` is an **overfit of the single `grasp the banana`
episode**: `pi0_force_fvlmoe`, **batch 4, 200 steps, on one display-free GPU** (also the config
defaults). One command — it selects a display-free GPU, runs convert→norm-stats→train, and
regenerates the plot:

```bash
bash examples/force/run_smoke_repro.sh        # auto-pick a display-free GPU
bash examples/force/run_smoke_repro.sh 1      # or pin an explicit GPU index (nvidia-smi order)
```

If every GPU has a monitor attached, it **refuses to start** (rather than risk crashing X) and tells
you how to proceed. Inspect your GPUs first with:

```bash
uv run examples/force/select_gpu.py           # lists each GPU as DISPLAY or free, and recommends one
```

…or run the steps explicitly (set `<gpu>` to a display-free index from `select_gpu.py`):

```bash
# 0. (once) Convert the raw JSONL+JPEG dataset to LeRobot format (-> ~/.cache/huggingface/lerobot/force/banana).
uv run examples/force/convert_force_data_to_lerobot.py --data_dir data --repo_id force/banana

# 1. Norm stats (adds a "force" key alongside state/actions for the force-aware configs).
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
  uv run scripts/compute_norm_stats.py --config-name pi0_force_fvlmoe

# 2. Train: ONE display-free GPU, batch 4, 200 steps, log every step. Allocate GPU memory on demand
#    (XLA_PYTHON_CLIENT_PREALLOCATE=false), so it does not grab ~90% up front.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run scripts/train.py pi0_force_fvlmoe --exp-name=force_fvlmoe_smoke_repro \
  --batch-size=4 --num-train-steps=200 --log-interval=1 --no-wandb-enabled --overwrite

# 3. Plot the loss from the saved metrics.csv (verifiable — drawn from logged numbers).
uv run examples/force/plot_training_loss.py \
  --metrics checkpoints/pi0_force_fvlmoe/force_fvlmoe_smoke_repro/metrics.csv \
  --out examples/force/training_loss_smoke_repro.jpg
```

**What you should see** (single-episode overfit): flow-matching loss falls from ~3.6 to a few tenths
within 200 steps, while `param_norm` stays ~constant (the frozen VLM dominates the kernel norm, so a
flat `param_norm` confirms the freeze worked). The plot script prints both, e.g.
`loss: first=3.6 … last≈0.2` and `param_norm: … spread ≈ 0% of initial`. Exact per-step values are
not bit-reproducible across GPUs/driver versions, but the shape is.

> **Why a separate `metrics.csv`?** `scripts/train.py` now writes `<checkpoint_dir>/metrics.csv`
> (`step,loss,grad_norm,param_norm`). The console progress bar (`tqdm_loggable`) does **not** emit
> parseable per-step loss, so the CSV is the source of truth for the curve. This is what makes the
> loss image auditable rather than hand-drawn.

## Memory / GPU notes

- **Use a display-free GPU** (see the warning above). Training on the GPU that renders your desktop
  can hang the driver during weight-load/compile and bounce you to the login screen — even before
  step 0 and even with VRAM to spare. `select_gpu.py` finds one; `run_smoke_repro.sh` enforces it.
- **One GPU at a time.** On a 24 GB card, `pi0` with the frozen VLM at **batch 4** uses ~17 GB.
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` allocates on demand; on a *headless* server you may instead
  cap with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8` (avoid `0.9` on a GPU that drives a display). Batch 16
  (an earlier default) OOMs a 24 GB card.
- **Dual-RTX-3090 desktops:** two 3090s spiking together can trip a marginal PSU (instant power-off /
  back to the login screen). We use only one GPU; if you still get hard power-offs, power-limit first
  (persists until reboot): `sudo nvidia-smi -pm 1 && sudo nvidia-smi -i <gpu> -pl 280`.
- `batch_size` must be divisible by the number of *visible* devices (a `train.py` assertion).

## Scale up to real training (incl. multi-GPU)

The configs are smoke-sized (batch 4 / 200 steps) to overfit the one provided episode. For real
multi-episode training: add episode folders, re-run the converter + `compute_norm_stats`, then raise
`--batch-size` / `--num-train-steps` (ForceVLA used ~10k steps single-task).

**On a multi-GPU host, choose the GPUs explicitly** — JAX otherwise uses *every visible* device
(this is what caused the original 2-GPU OOM crash):

```bash
# A) Stay on ONE GPU (safest; reproduces the smoke test):
CUDA_VISIBLE_DEVICES=0 uv run scripts/train.py pi0_force_fvlmoe --exp-name=run [other overrides]

# B) Data-parallel across 2 GPUs — the full model is REPLICATED on each (so it must fit on one GPU);
#    batch_size must be divisible by the device count:
CUDA_VISIBLE_DEVICES=0,1 uv run scripts/train.py pi0_force_fvlmoe --exp-name=run --batch-size=8

# C) Shard the model across 2 GPUs with FSDP (lower per-GPU memory, enables bigger batches):
CUDA_VISIBLE_DEVICES=0,1 uv run scripts/train.py pi0_force_fvlmoe --exp-name=run --fsdp-devices=2 --batch-size=8
```

On a **headless training server** (no desktop on the GPU) `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` is
safe; the preallocation/GUI-freeze caveat above only applies to a GPU that also drives a display.

## Serve & infer

Serving also loads the model and JIT-compiles on the GPU, so pin a **display-free** GPU here too
(same reason as training):

```bash
# Serve a trained checkpoint (pick the config you trained). <gpu> = a display-free index (select_gpu.py).
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_force_fvlmoe \
  --policy.dir=checkpoints/pi0_force_fvlmoe/force_fvlmoe_smoke_repro/199

# (separate terminal) Send observations to the server.
uv run examples/force/main.py
```

### Inference observation contract

The robot client sends an observation dict with these exact keys (the policy server applies the
`ForceInputs` → normalization → tokenization pipeline internally):

```python
{
    "observation/image":        uint8[224, 224, 3],
    "observation/wrist_image":  uint8[224, 224, 3],
    "observation/state":        float[7],   # [tcp_pos_xyz(3), tcp_rotvec(3), gripper_width(1)]
    # [mean_fx..mean_tz, difference_fx..difference_tz], difference=0.5*(left-right)
    "observation/gripper_wrench": float[12],
    "prompt":                   str,
}
```

`policy.infer(obs)["actions"]` returns an `(action_horizon, 7)` chunk of TCP velocities + gripper
target. The `state`/`gripper_wrench` layouts must match the conversion script exactly.

## How force is wired in (for reference)

- `Observation.force` (`src/openpi/models/model.py`) — new optional field; `None` ⇒ vanilla π₀.
- `src/openpi/models/fvlmoe.py` — the FVLMoE module (M2).
- `src/openpi/models/pi0.py` — `force_proj` + M1 token in `embed_suffix`; M2 additive guidance in
  `compute_loss` / `sample_actions`.
- `src/openpi/models/pi0_config.py` — input `force_dim` and independent output `safe_force_dim`, plus
  `force_aware` / `force_fusion` / `freeze_vlm` flags and freeze filter.
- `src/openpi/policies/force_policy.py` — `ForceInputs` / `ForceOutputs`.
- `src/openpi/training/config.py` — `LeRobotForceDataConfig` + the three configs above.

## Troubleshooting

See `examples/force/SMOKE_REPRO_LOG.md` for a durable debug log: the machine's verified GPU facts,
the exact incident root-cause, and a checklist of what to inspect in the training output when a run
crashes or the loss looks wrong.
