> **部分内容已过时（2026-09-16）。**
> 本文描述的训练契约（输入 12 维、目标 6 维、四阶段分组）已被 `outlines/todo_training_contract.md` 取代：
> 输入改为 57 维（原始触觉电压 + 开度 + 已归零的末端力），目标改为 1 维标量，
> 分布目标按 grasp / hold 两组统计。架构与文件地图部分仍然有效。

# DraftVLA — Implementation Summary

**DraftVLA (Damage-Aware ForceVLA)** extends π₀ (flow-matching VLA) with a force/torque
modality and a supervised *Physical Interaction Token* `z_phy`. It builds on the ForceVLA
FVLMoE late-fusion path (`force_fusion="fvlmoe"`) and adds a physical branch that (a) injects
additive action guidance `G_phy` into the flow head and (b) predicts a stage-level **safe
wrench distribution** and a **soft prototype** over damage-aware (fruit, stage) groups.

The current 2026-08-25 training contract supersedes plan v2.1's flange-input decision: it uses the per-frame 12-D
`[left/right estimated fingertip-wrench mean, signed half-difference]` as the model input. Its
stage-level physical labels remain the 12-D `[mu_mean(6), sigma_mean(6)]` aggregate distribution.
Everything below is optional and backward-compatible: with `force_aware=False` /
`phy_enabled=False` the model is byte-for-byte vanilla π₀.

The new tactile-input Leonardo run has not started yet. The checked-in
`examples/force/train_draftvla_{smoke,full}.log` files and `assets/pi0_draftvla/draftvla/fruits/`
belong to the older 6-D `draftvla/fruits` input contract. They are useful only as historical
diagnostics: do not resume their checkpoint or reuse their norm stats for `draftvla/fruits_tactile`.

---

## 1. File map — where each concern lives

| Concern | File | What it owns |
|---|---|---|
| Force field on model I/O | `src/openpi/models/model.py` | `Observation.force` — optional conditioning vector; DraftVLA uses `[mean(6), difference(6)]`; `None` ⇒ force off. |
| Architecture flags | `src/openpi/models/pi0_config.py` | Input `force_dim`, independent output `safe_force_dim`, `force_aware`, `force_fusion`, `freeze_vlm`, `fvlmoe_*`, and the `phy_*` hparams + validation + `get_freeze_filter()`. |
| FVLMoE fusion block | `src/openpi/models/fvlmoe.py` | Self-attn + FFN + top-1 4-expert MoE; fuses frozen VLM prefix with the force token. Returns guidance (+ pre-projection hidden when `return_hidden=True`). |
| Physical branch | `src/openpi/models/physical.py` | `z_phy` projector + 3 heads (`PhysicalActionProjector`, `SafeForceDistributionHead`, `PrototypeClassifier`) + KL / entropy / masked-mean helpers. |
| Model wiring | `src/openpi/models/pi0.py` | `force_proj`, `fvlmoe`, `phy_*` modules; `_force_guidance()`; injection in `compute_loss` (train) and `sample_actions` (inference); `compute_train_losses()` (total loss + metrics). |
| Robot transforms | `src/openpi/policies/draftvla_policy.py` | `observation/gripper_wrench → force`; carries supervision label keys (train only). |
| Force transforms (base) | `src/openpi/policies/force_policy.py` | `ForceInputs`/`ForceOutputs` — `observation/force_torque → force`, output = first 7 action dims. |
| Data config + registry | `src/openpi/training/config.py` | `LeRobotDraftVLADataConfig` + configs `pi0_draftvla`, `pi0_draftvla_forcevla`, `pi0_draftvla_noforce`. |
| Aux label routing | `src/openpi/training/data_loader.py` | `AUX_KEYS` split into a separate `aux` dict so labels never enter the model input PyTree. |
| Weight back-fill | `src/openpi/training/weight_loaders.py` | `extra_missing_regex` lets new params (`force_proj`, `fvlmoe`, `phy_*`) init fresh from a `pi0_base` load. |
| Training loop | `scripts/train.py` | Passes `aux` + `step` into `compute_train_losses`; writes `metrics.csv`. |
| Diagnostics | `src/openpi/training/summary.py` | Post-run `training_summary.txt` explaining *why* each loss did/didn't move. |
| Data conversion | `examples/force/convert_draftvla_data_to_lerobot.py` | JSONL+JPEG → LeRobot `draftvla/fruits_tactile`; episode denylist + `[mean, signed half-difference]` wrench. |
| Prototypes/labels | `examples/force/build_safe_group_prototypes.py` | Offline K-means (fixed, non-trainable) prototypes + group labels. |
| Docs / run flow | `examples/force/README.md` | Full convert → norm-stats → train → serve flow + inference contract. |

---

## 2. Data pipeline

Source dataset: `../DamageVLA_training_post_process_20260821` (26 episodes, 20,344 frames, UR5e
pick & place over 3 fruits, per-frame damage-aware labels).

**Conversion** (`convert_draftvla_data_to_lerobot.py` → LeRobot `draftvla/fruits_tactile`) does two
steps beyond the plain force converter:

1. **Episode denylist** — 4 failed episodes are removed at the
   *episode* level. Masking on `supervision_valid` does **not** exclude them; only the denylist
   does. Denylisted episodes are never written.
2. **Tactile input** — `gripper_wrench = concat(0.5 × (left_estimated + right_estimated),
   0.5 × (left_estimated - right_estimated))`. The raw UR flange `force_torque` remains in the JSONL
   but does not enter this training dataset. Safe labels/prototypes still describe the mean half only.

Images in `rgb/` and `wrist/` are already 224×224 — the converter must not re-crop/resize.

**Transform pipeline** (`LeRobotDraftVLADataConfig.create`, shared by train + inference):

1. `RepackTransform` — dataset keys → common `observation/*` layout (`image`, `wrist_image`,
   `state`, `gripper_wrench`, `actions`, `prompt`, + label keys).
2. `DraftVLAInputs` — wraps `ForceInputs` (state=TCP xyz(3)+rotvec(3)+gripper(1); force(12);
   two RGB views), then passes through the four label keys **when present**.
3. `Normalize` — z-score on `state`, `force`, `actions` via `norm_stats`
   (`compute_norm_stats.py` adds the `"force"` key only when `model.force_aware`). Label keys
   are **not** normalized here — the safe distribution is normalized inside the model.
4. `ModelTransformFactory` — tokenize prompt, resize images to 224×224 in `[-1,1]`.

**Supervision label keys** (`draftvla_policy.AUX_KEYS`, present in training only):

| Key | Shape | Meaning |
|---|---|---|
| `gt_safe_distribution` | `[12]` | Stage-level safe wrench dist: `[mu×6, sigma×6]`. |
| `soft_prototype_target` | `[4]` | Soft target over K=4 offline prototypes. |
| `supervision_valid` | `()` bool | 56.4% of kept frames carry a physical label. |
| `group_id` | `()` int32 | Logging only — never enters the model. |

`data_loader.py` splits these into a separate `aux` dict (`yield Observation, actions, aux`).
`Observation.from_dict` ignores unknown keys, so `group_id` provably never reaches the model.

---

## 3. Model / forward design

### 3.1 FVLMoE late fusion (`fvlmoe.py`)
```
E_in    = concat([vl_tokens, force_token], axis=1)     # force_token = force_proj(force) at VLM width
x       = self_attention(E_in) + residual              # force attends to all V-L context
x       = ffn(x) + residual
h       = moe(x) + residual                            # sparse top-1, 4 experts
out     = out_proj(h)                                  # → action-expert width
```
- **Guidance** = trailing `action_horizon` tokens of `out`, added elementwise onto the action
  hidden states (ForceVLA "final H_action tokens from E_FVLMoE").
- **`h[:, -1, :]`** (pre-projection, VLM width) = the appended force token after attending to
  the whole prefix → the *contextualized force token* feeding `z_phy`.
- Late fusion (after the frozen VLM) is essential — ForceVLA's ablation shows early fusion
  collapses to 0% success.

### 3.2 Physical branch (`physical.py`, plan §6.2)
`_force_guidance()` computes, from the FVLMoE hidden, in one place shared by train + inference:

- `z_phy = PhysicalProjector(h[:, -1, :])` → L2-normalized direction (`phy_dim=128`).
- `G_phy = PhysicalActionProjector(z_phy)` → `[b, T, d_act]` additive action guidance
  (**this is what makes the branch affect action generation** — plan rule 3).
- `(mu_norm, sigma_norm) = SafeForceDistributionHead(z_phy)` → stage-level `[b,6]` each, in
  **normalized** label space; de-normalized to `mu_pred/sigma_pred` with **one shared per-dim
  scale** (KL invariant under the affine change of variable — do **not** use separate scales).
- `proto_logits = PrototypeClassifier(z_phy)` → `[b, K=4]` over **fixed offline** prototypes
  (K-means centers are metadata, never trainable — plan rule 8).

Both ForceVLA guidance (`G_fvl`) and `G_phy` are added at the action hidden states; a
`g_phy_rel = ||G_phy|| / ||G_fvl||` diagnostic tracks whether the physical token actually moves
the action head.

### 3.3 Two other force paths (present, not used by `pi0_draftvla`)
- **M1 `"token"`** — `force_proj` → one action-expert-width token appended in `embed_suffix`
  (its own attention block, like the state token).
- **Baseline** — `force_aware=False`, vanilla π₀ with the VLM still frozen.

### 3.4 Freeze regime (`get_freeze_filter`, `freeze_vlm=True`)
Freezes the SigLIP image tower (`.*img.*`) + the Gemma VLM expert (gemma params that are **not**
the action expert `_1`). Trainable: action expert, `force_proj`, `fvlmoe`, and all `phy_*`
modules. `freeze_vlm` is decoupled from `force_aware` so ablations freeze identically.

---

## 4. Training logic

Entry: `compute_train_losses()` (`pi0.py`) → `(total_scalar_loss, metrics)`.

```
loss_flow  = mean(per-token flow-matching loss)
loss_dist  = masked_mean( KL(p_gt || p_pred) over 6 wrench dims , supervision_valid )   # normalized space
loss_proto = masked_mean( soft-label cross-entropy(proto_logits, soft_target) , supervision_valid )
total      = loss_flow + lambda_dist * loss_dist + lambda_proto(step) * loss_proto
```

- **Masking** — 43.6% of kept frames are unlabeled; `masked_mean` averages over `supervision_valid`
  only and is safe when a step has zero valid frames. Unlabeled frames get a uniform prototype
  target and are masked out of both physical losses.
- **`lambda_proto` ramp** (plan §7.5 phase 2) — linear `0 → lambda_proto` over
  `phy_proto_ramp_steps` (0 in `pi0_draftvla`, i.e. constant). Phase 1 = `L_flow + L_dist`.
- **KL** is forward `KL(gt || pred)` (mode-covering / moment matching), with a `SIGMA_FLOOR=1e-4`
  guard and a `where` to avoid `0·inf` NaNs. Both mu and sigma normalized by the **same**
  `phy_label_scale`; de-normalized only for reporting `mu_pred`/`sigma_pred`.
- **Diagnostics** logged per step: `z_phy_cos` (collapse: ~1 = all samples one direction),
  `z_phy_norm` (~1 by construction), `loss_dist_baseline` (constant μ=0,σ=1 predictor — ratio
  <1 beats it), `proto_entropy` (~log K = uniform/collapsed), `proto_acc`, `loss_proto_excess`
  (over target entropy `H(y)`), per-dim `kl_{fx..tz}`, and `g_phy_rel`.

`scripts/train.py` threads `aux` + `state.step` into the loss, writes every metric to
`<checkpoint_dir>/metrics.csv`, and at the end `summary.py` writes `training_summary.txt`.

**Configs** (all init from `pi0_base`, freeze the VLM, `action_dim=32`, `action_horizon=8`):

| Config | Force | Physical branch | Role |
|---|---|---|---|
| `pi0_draftvla` | FVLMoE | **ON** (`phy_enabled`, `lambda_dist=1.0`, `lambda_proto=0.05`) | Main model. |
| `pi0_draftvla_forcevla` | FVLMoE | OFF | Ablation: fusion but no physical supervision. |
| `pi0_draftvla_noforce` | none | OFF | Ablation: vanilla π₀ on the same data. |

> Registry defaults are smoke-sized (`batch_size=4`, `num_train_steps=200`). The Leonardo full-run
> target is `batch=16`, `steps=10000`, supplied by `train_config.conf` through `run_train.sh`.

**Run flow:**
```bash
# 1. Build fixed prototypes + labels (verifies bit-for-bit against shipped labels)
uv run examples/force/build_safe_group_prototypes.py --verify-parity
# 2. Convert to LeRobot (episode denylist + tactile mean/difference input applied here)
uv run examples/force/convert_draftvla_data_to_lerobot.py --repo-id draftvla/fruits_tactile
# 3. Norm stats (required before a new config)
uv run scripts/compute_norm_stats.py --config-name pi0_draftvla
# 4. Train (pin one display-free GPU; see select_gpu.py)
CUDA_VISIBLE_DEVICES=<gpu> XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi0_draftvla --exp-name=draftvla_full --batch-size=16 --num-train-steps=10000
```

**Full-run status:** not started for the 12-D tactile-input dataset. The loss values in the old
checked-in log were produced with the incompatible 6-D `draftvla/fruits` contract and are not a
baseline for the Leonardo run. The new run will write its authoritative `metrics.csv` and
`training_summary.txt` under `checkpoints/pi0_draftvla/draftvla_full/`.

---

## 5. Inference logic

`sample_actions()` (`pi0.py`):
1. Preprocess observation, embed prefix, fill the KV cache with one VLM forward pass.
2. **If** `force_aware and force_fusion=="fvlmoe" and observation.force is not None`: compute
   `guidance = _force_guidance(prefix_out, force)` **once** (it depends only on prefix + force,
   not the denoising step).
3. Flow-matching denoising loop (`num_steps=10`, t: 1→0). Each step adds `guidance` (which
   already includes `G_phy`) onto the action hidden before `action_out_proj`.
4. Returns the `(action_horizon, action_dim)` action chunk.

> Note: `sample_actions` returns **actions only**. `mu_pred` / `proto_probs` are not emitted at
> inference — `DraftVLAOutputs` only attaches `safe_force_distribution` / `prototype_probs`
> *if* those keys are present, so the served policy currently returns just the `(8, 7)` action
> chunk (TCP velocity×6 + gripper target). The physical heads shape training gradients but are
> not part of the serving output as wired today.

**Serving** (JAX/Orbax checkpoint — auto-detected because there is no `model.safetensors`):
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_draftvla \
  --policy.dir=checkpoints/pi0_draftvla/draftvla_full/<step>
# separate terminal
uv run examples/force/main.py
```
After training, point `--policy.dir` at a **step directory** (for a completed 10k run this is
normally `.../9999`), not `params/` or any single file;
`create_trained_policy` loads the Orbax `params/` tree and the `assets/` norm-stats under it.

**Inference observation contract** (client → server; `DraftVLAInputs`→normalize→tokenize runs
server-side):
```python
{
  "observation/image":        uint8[224,224,3],
  "observation/wrist_image":  uint8[224,224,3],
  "observation/state":        float[7],   # [tcp_pos_xyz(3), tcp_rotvec(3), gripper_width(1)]
  # [mean_fx..mean_tz, difference_fx..difference_tz]
  "observation/gripper_wrench": float[12],
  "prompt":                   str,
}
```
`policy.infer(obs)["actions"]` → `(8, 7)`. `state`/`gripper_wrench` layouts must match the
converter exactly. The client must compute the same mean and signed left-minus-right half-difference
used during training.

---

## 6. Weight back-fill

New params (`force_proj`, `fvlmoe`, `phy_proj`, `phy_action_proj`, `phy_dist_head`,
`phy_proto_cls`) are absent from `pi0_base`. `CheckpointWeightLoader.extra_missing_regex`
(e.g. `.*(force_proj|fvlmoe|phy_proj|phy_action_proj|phy_dist_head|phy_proto_cls).*` for
`pi0_draftvla`) lets those initialize from the freshly constructed model instead of failing the
pytree-equality check; the ablation configs use narrower regexes matching only their own extra
params.

---

## 7. Tests

- `src/openpi/models/fvlmoe_test.py` — FVLMoE forward/shapes.
- `src/openpi/models/physical_test.py` — physical heads + KL/entropy helpers.
- `src/openpi/policies/force_policy_test.py` — force transforms.
- Force/physical cases in `src/openpi/models/pi0_test.py` — forward/loss/sample, freeze,
  back-fill.
Run: `uv run pytest --strict-markers -m "not manual"`.
