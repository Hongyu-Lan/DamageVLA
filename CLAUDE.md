# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

openpi holds open-source robotics vision-language-action (VLA) models from Physical Intelligence: **π₀** (flow-matching VLA), **π₀-FAST** (autoregressive VLA using the FAST action tokenizer), and **π₀.₅** (improved π₀ with knowledge insulation; only the flow-matching head is supported here). The repo provides base checkpoints, fine-tuning, and inference. Models have both a **JAX** (primary, Flax NNX) and a **PyTorch** implementation.

## Objective
Extend **π₀** (flow-matching VLA) to Force-aware **π₀** following the ForceVLA design, fine-tuned from the pretrained **π₀** (`pi0_base`, VLM frozen) on our custom dataset so it also uses a force/torque input to predict actions.

**Current state:** implemented in JAX with two fusion variants — M1 (force as a single conditioning token) and M2 (faithful FVLMoE late fusion) — plus a no-force baseline for ablation (configs `pi0_force_token` / `pi0_force_fvlmoe` / `pi0_force_baseline`). Forward/loss/sample, the VLM freeze, and the pretrained-weight back-fill are unit-tested; a single-episode training smoke test passed (loss 3.6→0.13, VLM `param_norm` constant); and end-to-end serving is verified (`serve_policy.py` + `examples/force/main.py` → `(8,7)` action chunks). **Remaining:** fine-tune on the full multi-episode dataset and evaluate task success. See "Force-aware π₀ (ForceVLA extension)" under Architecture for the implementation map and `examples/force/` for run commands.

## Environment & Common Commands

Dependencies are managed with [uv](https://docs.astral.sh/uv/); Python is pinned to 3.11.

```bash
# First-time setup (submodules are required: third_party/aloha, third_party/libero)
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync          # GIT_LFS_SKIP_SMUDGE=1 is required (pulls LeRobot)
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Lint / format (CI runs these via pre-commit)
ruff check .
ruff format .
pre-commit install && pre-commit run --all-files

# Tests — mirrors CI exactly. testpaths are src/, scripts/, packages/; tests are *_test.py co-located with source.
uv run pytest --strict-markers -m "not manual"
uv run pytest src/openpi/models/pi0_test.py            # single file
uv run pytest src/openpi/models/pi0_test.py::test_name # single test
```

Tests marked `@pytest.mark.manual` are excluded by default (they are heavy / require a GPU). `src/openpi/conftest.py` forces JAX onto the CPU backend when no GPU is present.

### Run an experiment (JAX)

```bash
uv run scripts/compute_norm_stats.py --config-name <config>          # required before training a new config
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config> --exp-name=<name> --overwrite
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config> --policy.dir=checkpoints/<config>/<name>/<step>
```

`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` lets JAX use 90% of GPU memory. For multi-GPU, set `--fsdp-devices=<n>` (model-parallel FSDP; no multi-node support). Override the checkpoint cache (default `~/.cache/openpi`) with `OPENPI_DATA_HOME`.

**Multi-GPU training footgun:** `scripts/train.py` meshes over `jax.device_count()` = *every visible* GPU; `fsdp_devices=1` does **not** restrict the count (it replicates the model on each). Pin one GPU with `CUDA_VISIBLE_DEVICES`, and pick a **display-free** GPU — training on the GPU driving your monitors can crash the whole desktop to the login screen during the heavy `pi0_base` load + JIT, *before step 0* (a driver crash, not OOM). `examples/force/run_smoke_repro.sh` auto-selects a display-free GPU (`uv run examples/force/select_gpu.py` lists them) and is the one-command convert→norm-stats→train→plot reproduction of the smoke test.

## Architecture

### The config registry is the entry point for everything

`src/openpi/training/config.py` is the most important file. `_CONFIGS` is a list of `TrainConfig` objects keyed by `name`, resolved via `get_config(name)`. Training, norm-stat computation, serving, and inference are **all driven by a named config string** (e.g. `pi05_libero`). To add a new robot/dataset, you add a `TrainConfig` and (usually) a `DataConfigFactory` here. A `TrainConfig` wires together:

- **`model`** — a `BaseModelConfig` (`Pi0Config` or `Pi0FASTConfig`) carrying architecture + the shared `action_dim` / `action_horizon` / `max_token_len`.
- **`data`** — a `DataConfigFactory` that builds the data/transform pipeline (see below).
- **`weight_loader`** — loads (possibly partial) pretrained weights after init.
- optimizer, lr schedule, EMA, `fsdp_devices`, `freeze_filter` (for LoRA / frozen weights), `pytorch_weight_path`, etc.

### Data transform pipeline (training and inference share it)

Raw data is mapped to the model's normalized format through an ordered pipeline defined per-robot in `DataConfigFactory.create()` (`src/openpi/transforms.py` holds the transform primitives):

1. **`repack_transforms`** — rename dataset-specific keys into a common flat layout (`RepackTransform`).
2. **`data_transforms`** — robot-specific input/output mapping, e.g. `AlohaInputs`/`AlohaOutputs`, `LiberoInputs`/`DroidInputs` in `src/openpi/policies/*_policy.py`. These convert between the robot's action/state space and the model's; they also do things like delta-action conversion.
3. **Normalization** — `Normalize`/`Unnormalize` using `norm_stats` (z-score or quantile). Stats live in the checkpoint's `assets/` dir and are produced by `compute_norm_stats.py`.
4. **`model_transforms`** — model-specific, e.g. tokenize the prompt and resize images (`ModelTransformFactory`).

`Observation` and `Actions` (`src/openpi/models/model.py`) are the structured, normalized model I/O. Transforms emit nested dicts; `Observation.from_dict()` converts them. The model always expects image keys `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` at 224×224 in `[-1, 1]`.

### Models

`BaseModelConfig.create()` → `BaseModel` (an `nnx.Module`) exposing `compute_loss()` and `sample_actions()`. The backbone is PaliGemma = Gemma LLM (`models/gemma.py`) + SigLIP vision (`models/siglip.py`). π₀/π₀.₅ use a flow-matching action head (`models/pi0.py`); π₀-FAST is autoregressive over FAST action tokens (`models/pi0_fast.py`, `models/gemma_fast.py`). LoRA support lives in `models/lora.py`.

### Policy & remote inference

A `Policy` (`src/openpi/policies/policy.py`) wraps a model plus input/output transforms; `create_trained_policy(config, dir)` (`policies/policy_config.py`) assembles it and **auto-detects PyTorch vs JAX** by checking for `model.safetensors` in the checkpoint dir (present → PyTorch; otherwise JAX, loading Orbax `params/`) — works for both local dirs and `gs://openpi-assets`. `Policy.infer(obs)` runs transforms → `sample_actions` → transforms and returns an action chunk. Note `infer` carries `state` (and `force`, when present) into the output tree so the strict output `Unnormalize` — which covers every normalized key — succeeds.

Inference is typically **remote**: `scripts/serve_policy.py` runs the policy inside a `WebsocketPolicyServer` (`src/openpi/serving/`) on port 8000. The robot runs the lightweight **`openpi-client`** package (`packages/openpi-client/`, a separate uv workspace member with minimal deps, Python 3.7+) which connects over a websocket, streams observations, and receives actions (msgpack-numpy serialization). This keeps the GPU policy server decoupled from the robot environment. `examples/` contains per-robot integrations (`aloha_real`, `aloha_sim`, `droid`, `libero`, `ur5`, `simple_client`).

### JAX vs PyTorch

JAX is the original/primary implementation. The PyTorch implementation (`src/openpi/models_pytorch/`) is newer and **does not support** π₀-FAST, mixed-precision, FSDP, LoRA, or EMA. Using PyTorch requires patching the installed `transformers` library (copy `src/openpi/models_pytorch/transformers_replace/*` over the installed `transformers` package — see README "PyTorch Support"; note this mutates the uv cache globally). Convert a JAX checkpoint with `examples/convert_jax_model_to_pytorch.py`, then train via `scripts/train_pytorch.py` (use `torchrun` for multi-GPU DDP) and point the config's `pytorch_weight_path` at the converted weights.

### Force-aware π₀ (ForceVLA extension — implemented, JAX only)

The Objective above is realized as an **optional, backward-compatible** 6-axis force/torque modality on π₀: with it off the model is byte-for-byte vanilla π₀. The integration spans several files — to change behavior, edit the one that owns the concern:

- **`Observation.force`** (`models/model.py`) — new optional trailing field (wrench `[fx,fy,fz,tx,ty,tz]`), normalized upstream (z-score) exactly like `state`. `None` ⇒ force-awareness off. Declared **last** to keep the PyTree field order of existing fields unchanged.
- **`Pi0Config` flags** (`models/pi0_config.py`) — `force_aware`, `force_dim`, `force_fusion` (`"token"` | `"fvlmoe"`), `freeze_vlm`, and `fvlmoe_*` hparams. `get_freeze_filter()` gained a `freeze_vlm` branch that freezes the SigLIP image tower + the Gemma VLM expert while leaving the **action expert + force modules** trainable (the ForceVLA fine-tuning regime). `freeze_vlm` is decoupled from `force_aware` so the no-force baseline can freeze identically.
- **Two fusion paths** (`models/pi0.py`): **M1 `"token"`** — `force_proj` maps the wrench to one action-expert-width token appended in `embed_suffix` (its own attention block, like the state token). **M2 `"fvlmoe"`** — faithful ForceVLA late fusion (`models/fvlmoe.py`, self-attention + FFN + sparse top-1 4-expert MoE): force is fused with the **frozen** VLM prefix output, and the trailing `action_horizon` fused tokens are added as **guidance** onto the action hidden states. In `sample_actions` the guidance depends only on the prefix + force (not the denoising step), so it's computed once and reused across steps.
- **`ForceInputs`/`ForceOutputs`** (`policies/force_policy.py`) — robot transform mapping `observation/force_torque` → `force` (mirrors `LiberoInputs` otherwise); output returns the first 7 action dims. `state` layout = TCP `xyz`(3) + `rotation_vector`(3) + `gripper_width`(1); `action_7` = TCP velocity(6) + gripper target(1), already relative (**no delta conversion**).
- **`LeRobotForceDataConfig` + three configs** (`training/config.py`) — `pi0_force_baseline` / `pi0_force_token` / `pi0_force_fvlmoe` differ **only** in the force path, for a clean ablation. All init from `pi0_base` and freeze the VLM.
- **Weight back-fill** (`training/weight_loaders.py`) — `CheckpointWeightLoader.extra_missing_regex` lets new params absent from `pi0_base` (`force_proj`, `fvlmoe`) initialize from the fresh model instead of failing the pytree-equality check. `compute_norm_stats.py` adds the `"force"` stats key only when `model.force_aware`.

**Data + run commands** live in `examples/force/` (`convert_force_data_to_lerobot.py` → LeRobot repo_id `force/banana`; `README.md` has the full convert → norm-stats → train → serve flow and the exact inference observation contract). Tests: `models/fvlmoe_test.py`, `policies/force_policy_test.py`, and the force cases in `models/pi0_test.py`. **Status:** M1/M2 forward/loss/sample + freeze + back-fill are unit-tested; the 200-step single-episode FVLMoE smoke train is reproducible (`examples/force/run_smoke_repro.sh` → loss 3.6→0.13, `param_norm` constant); and **end-to-end serving is verified** (`serve_policy.py` + `examples/force/main.py` → `(8,7)` chunks). Two serving bugs were fixed along the way: the cv2/LeRobot import-order segfault in `serve_policy.py`, and a strict-`Unnormalize` failure on the input-only `force` key (now carried through in `policy.py`).

## Conventions

- Ruff with line length 120; imports are forced single-line and sorted within sections (see `[tool.ruff]` in `pyproject.toml`). `print` is allowed.
- Runtime array shape/dtype checking uses jaxtyping + beartype via `@at.typecheck` and the `at.*` aliases in `src/openpi/shared/array_typing.py`.
- `third_party/` and `src/openpi/models_pytorch/transformers_replace/` are excluded from linting.
- `scripts/train.py`, `scripts/compute_norm_stats.py`, and `scripts/serve_policy.py` import `openpi.training.data_loader` (LeRobot/PyAV) **before** any cv2-importing openpi module, fenced with `# isort: off` / `# isort: on`. Loading cv2 first segfaults when the PyAV/Arrow backend imports afterward on some systems — do not let isort/ruff reorder those lines.

## Policy & requirements
- Read the ForceVLA paper (`forceVLA.pdf` at the repo root), and follow its framework to design Force-aware **π₀**. 
- According to our custom data (`data/pi0_train_20260611_204757`), revise the codes to make the custom data fit the new designed Force-aware **π₀**. The inputs you will use in our data are "image" (`data/pi0_train_20260611_204757/rgb/*.jpg`), "wrist_image" (`data/pi0_train_20260611_204757/wrist/*.jpg`), "action_7" (`data/pi0_train_20260611_204757/observations.jsonl`), and "force_torque" (`data/pi0_train_20260611_204757/observations.jsonl`).
- Add the corresponding configs and instructions for training and inference in the newly designed Force-aware **π₀**.
- interview me to find the real goal of this project. Bias small, compartmentalized specs. Make me verify key decisions explicitly so nothing is missed.
- Outline the evaluation criteria you will use to ensure a high-quality final product. Be precise. 
- Before you start, define the precise criteria for a great result, use a past example as the format to match, and have a second AI to check the final output.

## Priority
- Give priority to ## Overview, ## Objective, and ## Policy & requirements in this readme file.