# Force-aware π₀ — smoke-train debug log

Small, durable record of what matters when running / re-running the FVLMoE smoke training.
Read this FIRST next time something crashes or the loss looks wrong. Updated 2026-06-19.

## Incident 1 (2026-06-19): 2-GPU OOM crash (a different multi-GPU host)

- Following `examples/force/README.md` as written ran the **config defaults (batch 16, 3000 steps)**,
  not the **smoke settings (batch 4, 200 steps)** that actually produced the published loss image.
  Batch 16 OOMs a 24 GB card here.
- Nothing pinned the job to one GPU. `fsdp_devices=1` does **not** limit the device count — it only
  turns off model sharding. `scripts/train.py` builds its mesh from `jax.device_count()` =
  *every visible GPU* (`src/openpi/training/sharding.py:22`) and data-parallels across all of them,
  replicating the full model on each → "both GPUs at ~22/24 GB".
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (from the old README) preallocates ~22 GB **per GPU** up front.
  GPU 0 here also drives the desktop (Xorg + gnome-shell), so OOM/preallocation can freeze the GUI =
  "my PC crashed".

## Incident 2 (2026-06-19): desktop crash to login screen on a 2-GPU host

- Symptom: ran `run_smoke_repro.sh` on a box with **2× RTX 3090**; the **whole session crashed back to
  the login screen** (all programs killed). This is an **X/display-driver crash, not an OOM**.
- Log evidence (`train_smoke_repro.log`, their run): reached `Restoring checkpoint from
  .../pi0_base/params` then stopped — **no `Step`/`Progress` lines at all** → it died during GPU
  weight-placement + JIT-compile, **before step 0**. No Python traceback because X (and the process)
  were hard-killed.
- Root cause: the old script pinned `CUDA_VISIBLE_DEVICES=0`, which on that host is the **display GPU**.
  Slamming the display GPU with the pi0_base load + XLA compile hung its driver → X reset. (A dual-3090
  PSU power transient can compound it.) My 1-GPU box survived the same load only because its desktop was
  idle and batch 4 is light — luck, not safety.
- Fix: **run on a display-free GPU.** `examples/force/select_gpu.py` finds which GPUs run graphics
  (type "G") processes via `nvidia-smi -q -x` (the `display_active` field is unreliable — it read
  "Disabled" on a GPU that was in fact driving the desktop). `run_smoke_repro.sh` now auto-picks a
  display-free GPU, sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` (indices match nvidia-smi), **preflight-aborts
  unless JAX sees exactly 1 device**, and refuses to start if every GPU has a display. For PSU-marginal
  dual-3090 rigs also: `sudo nvidia-smi -pm 1 && sudo nvidia-smi -i <gpu> -pl 280`.

## Fix (current procedure)

Run on a **display-free** GPU — the wrapper handles it:

```bash
bash examples/force/run_smoke_repro.sh        # auto-pick a display-free GPU
bash examples/force/run_smoke_repro.sh 1      # or pin an explicit (display-free) index
uv run examples/force/select_gpu.py           # list GPUs: which is DISPLAY vs free
```

Writes `train_smoke_repro.log`, `<ckpt>/metrics.csv`, and `training_loss_smoke_repro.jpg`.

## Machine facts (verified 2026-06-19, the box this repo lives on)

- **One** GPU only: `lspci` → a single NVIDIA `2204` (RTX 3090); `nvidia-smi` → 1× RTX 3090, 24 GB;
  **`jax.devices()` → `[CudaDevice(id=0)]`, `device_count == 1`**. No second GPU, no fallen-off-bus
  in `dmesg`.
- ✅ **2-GPU report resolved:** the crash was on a **different, multi-GPU host** (user-confirmed),
  not this 1-GPU box. There, JAX used both visible GPUs (model replicated on each) and
  `MEM_FRACTION=0.9` preallocated ~22 GB on each → OOM. On any multi-GPU host, pin GPUs explicitly
  (`CUDA_VISIBLE_DEVICES=…`); see the README "Scale up to real training" for data-parallel vs FSDP.
- Binaries not on the non-interactive PATH: use `/home/ziren2/anaconda3/bin/uv run …`.

## Is the published loss image real? (the "illusion" question)

- **The training run is real.** A complete Orbax checkpoint exists at
  `checkpoints/pi0_force_fvlmoe/force_fvlmoe_smoke/199/` (`params/`, `train_state/`,
  `assets/force/banana/norm_stats.json` whose stat keys are `state, actions, force`). Something
  genuinely trained for 200 steps.
- **But the original plot was NOT verifiable.** `_CHECKPOINT_METADATA` has `"metrics": {}`; there is
  **no `wandb/` run, no metrics CSV/JSON, and no committed plotting script**. The published
  `training_loss_smoke.jpg` was drawn by a throwaway script, so its exact per-step values could not
  be reproduced from saved artifacts. Not fabricated, but not auditable — a real defect.
- **Fix:** `train.py` already prints `Step N: loss=…, grad_norm=…, param_norm=…`. We now (a) tee that
  to `train_smoke_repro.log` and (b) regenerate the plot from it with the committed
  `examples/force/plot_training_loss.py`. The image is now reproducible from logged numbers.

## What to check in the log when debugging

1. **Device count** — `train.py` logs `Running on: <host>`; confirm only 1 device is used. A fast
   external check: `CUDA_VISIBLE_DEVICES=0 uv run python -c "import jax; print(jax.devices())"`.
2. **Divisibility assertion** (`train.py:205`): `batch_size % jax.device_count() == 0`, else it
   raises before training.
3. **VLM stayed frozen** — `param_norm` should be ~constant across steps (the plot script prints its
   spread). If `param_norm` moves a lot, the freeze filter / `freeze_vlm` is not taking effect.
4. **Loss trajectory** — single-episode overfit should fall from ~3.6 to a few tenths within 200
   steps. Flat-high loss ⇒ data/normalization/frozen-everything problem.
5. **Native segfault at import** ⇒ the cv2 / LeRobot import-order regression. `scripts/train.py` and
   `scripts/compute_norm_stats.py` must import `openpi.training.data_loader` **before** any
   cv2-importing module (kept behind `# isort: off`). Do not let isort reorder them.

## Reproduction run results (this machine, 2026-06-19) — PASSED

- env: `CUDA_VISIBLE_DEVICES=0`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, wandb disabled
- config: `pi0_force_fvlmoe`, batch 4, 200 steps, seed 42 (default), `freeze_vlm=True`
- **loss: first=3.5814  last=0.1302  min=0.1146  max=3.9718**  → falls 3.6 → ~0.13 as expected.
- **param_norm spread: 0.000488 (0.00% of initial 1389.3459)** → VLM verified frozen. (The ~5e-4
  drift is float32 rounding in the global-norm reduction, not weight updates.)
- peak GPU memory: ~17.0 GB / 24576 MiB (single GPU; no OOM, desktop unaffected)
- speed: ~1.1 s/it, ~4 min for 200 steps (after a one-time ~1 min weight-restore + compile)
- artifacts: `train_smoke_repro.log`, `metrics.csv`, `training_loss_smoke_repro.jpg` (visually
  matches the original `training_loss_smoke.jpg`), checkpoint
  `checkpoints/pi0_force_fvlmoe/force_fvlmoe_smoke_repro/199`
- **Conclusion on the "illusion":** the published image is a faithful render of a real run — the
  reproduced curve overlays it. The defect was auditability (no saved data/script), now fixed by
  `metrics.csv` + `plot_training_loss.py`.

## Serving verification (2026-06-19) — PASSED, after fixing 2 bugs

`serve_policy.py policy:checkpoint --policy.config=pi0_force_fvlmoe --policy.dir=.../199` +
`examples/force/main.py` → 5 steps, each returning `(8, 7)` action chunks; values finite
(min −1.39, max 0.11). Two real bugs were found and fixed to get there:

1. **Segfault (exit 139) on server startup** — same cv2-before-LeRobot native conflict as the
   training scripts. `serve_policy.py` lacked the import-order guard. Reproduce:
   `python -c "import openpi.transforms; import openpi.training.data_loader"` segfaults;
   swapping the order is clean. Fixed by importing `openpi.training.data_loader` first behind
   `# isort: off` in `serve_policy.py`.
2. **`ValueError: Selector key force not found in tree`** at inference — the output `Unnormalize`
   is `strict=True` over the full norm stats (`state, actions, force`), but `Policy.infer` only
   put `state` + `actions` in the output tree; `force` is input-only. Fixed in
   `src/openpi/policies/policy.py` by carrying `force` into the output tree (mirroring `state`),
   so the strict check passes; output transforms drop it. Training never hit this (no Unnormalize
   in the loss path) — it is inference-only.
