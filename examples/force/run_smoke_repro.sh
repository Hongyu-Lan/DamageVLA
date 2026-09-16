#!/usr/bin/env bash
# Reproduce the Force-aware pi0 (FVLMoE / M2) training smoke test on a SINGLE, DISPLAY-FREE GPU,
# producing a per-step training log and a verifiable loss plot regenerated from that log.
#
# Run from the repo root:
#   bash examples/force/run_smoke_repro.sh        # auto-pick a display-free GPU
#   bash examples/force/run_smoke_repro.sh 1      # or pin an explicit GPU index (nvidia-smi order)
#
# WHY a specific GPU matters: training on the GPU that drives your monitors can hang its driver
# during the heavy pi0_base load + JIT-compile and crash the whole X session (back to the login
# screen) -- before training even starts. On a multi-GPU box we therefore run on a GPU with NO
# display attached. See examples/force/README.md and SMOKE_REPRO_LOG.md.
set -euo pipefail

# `uv` may not be on a non-interactive PATH; allow overriding with `UV=/abs/path/uv bash ...`.
UV="${UV:-uv}"

CONFIG=pi0_force_fvlmoe
EXP="${EXP:-force_fvlmoe_smoke_repro}"
LOG=examples/force/train_smoke_repro.log
METRICS="checkpoints/${CONFIG}/${EXP}/metrics.csv"   # written by train.py
OUT=examples/force/training_loss_smoke_repro.jpg

# Make CUDA device indices match `nvidia-smi` indices, so "GPU 1" means the same thing everywhere.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# --- Choose ONE display-free GPU: explicit arg > preset CUDA_VISIBLE_DEVICES > auto-pick ----------
if [ "${1:-}" != "" ]; then
  export CUDA_VISIBLE_DEVICES="$1"
  echo "Using GPU $CUDA_VISIBLE_DEVICES (command-line argument)."
elif [ "${CUDA_VISIBLE_DEVICES:-}" != "" ]; then
  echo "Using GPU $CUDA_VISIBLE_DEVICES (preset CUDA_VISIBLE_DEVICES)."
else
  GPU="$("$UV" run examples/force/select_gpu.py --quiet || echo NONE)"
  if [ "$GPU" = "NONE" ] || [ -z "$GPU" ]; then
    echo "ERROR: no display-free GPU found. Training on the GPU that drives your desktop can crash X."
    echo "Run 'uv run examples/force/select_gpu.py' for options, then pass an index explicitly, e.g.:"
    echo "  bash examples/force/run_smoke_repro.sh 1"
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="$GPU"
  echo "Auto-selected display-free GPU $CUDA_VISIBLE_DEVICES (override with: bash $0 <index>)."
fi

# Allocate GPU memory on demand instead of preallocating ~90% up front (gentler on a shared box).
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Never block on a Weights & Biases login for a smoke test.
export WANDB_MODE="${WANDB_MODE:-disabled}"

# --- Preflight: confirm exactly ONE GPU is visible to JAX (catches a pin that did not take) -------
echo "==== [0/3] preflight: JAX device check ===="
NDEV="$("$UV" run python -c 'import jax; print(jax.device_count())' 2>/dev/null | tail -1)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -> JAX sees ${NDEV:-0} device(s)."
if [ "${NDEV:-0}" != "1" ]; then
  echo "ERROR: expected exactly 1 visible GPU but JAX sees '${NDEV:-0}'. Aborting BEFORE the heavy run"
  echo "to avoid using the display GPU / both GPUs (which can crash the desktop). Pin one explicitly:"
  echo "  bash examples/force/run_smoke_repro.sh <display-free index>"
  exit 1
fi

echo "==== [1/3] compute_norm_stats ($CONFIG) ===="
"$UV" run scripts/compute_norm_stats.py --config-name "$CONFIG"

echo "==== [2/3] train ($CONFIG): batch 4, 200 steps, log every step, GPU $CUDA_VISIBLE_DEVICES ===="
"$UV" run scripts/train.py "$CONFIG" \
  --exp-name="$EXP" \
  --batch-size=4 \
  --num-train-steps=200 \
  --log-interval=1 \
  --no-wandb-enabled \
  --overwrite 2>&1 | tee "$LOG"

echo "==== [3/3] plot loss from $METRICS ===="
"$UV" run examples/force/plot_training_loss.py --metrics "$METRICS" --out "$OUT"

echo "Done. Console log: $LOG ; metrics: $METRICS ; plot: $OUT"
