#!/usr/bin/env bash
# Reproduce the Force-aware pi0 (FVLMoE / M2) training-loss smoke test ON THIS MACHINE and
# regenerate the loss plot from the per-step metrics it logs.
#
#   bash examples/force/run_training_loss_smoke.sh        # default: GPU 1 (display-free on this box)
#   bash examples/force/run_training_loss_smoke.sh 0      # or pin another GPU index (nvidia-smi order)
#
# Why this script (vs. run_smoke_repro.sh): on this 2x RTX-3090 box BOTH GPUs have an Xorg bound,
# so the auto-picker in run_smoke_repro.sh bails. GPU 0 runs the real desktop (gnome-shell, nautilus);
# GPU 1 holds only a bare idle Xorg -> it is the safe one, so we DEFAULT to GPU 1 here. Training on
# the desktop GPU can hang its driver during the heavy pi0_base load + JIT and bounce X to the login
# screen BEFORE step 0 (a driver crash, not OOM). It also uses `uv run --no-sync` to reuse the
# already-synced openpi/.venv instead of re-resolving dependencies.
set -euo pipefail

UV="${UV:-uv}"
CONFIG=pi0_force_fvlmoe
EXP="${EXP:-force_fvlmoe_smoke}"
LOG=examples/force/train_training_loss_smoke.log
METRICS="checkpoints/${CONFIG}/${EXP}/metrics.csv"
OUT="${OUT:-examples/force/training_loss_smoke_repro.jpg}"

# CUDA indices == nvidia-smi indices.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Pin exactly one GPU (arg > preset > default 1).
export CUDA_VISIBLE_DEVICES="${1:-${CUDA_VISIBLE_DEVICES:-1}}"
# Allocate on demand (gentle on a shared box) and never block on a W&B login.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE="${WANDB_MODE:-disabled}"

echo "==== [0/3] preflight: GPU $CUDA_VISIBLE_DEVICES, JAX device check ===="
NDEV="$("$UV" run --no-sync python -c 'import jax; print(jax.device_count())' 2>/dev/null | tail -1)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -> JAX sees ${NDEV:-0} device(s)."
if [ "${NDEV:-0}" != "1" ]; then
  echo "ERROR: expected exactly 1 visible GPU but JAX sees '${NDEV:-0}'. Aborting before the heavy run."
  echo "Pin one explicitly, e.g.: bash examples/force/run_training_loss_smoke.sh 1"
  exit 1
fi

echo "==== [1/3] compute_norm_stats ($CONFIG) ===="
"$UV" run --no-sync scripts/compute_norm_stats.py --config-name "$CONFIG"

echo "==== [2/3] train ($CONFIG): batch 4, 200 steps, log every step, GPU $CUDA_VISIBLE_DEVICES ===="
"$UV" run --no-sync scripts/train.py "$CONFIG" \
  --exp-name="$EXP" \
  --batch-size=4 \
  --num-train-steps=200 \
  --log-interval=1 \
  --no-wandb-enabled \
  --overwrite 2>&1 | tee "$LOG"

echo "==== [3/3] plot loss from $METRICS ===="
"$UV" run --no-sync examples/force/plot_training_loss.py --metrics "$METRICS" --out "$OUT"

echo "Done. Console log: $LOG ; metrics: $METRICS ; plot: $OUT"
