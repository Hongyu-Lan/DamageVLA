#!/usr/bin/env bash
# Run the Force-aware pi0 training-loss smoke test using settings from a config file.
#
#   bash run_smoke_with_config.sh                       # uses ./train_config.conf
#   CONFIG_FILE=my.conf bash run_smoke_with_config.sh   # use a different config file
#
# It reads train_config.conf, maps MODE -> the registered config name, resolves the
# GPU(s), translates the rest into scripts/train.py overrides, then runs:
#   compute_norm_stats -> train (logging every step) -> plot the loss curve.
set -euo pipefail
cd "$(dirname "$0")"   # repo root, so relative paths resolve from anywhere

# --- Load the config file ----------------------------------------------------
CONFIG_FILE="${CONFIG_FILE:-train_config.conf}"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "ERROR: config file '$CONFIG_FILE' not found (run from the repo root, or set CONFIG_FILE=)."
  exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG_FILE"
echo "Loaded config: $CONFIG_FILE"

# --- Map training MODE -> registered config name -----------------------------
case "${MODE:-}" in
  fvlmoe)  CONFIG=pi0_force_fvlmoe ;;
  token)   CONFIG=pi0_force_token ;;
  vanilla) CONFIG=pi0_force_baseline ;;
  *) echo "ERROR: MODE must be one of: fvlmoe | token | vanilla (got '${MODE:-}')."; exit 1 ;;
esac

UV="${UV:-uv}"
EXP="${EXP_NAME:-force_smoke_cfg}"
LOG="examples/force/train_${EXP}.log"
METRICS="checkpoints/${CONFIG}/${EXP}/metrics.csv"
OUT="${OUT:-examples/force/training_loss_${EXP}.jpg}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID          # CUDA indices == nvidia-smi indices
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # allocate on demand (gentle on a shared box)
export WANDB_MODE="${WANDB_MODE:-disabled}"  # never block on a W&B login

# --- Resolve GPU(s): explicit GPUS > auto-pick the lightest-graphics GPU ------
if [ -n "${GPUS:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
  echo "Using GPU(s) $CUDA_VISIBLE_DEVICES (from config GPUS)."
else
  GPU="$(python3 - <<'PY'
import subprocess, sys, xml.etree.ElementTree as ET
try:
    root = ET.fromstring(subprocess.run(["nvidia-smi","-q","-x"],capture_output=True,text=True,check=True).stdout)
except Exception:
    print("NONE"); sys.exit(0)
best, best_mem = None, None
for i, gpu in enumerate(root.findall("gpu")):
    procs = gpu.find("processes"); g_mem = 0
    if procs is not None:
        for p in procs.findall("process_info"):
            if (p.findtext("type") or "").strip() == "G":
                try: g_mem += int((p.findtext("used_memory") or "0 MiB").split()[0])
                except ValueError: pass
    if best_mem is None or g_mem < best_mem:
        best, best_mem = i, g_mem
print(best if best is not None else "NONE")
PY
)"
  if [ "$GPU" = "NONE" ] || [ -z "$GPU" ]; then
    echo "ERROR: could not auto-pick a GPU via nvidia-smi. Set GPUS= in $CONFIG_FILE."
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="$GPU"
  echo "Auto-selected display-free GPU $CUDA_VISIBLE_DEVICES (lightest graphics load)."
fi

# --- Preflight: JAX must see as many GPUs as we requested --------------------
WANT="$(printf '%s' "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')"
echo "==== [0/3] preflight: JAX device check (expect $WANT) ===="
NDEV="$("$UV" run python -c 'import jax; print(jax.device_count())' 2>/dev/null | tail -1)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -> JAX sees ${NDEV:-0} device(s)."
if [ "${NDEV:-0}" != "$WANT" ]; then
  echo "ERROR: expected $WANT visible GPU(s) but JAX sees '${NDEV:-0}'. Aborting before the heavy run."
  exit 1
fi
if [ "${FSDP_DEVICES:-1}" -gt 1 ] && [ $(( WANT % FSDP_DEVICES )) -ne 0 ]; then
  echo "ERROR: FSDP_DEVICES=$FSDP_DEVICES must divide the number of visible GPUs ($WANT)."
  exit 1
fi

# --- Assemble scripts/train.py overrides ------------------------------------
ARGS=(
  "$CONFIG"
  --exp-name="$EXP"
  --batch-size="$BATCH_SIZE"
  --num-train-steps="$NUM_TRAIN_STEPS"
  --fsdp-devices="${FSDP_DEVICES:-1}"
  --log-interval="${LOG_INTERVAL:-1}"
  --lr-schedule.peak-lr="$PEAK_LR"
  --lr-schedule.warmup-steps="$WARMUP_STEPS"
  --lr-schedule.decay-steps="$DECAY_STEPS"
  --lr-schedule.decay-lr="$DECAY_LR"
  --optimizer.clip-gradient-norm="$CLIP_GRAD_NORM"
  --no-wandb-enabled
  --overwrite
)
# FVLMoE-only model hyperparameters (ignored by the token / vanilla configs).
if [ "$MODE" = "fvlmoe" ]; then
  ARGS+=(
    --model.fvlmoe-num-experts="$NUM_EXPERTS"
    --model.fvlmoe-num-heads="$NUM_HEADS"
    --model.fvlmoe-mlp-ratio="$MLP_RATIO"
  )
fi

echo "==== [1/3] compute_norm_stats ($CONFIG) ===="
"$UV" run scripts/compute_norm_stats.py --config-name "$CONFIG"

echo "==== [2/3] train: MODE=$MODE config=$CONFIG batch=$BATCH_SIZE steps=$NUM_TRAIN_STEPS lr=$PEAK_LR ===="
echo "     overrides: ${ARGS[*]}"
"$UV" run scripts/train.py "${ARGS[@]}" 2>&1 | tee "$LOG"

echo "==== [3/3] plot loss from $METRICS ===="
"$UV" run examples/force/plot_training_loss.py --metrics "$METRICS" --out "$OUT"

echo "Done. config=$CONFIG exp=$EXP"
echo "  console log: $LOG"
echo "  metrics:     $METRICS"
echo "  loss plot:   $OUT"
