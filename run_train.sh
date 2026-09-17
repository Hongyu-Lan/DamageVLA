#!/usr/bin/env bash
# Train DraftVLA / Force-aware pi0 using settings from a config file.
#
#   bash run_train.sh                       # uses ./train_config.conf
#   CONFIG_FILE=my.conf bash run_train.sh   # use a different config file
#   SKIP_NORM_STATS=1 bash run_train.sh     # reuse existing norm stats (they are per-dataset)
#
# Reads train_config.conf, maps MODE -> the registered config name, resolves the GPU(s), translates
# the rest into scripts/train.py overrides, then runs: compute_norm_stats -> train.
#
# All losses (loss_flow / loss_dist / loss_proto / loss_proto_excess / proto_acc / ...) are logged
# every LOG_INTERVAL steps to TensorBoard, to metrics.csv, and to the console. Watch them live with:
#   tensorboard --logdir checkpoints/<config>/<exp>/tensorboard
# The script prints the exact command at startup.
#
# Spec: outlines/draftvla_plan.md (v2.1).
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
  task12)            CONFIG=pi0_draftvla_task12 ;;
  task12_forcevla)   CONFIG=pi0_draftvla_task12_forcevla ;;
  task12_noforce)    CONFIG=pi0_draftvla_task12_noforce ;;
  draftvla)          CONFIG=pi0_draftvla ;;
  draftvla_20260905) CONFIG=pi0_draftvla_20260905 ;;
  draftvla_20260905_20260911) CONFIG=pi0_draftvla_20260905_20260911 ;;
  draftvla_forcevla) CONFIG=pi0_draftvla_forcevla ;;
  draftvla_noforce)  CONFIG=pi0_draftvla_noforce ;;
  fvlmoe)            CONFIG=pi0_force_fvlmoe ;;
  token)             CONFIG=pi0_force_token ;;
  vanilla)           CONFIG=pi0_force_baseline ;;
  *) echo "ERROR: MODE must be one of: task12 | task12_forcevla | task12_noforce | draftvla | draftvla_20260905 | draftvla_20260905_20260911 | draftvla_forcevla | draftvla_noforce | fvlmoe | token | vanilla (got '${MODE:-}')."; exit 1 ;;
esac

UV="${UV:-uv}"
EXP="${EXP_NAME:-draftvla}"
LOG="examples/force/train_${EXP}.log"
CKPT_DIR="checkpoints/${CONFIG}/${EXP}"
METRICS="${CKPT_DIR}/metrics.csv"
TB_DIR="${CKPT_DIR}/tensorboard"

export CUDA_DEVICE_ORDER=PCI_BUS_ID            # CUDA indices == nvidia-smi indices
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false     # allocate on demand (gentle on a shared box)
export WANDB_MODE="${WANDB_MODE:-disabled}"    # never block on a W&B login; TensorBoard is the log

# --- Resolve GPU(s): explicit GPUS > auto-pick a display-free GPU -------------
# JAX meshes over EVERY visible GPU and FSDP_DEVICES does not cap that, so pin explicitly. Training
# on the GPU driving your monitors can crash the desktop during the pi0_base load + JIT.
# A Slurm GPU allocation is authoritative. Overriding it with a physical node index can expose an
# unallocated device or hide the allocated one, depending on the cluster's cgroup mapping.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "ERROR: Slurm job $SLURM_JOB_ID did not set CUDA_VISIBLE_DEVICES. Launch this script inside a GPU job step."
    exit 1
  fi
  echo "Using Slurm-assigned GPU(s): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES."
elif [ -n "${GPUS:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
  echo "Using GPU(s) $CUDA_VISIBLE_DEVICES (from config GPUS)."
else
  GPU="$("$UV" run examples/force/select_gpu.py --quiet 2>/dev/null | tail -1)"
  if [ -z "$GPU" ] || [ "$GPU" = "NONE" ]; then
    echo "ERROR: could not auto-pick a GPU. Set GPUS= in $CONFIG_FILE."
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="$GPU"
  echo "Auto-selected display-free GPU $CUDA_VISIBLE_DEVICES."
fi

# --- Preflight: JAX must see as many GPUs as we requested --------------------
WANT="$(printf '%s' "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')"
if [ -n "${EXPECTED_GPUS:-}" ] && [ "$WANT" != "$EXPECTED_GPUS" ]; then
  echo "ERROR: this run requested $EXPECTED_GPUS GPU(s), but Slurm exposed $WANT in CUDA_VISIBLE_DEVICES."
  exit 1
fi
echo "==== [0/2] preflight: JAX device check (expect $WANT) ===="
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
# FVLMoE model hyperparameters (the vanilla / no-force configs have no FVLMoE).
case "$MODE" in
  fvlmoe|draftvla|draftvla_20260905|draftvla_20260905_20260911|draftvla_forcevla|task12|task12_forcevla)
    ARGS+=(
      --model.fvlmoe-num-experts="$NUM_EXPERTS"
      --model.fvlmoe-num-heads="$NUM_HEADS"
      --model.fvlmoe-mlp-ratio="$MLP_RATIO"
    )
    ;;
esac
# Physical Interaction Token hyperparameters (only pi0_draftvla has these fields).
if [ "$MODE" = "draftvla" ] || [ "$MODE" = "draftvla_20260905" ] || [ "$MODE" = "draftvla_20260905_20260911" ] || [ "$MODE" = "task12" ]; then
  ARGS+=(
    --model.phy-dim="$PHY_DIM"
    --model.phy-num-prototypes="$NUM_PROTOTYPES"
    --model.lambda-dist="$LAMBDA_DIST"
    --model.lambda-proto="$LAMBDA_PROTO"
    --model.phy-proto-ramp-start="$PROTO_RAMP_START"
    --model.phy-proto-ramp-steps="$PROTO_RAMP_STEPS"
  )
fi
# Learnable G_phy gain (contract §0c) -- pi0_draftvla_task12 only.
if [ "$MODE" = "task12" ]; then
  ARGS+=(--model.phy-action-gain-init="${PHY_ACTION_GAIN_INIT:-13}")
fi

if [ "${SKIP_NORM_STATS:-0}" = "1" ]; then
  echo "==== [1/2] compute_norm_stats: SKIPPED (SKIP_NORM_STATS=1) ===="
else
  echo "==== [1/2] compute_norm_stats ($CONFIG) ===="
  "$UV" run --frozen scripts/compute_norm_stats.py --config-name "$CONFIG"
fi

echo "==== [2/2] train: MODE=$MODE config=$CONFIG batch=$BATCH_SIZE steps=$NUM_TRAIN_STEPS lr=$PEAK_LR ===="
echo "     overrides: ${ARGS[*]}"
echo "     TensorBoard: tensorboard --logdir $TB_DIR"
"$UV" run --frozen scripts/train.py "${ARGS[@]}" 2>&1 | tee "$LOG"

echo
echo "Done. config=$CONFIG exp=$EXP"
echo "  console log: $LOG"
echo "  metrics:     $METRICS"
echo "  tensorboard: tensorboard --logdir $TB_DIR"
echo "  checkpoints: $CKPT_DIR"
