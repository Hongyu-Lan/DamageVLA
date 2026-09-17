#!/usr/bin/env bash
#SBATCH --job-name=draftvla
#SBATCH --account=iscrc_vla
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
# Slurm executes a copied script from /var/spool/slurmd, so $0 no longer
# points into this repository.  The submit helper always submits from the
# repository root and Slurm preserves that directory here.
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"

# shellcheck disable=SC1091
source ./env_leonardo.sh

CONFIG_FILE="${CONFIG_FILE:-train_config_leonardo_smoke.conf}"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "ERROR: config file '$CONFIG_FILE' does not exist in $PWD." >&2
  exit 1
fi
# Dataset-specific settings live beside the optimization settings. Old config files omit these
# values and therefore retain the original 20260821 behavior.
# shellcheck disable=SC1090
source "$CONFIG_FILE"

# 2026-09-16 contract defaults (Plan B). All overridable from the conf file.
RAW_DATA_DIR="${RAW_DATA_DIR:-$DRAFTVLA_ROOT/../DamageVLA_training_post_process_20260916}"
LABELS_DIR="${LABELS_DIR:-prototype_metadata_task12_trainonly}"
TRAIN_EPISODES_FILE="${TRAIN_EPISODES_FILE:-examples/force/train_episodes_20260917.txt}"
VAL_EPISODES_FILE="${VAL_EPISODES_FILE:-examples/force/val_episodes_20260917.txt}"
DATASET_REPO_ID="${DATASET_REPO_ID:-draftvla/task12_tactile_train}"
VAL_DATASET_REPO_ID="${VAL_DATASET_REPO_ID:-draftvla/task12_tactile_val}"
EXPECTED_RAW_EPISODES="${EXPECTED_RAW_EPISODES:-83}"
EXPECTED_RAW_FRAMES="${EXPECTED_RAW_FRAMES:-47591}"
EXPECTED_LEROBOT_EPISODES="${EXPECTED_LEROBOT_EPISODES:-62}"
EXPECTED_LEROBOT_FRAMES="${EXPECTED_LEROBOT_FRAMES:-35549}"
EXPECTED_VAL_EPISODES="${EXPECTED_VAL_EPISODES:-15}"
EXPECTED_VAL_FRAMES="${EXPECTED_VAL_FRAMES:-8596}"

if [ ! -x .venv/bin/python ] || ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: dependency environment is missing. Run 'bash setup_leonardo.sh' on a login node first."
  exit 1
fi

echo "job_id=${SLURM_JOB_ID:-unset} host=$(hostname) started=$(date --iso-8601=seconds)"
echo "config=$CONFIG_FILE"
echo "raw_data_dir=$RAW_DATA_DIR"
echo "dataset_repo_id=$DATASET_REPO_ID"
echo "requested_gpus=${SLURM_GPUS:-unset} allocated_gpus=${SLURM_JOB_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "HF_LEROBOT_HOME=$HF_LEROBOT_HOME"
nvidia-smi
uv run --frozen python -c 'import jax; print("JAX devices:", jax.devices())'

echo "==== [data 1/2] validating raw JSONL, images, and label sidecars ===="
uv run --frozen python examples/force/validate_draftvla_data.py \
  --data-dir "$RAW_DATA_DIR" \
  --labels-dir "$LABELS_DIR" \
  --expected-episodes "$EXPECTED_RAW_EPISODES" \
  --expected-frames "$EXPECTED_RAW_FRAMES"

convert_repo() {  # repo_id episodes_file expected_episodes expected_frames
  local validate=(
    uv run --frozen python examples/force/validate_lerobot_dataset.py
    --repo-id "$1" --expected-episodes "$3" --expected-frames "$4"
  )
  if [ "${FORCE_RECONVERT:-0}" = "1" ] || ! "${validate[@]}"; then
    echo "==== [data 2/2] converting $3 episodes to LeRobot ($1) ===="
    uv run --frozen python examples/force/convert_draftvla_data_to_lerobot.py \
      --data-dir "$RAW_DATA_DIR" \
      --labels-dir "$LABELS_DIR" \
      --episodes-file "$2" \
      --repo-id "$1"
    "${validate[@]}"
  else
    echo "==== [data 2/2] reusing the validated converted dataset ($1) ===="
  fi
}
convert_repo "$DATASET_REPO_ID" "$TRAIN_EPISODES_FILE" "$EXPECTED_LEROBOT_EPISODES" "$EXPECTED_LEROBOT_FRAMES"
convert_repo "$VAL_DATASET_REPO_ID" "$VAL_EPISODES_FILE" "$EXPECTED_VAL_EPISODES" "$EXPECTED_VAL_FRAMES"

if [ "${PREPARE_ONLY:-0}" = "1" ]; then
  echo "PREPARE_ONLY=1: conversion complete; training skipped."
  exit 0
fi

echo "==== training ===="
CONFIG_FILE="$CONFIG_FILE" bash run_train.sh
echo "finished=$(date --iso-8601=seconds)"
