#!/usr/bin/env bash
#SBATCH --job-name=draftvla-eval
#SBATCH --account=iscrc_vla
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source ./env_leonardo.sh
mkdir -p eval_reports
for spec in \
  "pi0_draftvla_task12:task12_full" \
  "pi0_draftvla_task12_forcevla:task12_forcevla_full" \
  "pi0_draftvla_task12_noforce:task12_noforce_full"; do
  cfg="${spec%%:*}"; exp="${spec##*:}"
  for step in 5000 9999; do
    echo "==================== $cfg @ $step ===================="
    uv run --frozen python examples/force/eval_draftvla_offline.py \
      --config-name "$cfg" \
      --checkpoint-dir "checkpoints/$cfg/$exp/$step" \
      --val-repo-id draftvla/task12_tactile_val \
      --output-json "eval_reports/${cfg}_${step}.json"
  done
done
echo "ALL EVALS DONE"
