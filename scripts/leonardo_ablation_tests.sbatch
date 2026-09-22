#!/usr/bin/env bash
# §4 go/no-go gate for the B1/B2/B3 ablation switches (outlines/todo_ablation_switches.md).
# Runs on a compute node: the login node cannot finish this suite inside a sane wall time.
#SBATCH --job-name=draftvla-abl-tests
#SBATCH --account=iscrc_vla
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source ./env_leonardo.sh
# NO PIPE: a pipeline would hand back tail's exit status and hide a failure or a timeout.
uv run --frozen pytest src/openpi/models/physical_test.py src/openpi/models/pi0_test.py -v --durations=10
echo "PYTEST_EXIT=$?"
