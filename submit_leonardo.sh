#!/usr/bin/env bash
# Submit a one-GPU gate, an optional four-GPU communication gate, or the one-GPU 10k-step run.
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-smoke}"
case "$MODE" in
  smoke)
    CONFIG_FILE=train_config_leonardo_smoke.conf
    WALLTIME=02:00:00
    NUM_GPUS=1
    NUM_CPUS=16
    MEMORY=120G
    ;;
  smoke4)
    CONFIG_FILE=train_config_leonardo_smoke4.conf
    WALLTIME=02:00:00
    NUM_GPUS=4
    NUM_CPUS=32
    MEMORY=480G
    ;;
  full)
    CONFIG_FILE=train_config.conf
    WALLTIME=24:00:00
    NUM_GPUS=1
    NUM_CPUS=16
    MEMORY=120G
    ;;
  smoke_20260905)
    CONFIG_FILE=train_config_leonardo_smoke_20260905.conf
    WALLTIME=02:00:00
    NUM_GPUS=1
    NUM_CPUS=16
    MEMORY=120G
    ;;
  full_20260905)
    CONFIG_FILE=train_config_20260905.conf
    WALLTIME=24:00:00
    NUM_GPUS=1
    NUM_CPUS=16
    MEMORY=120G
    ;;
  smoke_combined)
    CONFIG_FILE=train_config_leonardo_smoke_20260905_20260911.conf
    WALLTIME=02:00:00
    NUM_GPUS=1
    NUM_CPUS=16
    MEMORY=120G
    ;;
  full_combined)
    CONFIG_FILE=train_config_20260905_20260911.conf
    WALLTIME=24:00:00
    NUM_GPUS=1
    NUM_CPUS=16
    MEMORY=120G
    ;;
  smoke_task12)
    CONFIG_FILE=train_config_leonardo_smoke_task12.conf
    WALLTIME=02:00:00
    NUM_GPUS=1
    NUM_CPUS=16
    MEMORY=120G
    ;;
  full_task12)
    CONFIG_FILE=train_config_task12.conf
    WALLTIME=24:00:00
    NUM_GPUS=1
    NUM_CPUS=16
    MEMORY=120G
    ;;
  full_task12_forcevla)
    CONFIG_FILE=train_config_task12_forcevla.conf
    WALLTIME=24:00:00
    NUM_GPUS=1
    NUM_CPUS=16
    MEMORY=120G
    ;;
  full_task12_noforce)
    CONFIG_FILE=train_config_task12_noforce.conf
    WALLTIME=24:00:00
    NUM_GPUS=1
    NUM_CPUS=16
    MEMORY=120G
    ;;
  *)
    echo "Usage: bash submit_leonardo.sh [smoke|smoke4|full|smoke_20260905|full_20260905|smoke_combined|full_combined|smoke_task12|full_task12|full_task12_forcevla|full_task12_noforce]" >&2
    exit 2
    ;;
esac

if [ ! -x .venv/bin/python ] || [ ! -x .tools/uv/bin/uv ]; then
  echo "ERROR: environment not ready. Run: bash setup_leonardo.sh" >&2
  exit 1
fi

# The combined raw directory initially contains prototype labels computed independently per batch.
# Check the unified metadata before allocating a GPU, so stale/misaligned prototype IDs cannot be
# converted and trained accidentally. Run the combined postprocessor with --write to satisfy this gate.
if [ "$MODE" = "smoke_combined" ] || [ "$MODE" = "full_combined" ]; then
  (
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
    PROTOTYPE_METADATA="$RAW_DATA_DIR/prototype_metadata/group_metadata.json"
    .venv/bin/python -       "$PROTOTYPE_METADATA"       "$EXPECTED_PROTOTYPE_GROUPS"       "$EXPECTED_PROTOTYPE_EPISODES"       "$EXCLUDE_EPISODES" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected_groups = int(sys.argv[2])
expected_episodes = int(sys.argv[3])
expected_excluded = set(sys.argv[4].split())
if not path.is_file():
    raise SystemExit(f"ERROR: combined prototype metadata is missing: {path}")
metadata = json.loads(path.read_text())
actual_groups = int(metadata.get("n_group", -1))
actual_episodes = len(metadata.get("episodes_used", []))
actual_excluded = set(metadata.get("episodes_excluded", []))
if (
    actual_groups != expected_groups
    or actual_episodes != expected_episodes
    or actual_excluded != expected_excluded
):
    raise SystemExit(
        "ERROR: combined prototype labels are not ready: "
        f"groups={actual_groups}/{expected_groups}, "
        f"episodes={actual_episodes}/{expected_episodes}, "
        f"excluded={sorted(actual_excluded)}/{sorted(expected_excluded)}. "
        "Run postprocess_20260911.py on the combined raw directory with --write before submitting."
    )
print(
    f"Combined prototype gate: PASS ({actual_groups} groups, "
    f"{actual_episodes} contributing episodes, excluded={sorted(actual_excluded)})"
)
PY
  )
fi

echo "Submitting $MODE run with $CONFIG_FILE: ${NUM_GPUS}x A100, ${NUM_CPUS} CPUs, $MEMORY RAM, walltime $WALLTIME."
sbatch \
  --time="$WALLTIME" \
  --gres="gpu:$NUM_GPUS" \
  --cpus-per-task="$NUM_CPUS" \
  --mem="$MEMORY" \
  --export="ALL,CONFIG_FILE=$CONFIG_FILE,EXPECTED_GPUS=$NUM_GPUS" \
  scripts/leonardo_train.sbatch
