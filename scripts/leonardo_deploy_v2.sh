#!/usr/bin/env bash
# Run ON THE LEONARDO LOGIN NODE after the v2 commit is on GitHub.
# Copies exactly the files changed by the v2 training contract from a pinned commit into the
# (non-git) working tree, verifies every one by sha256, then submits the smoke job.
#   bash leonardo_deploy_v2.sh <commit-sha>
set -euo pipefail
COMMIT=${1:?usage: leonardo_deploy_v2.sh <commit-sha>}
DST=/leonardo_work/IscrC_VLA/DamageVLA/openpi_draftvla
TMP=$(mktemp -d /tmp/damagevla_v2.XXXXXX)
git clone --quiet https://github.com/Hongyu-Lan/DamageVLA.git "$TMP/repo"
git -C "$TMP/repo" checkout --quiet "$COMMIT"

FILES=(
  examples/force/convert_draftvla_data_to_lerobot.py
  examples/force/convert_draftvla_data_to_lerobot_test.py
  examples/force/README.md
  examples/force/main.py
  src/openpi/models/pi0.py
  src/openpi/models/pi0_test.py
  src/openpi/models/action_loss_weight_test.py
  src/openpi/policies/policy.py
  src/openpi/policies/draftvla_policy.py
  src/openpi/policies/draftvla_policy_test.py
  src/openpi/policies/force_policy_test.py
  src/openpi/training/config.py
  src/openpi/training/data_loader.py
  run_train.sh
  submit_leonardo.sh
  train_config_task12_v2.conf
  train_config_task12_v2_forcevla.conf
  train_config_task12_v2_noforce.conf
  train_config_leonardo_smoke_task12_v2.conf
  outlines/todo_training_contract.md
  scripts/leonardo_deploy_v2.sh
)
# Keep a copy of what is being replaced.
BK="$DST/.v2_backup_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$BK"
for f in "${FILES[@]}"; do
  if [ -f "$DST/$f" ]; then mkdir -p "$BK/$(dirname "$f")"; cp -p "$DST/$f" "$BK/$f"; fi
  mkdir -p "$DST/$(dirname "$f")"
  cp -p "$TMP/repo/$f" "$DST/$f"
done
echo "backup of replaced files: $BK"

echo "verifying sha256 against commit $COMMIT"
fail=0
for f in "${FILES[@]}"; do
  a=$(sha256sum "$TMP/repo/$f" | cut -d' ' -f1); b=$(sha256sum "$DST/$f" | cut -d' ' -f1)
  if [ "$a" = "$b" ]; then echo "  OK  $f"; else echo "  MISMATCH  $f"; fail=1; fi
done
[ $fail -eq 0 ] || { echo "ABORT: mismatch"; exit 1; }
echo "$COMMIT" > "$DST/.v2_source_commit"
rm -rf "$TMP"

cd "$DST"
echo "submitting the v2 smoke (also converts both v2 splits on first run)"
bash submit_leonardo.sh smoke_task12_v2
squeue -u "$USER"
