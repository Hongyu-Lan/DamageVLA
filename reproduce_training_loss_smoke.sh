#!/usr/bin/env bash
# One-command reproduction of the Force-aware pi0 (FVLMoE / M2) training-loss smoke test.
#
#   bash reproduce_training_loss_smoke.sh         # auto-pick the display-free GPU and run
#   bash reproduce_training_loss_smoke.sh 1       # or pin a GPU index (nvidia-smi order)
#
# This box has 2x RTX 3090 and a running desktop, so BOTH GPUs have an Xorg bound and the strict
# auto-picker in examples/force/run_smoke_repro.sh bails. Here we instead rank GPUs by how much
# GRAPHICS memory each is using and pick the lightest one -- i.e. the GPU NOT running gnome-shell /
# the desktop. Training on the desktop GPU can hang its driver during the heavy pi0_base load + JIT
# and bounce X to the login screen BEFORE step 0 (a driver crash, not OOM), so this matters.
#
# It then hands off to examples/force/run_training_loss_smoke.sh, which runs:
#   compute_norm_stats -> train (FVLMoE, batch 4, 200 steps, log every step) -> plot loss.
# Expected result: loss ~3.6 -> ~0.13 and param_norm ~constant (VLM stays frozen).
set -euo pipefail

# Always run from the repo root (the dir this script lives in), so relative paths resolve.
cd "$(dirname "$0")"

export CUDA_DEVICE_ORDER=PCI_BUS_ID   # CUDA indices == nvidia-smi indices

# --- Choose the GPU: explicit arg > preset CUDA_VISIBLE_DEVICES > auto-pick lightest-graphics ------
if [ "${1:-}" != "" ]; then
  GPU="$1"
  echo "Using GPU $GPU (command-line argument)."
elif [ "${CUDA_VISIBLE_DEVICES:-}" != "" ]; then
  GPU="$CUDA_VISIBLE_DEVICES"
  echo "Using GPU $GPU (preset CUDA_VISIBLE_DEVICES)."
else
  # Pick the GPU with the LEAST graphics-process ("G" type) memory -- the one not driving the desktop.
  GPU="$(python3 - <<'PY'
import subprocess, sys, xml.etree.ElementTree as ET
try:
    xml = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True, check=True).stdout
    root = ET.fromstring(xml)
except Exception:
    print("NONE"); sys.exit(0)
best, best_mem = None, None
for i, gpu in enumerate(root.findall("gpu")):
    procs = gpu.find("processes")
    g_mem = 0
    if procs is not None:
        for p in procs.findall("process_info"):
            if (p.findtext("type") or "").strip() == "G":
                mem = (p.findtext("used_memory") or "0 MiB").split()[0]
                try: g_mem += int(mem)
                except ValueError: pass
    if best_mem is None or g_mem < best_mem:
        best, best_mem = i, g_mem
print(best if best is not None else "NONE")
PY
)"
  if [ "$GPU" = "NONE" ] || [ -z "$GPU" ]; then
    echo "ERROR: could not query GPUs via nvidia-smi. Pin one explicitly, e.g.: bash $0 1"
    exit 1
  fi
  echo "Auto-selected display-free GPU $GPU (lightest graphics load; override with: bash $0 <index>)."
fi

exec env CUDA_VISIBLE_DEVICES="$GPU" bash examples/force/run_training_loss_smoke.sh "$GPU"
