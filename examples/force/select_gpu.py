"""Pick a *display-free* GPU for training, to avoid crashing the desktop on multi-GPU machines.

Training on the GPU that drives your monitors can hang its driver during the heavy weight-load +
JIT-compile phase and take down the whole X session (you get bounced to the login screen). This
script finds a GPU with **no graphics (type "G") process** attached -- i.e. no display -- and prints
its index. Use it together with ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` so the index matches ``nvidia-smi``.

  uv run examples/force/select_gpu.py            # human-readable report + recommendation
  uv run examples/force/select_gpu.py --quiet    # print only the chosen index (or "NONE")

Detection uses ``nvidia-smi -q -x`` (XML); ``display_active`` is NOT used because it is unreliable
(it can read "Disabled" on a GPU that is in fact driving the desktop).
"""

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET


def query_gpus() -> list[dict]:
    xml = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True, check=True).stdout
    root = ET.fromstring(xml)
    gpus = []
    for index, gpu in enumerate(root.findall("gpu")):
        procs = gpu.find("processes")
        # G = graphics (display) process, C = compute. A GPU with any graphics process drives a display.
        graphics = (
            [
                (p.findtext("process_name") or "<graphics>").strip()
                for p in procs.findall("process_info")
                if (p.findtext("type") or "").strip() == "G"
            ]
            if procs is not None
            else []
        )
        gpus.append(
            {
                "index": index,
                "bus": (gpu.get("id") or "").strip(),
                "used": (gpu.findtext("fb_memory_usage/used") or "?").strip(),
                "graphics": graphics,
            }
        )
    return gpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="Print only the chosen index (or NONE).")
    args = parser.parse_args()

    try:
        gpus = query_gpus()
    except Exception as e:  # any failure here should be reported, not crash the run script
        print("NONE" if args.quiet else f"could not query nvidia-smi: {e}")
        sys.exit(2)

    display_free = [g for g in gpus if not g["graphics"]]
    chosen = display_free[0]["index"] if display_free else None

    if args.quiet:
        print(chosen if chosen is not None else "NONE")
        return

    print(f"Found {len(gpus)} GPU(s) (index == nvidia-smi index when CUDA_DEVICE_ORDER=PCI_BUS_ID):")
    for g in gpus:
        label = "DISPLAY" if g["graphics"] else "free"
        procs = (" -> " + ", ".join(g["graphics"])) if g["graphics"] else ""
        print(f"  GPU {g['index']} ({g['bus']}): {label}, fb_used={g['used']}{procs}")

    if chosen is not None:
        print(f"\nRecommended (display-free) GPU: {chosen}")
        print(f"  bash examples/force/run_smoke_repro.sh {chosen}")
    else:
        print("\nWARNING: every GPU has a display attached -- training on any of them can crash X.")
        print("  (a) plug monitors into ONE GPU only and leave the other free; or")
        print("  (b) run from a headless/SSH session (no desktop on the GPU); or")
        print("  (c) accept the risk and pin one explicitly: bash examples/force/run_smoke_repro.sh <index>")


if __name__ == "__main__":
    main()
