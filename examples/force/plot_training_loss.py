"""Plot the Force-aware pi0 training-loss curve **from a saved metrics file** (verifiable).

`scripts/train.py` writes `<checkpoint_dir>/metrics.csv` with a row per `--log-interval` steps:

    step,loss,grad_norm,param_norm
    0,3.6021,6.7890,1234.5678
    1,3.5814,6.1212,1234.5677
    ...

This script renders the loss-vs-step plot from that CSV, so the image is derived from real logged
numbers rather than hand-drawn. It also reports the param_norm range, which should stay ~constant
when the VLM is frozen (a quick check that freezing actually worked).

Usage:
    uv run examples/force/plot_training_loss.py \
        --metrics checkpoints/pi0_force_fvlmoe/force_fvlmoe_smoke_repro/metrics.csv \
        --out examples/force/training_loss_smoke_repro.jpg

A `--log <train stdout>` fallback is kept for older runs that predate metrics.csv (best-effort regex).
"""

import argparse
import csv
import pathlib
import re

import matplotlib.pyplot as plt

plt.switch_backend("Agg")  # headless: no display needed

# Fallback only: matches "Step 12: loss=1.2345, grad_norm=6.7890, param_norm=1234.5678".
_STEP_RE = re.compile(r"Step\s+(\d+):\s+loss=([-+0-9.eE]+)(?:.*param_norm=([-+0-9.eE]+))?")


def parse_csv(path: pathlib.Path) -> tuple[list[int], list[float], list[float]]:
    steps, losses, param_norms = [], [], []
    with path.open() as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
            if row.get("param_norm") not in (None, ""):
                param_norms.append(float(row["param_norm"]))
    return steps, losses, param_norms


def parse_log(path: pathlib.Path) -> tuple[list[int], list[float], list[float]]:
    steps, losses, param_norms = [], [], []
    for line in path.read_text().splitlines():
        if m := _STEP_RE.search(line):
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            if m.group(3) is not None:
                param_norms.append(float(m.group(3)))
    return steps, losses, param_norms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=pathlib.Path, help="metrics.csv written by train.py (preferred).")
    parser.add_argument("--log", type=pathlib.Path, help="Fallback: train.py stdout log (best-effort regex).")
    parser.add_argument("--out", type=pathlib.Path, required=True, help="Output image path (.jpg/.png).")
    parser.add_argument(
        "--title",
        default="Force-aware π₀ (FVLMoE / M2) — training smoke test\n"
        "pi0_force_fvlmoe, single 'grasp the banana' episode, batch 4",
        help="Plot title.",
    )
    args = parser.parse_args()

    if args.metrics and args.metrics.exists():
        steps, losses, param_norms = parse_csv(args.metrics)
        source = args.metrics
    elif args.log and args.log.exists():
        steps, losses, param_norms = parse_log(args.log)
        source = args.log
    else:
        raise SystemExit("Provide --metrics <metrics.csv> (preferred) or --log <train stdout>.")

    if not steps:
        raise SystemExit(f"No (step, loss) points found in {source}.")

    # Report key numbers so the run is verifiable even without opening the image.
    print(f"parsed {len(steps)} points from {source}")
    print(f"loss: first={losses[0]:.4f}  last={losses[-1]:.4f}  min={min(losses):.4f}  max={max(losses):.4f}")
    if param_norms:
        spread = max(param_norms) - min(param_norms)
        rel = spread / max(abs(param_norms[0]), 1e-9)
        print(
            f"param_norm: first={param_norms[0]:.4f}  last={param_norms[-1]:.4f}  "
            f"spread={spread:.4g} ({rel:.2%} of initial) -> ~constant means the VLM stayed frozen"
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, losses, marker=".", linewidth=1.0, color="C0")
    ax.set_xlabel("training step")
    ax.set_ylabel("flow-matching loss (mean over action chunk)")
    ax.set_title(args.title)
    ax.grid(visible=True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
