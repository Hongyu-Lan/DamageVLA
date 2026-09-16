"""Plot every DraftVLA training curve from a saved metrics file (verifiable).

`scripts/train.py` writes `<checkpoint_dir>/metrics.csv` with one row per `--log-interval` steps and
a column per logged metric. The column set is discovered at runtime, so a DraftVLA run carries the
whole loss decomposition:

    step,frac_valid,grad_norm,lambda_proto,loss,loss_dist,loss_flow,loss_proto,loss_proto_excess,
    mu_pred_mean,param_norm,proto_acc,sigma_pred_mean,z_phy_norm

This renders a panel per metric, so the image is derived from real logged numbers rather than
hand-drawn. Same data as TensorBoard (`--logdir <checkpoint_dir>/tensorboard`); this is the static,
checked-in-able artifact.

Usage:
    uv run examples/force/plot_draftvla_losses.py \
        --metrics checkpoints/pi0_draftvla/draftvla_smoke/metrics.csv \
        --out examples/force/draftvla_smoke_losses.jpg
"""

import argparse
import csv
import math
import pathlib

import matplotlib.pyplot as plt

plt.switch_backend("Agg")  # headless: no display needed

# Panel order: the headline total first, then the loss decomposition, then the diagnostics.
_PREFERRED_ORDER = [
    "loss",
    "loss_flow",
    "loss_dist",
    "loss_proto",
    "loss_proto_excess",
    "proto_acc",
    "frac_valid",
    "lambda_proto",
    "mu_pred_mean",
    "sigma_pred_mean",
    "z_phy_norm",
    "grad_norm",
    "param_norm",
]

# What each panel is for, so the image is readable without the plan open.
_NOTES = {
    "loss": "total = flow + λ_dist·dist + λ_proto·proto",
    "loss_flow": "pi0 flow-matching objective",
    "loss_dist": "KL(p_gt‖p_pred), masked (plan §7.2)",
    "loss_proto": "soft-label CE over prototypes",
    "loss_proto_excess": "L_proto - H(y): ~0 means at the entropy bound",
    "proto_acc": "top-1 vs argmax(y_target), masked",
    "frac_valid": "fraction of the batch with physical labels (~0.47 expected)",
    "lambda_proto": "λ_proto after the ramp schedule",
    "mu_pred_mean": "predicted safe-dist mean (raw tactile units)",
    "sigma_pred_mean": "predicted safe-dist sigma (raw tactile units)",
    "z_phy_norm": "‖z_phy‖ — L2-normalized, so ~1.0",
    "grad_norm": "global grad norm",
    "param_norm": "kernel param norm — must stay ~CONSTANT (frozen VLM)",
}


def parse_csv(path: pathlib.Path) -> tuple[list[int], dict[str, list[float]]]:
    steps: list[int] = []
    series: dict[str, list[float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        keys = [k for k in (reader.fieldnames or []) if k != "step"]
        series = {k: [] for k in keys}
        for row in reader:
            steps.append(int(row["step"]))
            for k in keys:
                value = row.get(k)
                series[k].append(float(value) if value not in (None, "") else math.nan)
    return steps, series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics", type=pathlib.Path, required=True, help="path to metrics.csv")
    parser.add_argument("--out", type=pathlib.Path, required=True, help="output image path")
    parser.add_argument("--title", default="DraftVLA training", help="figure title")
    args = parser.parse_args()

    steps, series = parse_csv(args.metrics)
    if not steps:
        raise SystemExit(f"No rows found in {args.metrics}")

    # Preferred metrics first, then anything else the run happened to log.
    keys = [k for k in _PREFERRED_ORDER if k in series]
    keys += [k for k in series if k not in keys]

    ncols = 3
    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.1 * nrows), squeeze=False)

    for i, key in enumerate(keys):
        ax = axes[i // ncols][i % ncols]
        values = series[key]
        ax.plot(steps, values, lw=1.4, color="#0b6e99")
        ax.set_title(key, fontsize=11, fontweight="bold")
        ax.set_xlabel("step", fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)
        if note := _NOTES.get(key):
            ax.text(
                0.5, -0.30, note, transform=ax.transAxes, fontsize=7.5, color="#555", ha="center", va="top", wrap=True
            )
        finite = [v for v in values if math.isfinite(v)]
        if finite:
            first, last = finite[0], finite[-1]
            ax.text(
                0.97,
                0.94,
                f"{first:.4g} → {last:.4g}",
                transform=ax.transAxes,
                fontsize=8,
                ha="right",
                va="top",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
            )
            # A flat param_norm is the check that the VLM freeze actually held.
            if key == "param_norm":
                span = max(finite) - min(finite)
                rel = span / (abs(finite[0]) + 1e-9)
                ax.text(
                    0.03,
                    0.06,
                    f"relative drift: {rel:.2e}",
                    transform=ax.transAxes,
                    fontsize=8,
                    ha="left",
                    va="bottom",
                    color="#a33",
                )

    for j in range(len(keys), nrows * ncols):  # hide unused panels
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"{args.title}  ({len(steps)} logged points, steps {steps[0]}-{steps[-1]})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"Wrote {args.out}")

    for key in keys:
        finite = [v for v in series[key] if math.isfinite(v)]
        if finite:
            print(
                f"  {key:20s} first={finite[0]:12.6g}  last={finite[-1]:12.6g}  min={min(finite):12.6g}  max={max(finite):12.6g}"
            )


if __name__ == "__main__":
    main()
