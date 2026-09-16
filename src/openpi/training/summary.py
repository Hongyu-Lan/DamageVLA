"""End-of-training diagnostic summary for DraftVLA runs.

Reads the run's `metrics.csv` and prints a compact block that answers, from the log alone, *why* a
physical head did or did not learn. Written to `<checkpoint_dir>/training_summary.txt` as well as
stdout, so a run can be diagnosed after the fact without the GPU, TensorBoard, or the checkpoint.

The checks are the ones that distinguish causes a loss curve cannot:

  - LR schedule       -- did warmup_steps exceed num_train_steps? (the LR never reaches peak)
  - label supply      -- how many labeled frames per step actually reached L_dist / L_proto?
  - gradient delivery -- did each physical module receive gradient at all?
  - z_phy collapse    -- does the token carry per-sample information, or one direction for all?
  - vs trivial        -- does the safe-dist head beat a constant predictor?
  - proto collapse    -- is the classifier pinned at the uniform distribution, log(K)?
  - guidance weight   -- is G_phy big enough next to G_fvl to move the action head?
  - freeze integrity  -- did the frozen VLM stay frozen?

Deliberately conservative: every verdict is derived from logged numbers and says which number it
used, so a wrong verdict is checkable rather than authoritative.
"""

import csv
import math
import pathlib
import statistics

# A metric is "flat" if its relative change across the run is under this.
_FLAT_RELATIVE_CHANGE = 0.05
# z_phy mean pairwise cosine above this is treated as collapsed (it is L2-normalized).
_COLLAPSE_COSINE = 0.95
# |L_proto - log(K)| under this means the classifier is still uniform.
_UNIFORM_TOLERANCE = 0.02
# ||G_phy|| / ||G_fvl|| below this means the physical guidance is negligible.
_NEGLIGIBLE_GUIDANCE = 0.05


def _read(path: pathlib.Path) -> tuple[list[int], dict[str, list[float]]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        keys = [k for k in (reader.fieldnames or []) if k != "step"]
        steps: list[int] = []
        series: dict[str, list[float]] = {k: [] for k in keys}
        for row in reader:
            steps.append(int(row["step"]))
            for k in keys:
                v = row.get(k)
                series[k].append(float(v) if v not in (None, "") else math.nan)
    return steps, series


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def _window_mean(values: list[float], *, first: bool, frac: float = 0.1) -> float:
    """Mean of the first/last `frac` of the run -- robust to the per-step noise."""
    vals = _finite(values)
    if not vals:
        return math.nan
    n = max(1, int(len(vals) * frac))
    return statistics.mean(vals[:n] if first else vals[-n:])


# Losses that are masked by `supervision_valid`: on an all-masked step they are logged as 0 by
# construction, so averaging over every step drags them toward 0 and hides their real value. Any
# statistic on these MUST be taken over labeled steps only.
_MASKED_METRICS = frozenset(
    {"loss_dist", "loss_proto", "loss_proto_excess", "proto_acc", "proto_entropy", "loss_dist_baseline"}
    | {f"kl_{d}" for d in ("fx", "fy", "fz", "tx", "ty", "tz")}
)

# Diagnostics this summary needs to reach a verdict. Missing => the run predates them => say so
# rather than reporting "healthy", which is what absence of evidence would otherwise look like.
_REQUIRED_DIAGNOSTICS = (
    "lr",
    "num_valid",
    "z_phy_cos",
    "loss_dist_baseline",
    "proto_entropy",
    "g_phy_rel",
)


def build_summary(metrics_path: pathlib.Path, *, config_line: str = "") -> str:
    steps, series = _read(metrics_path)
    if not steps:
        return f"[summary] no rows in {metrics_path}"

    out: list[str] = []
    add = out.append
    verdicts: list[str] = []

    # Index of steps that actually carried labels, for the masked metrics.
    valid_key = "num_valid" if "num_valid" in series else ("frac_valid" if "frac_valid" in series else None)
    labeled_idx = (
        [i for i, v in enumerate(series[valid_key]) if math.isfinite(v) and v > 0]
        if valid_key
        else list(range(len(steps)))
    )

    def _values(key: str) -> list[float]:
        """The series for `key`, restricted to labeled steps when the metric is masked."""
        if key in _MASKED_METRICS and valid_key:
            return [series[key][i] for i in labeled_idx]
        return series[key]

    def has(key: str) -> bool:
        return key in series and bool(_finite(_values(key)))

    def first_last(key: str) -> tuple[float, float]:
        vals = _values(key)
        return _window_mean(vals, first=True), _window_mean(vals, first=False)

    add("=" * 78)
    add("DraftVLA training summary")
    add("=" * 78)
    if config_line:
        add(f"config     : {config_line}")
    add(f"metrics    : {metrics_path}")
    add(f"steps      : {steps[0]}..{steps[-1]}  ({len(steps)} logged points)")

    # --- 1. LR schedule ------------------------------------------------------------------------
    if has("lr"):
        lr_max = max(_finite(series["lr"]))
        _, lr_last = first_last("lr")
        add("")
        add(f"LR         : max reached={lr_max:.3e}  final={lr_last:.3e}")
        if lr_max < 1e-5:
            verdicts.append(
                f"LR NEVER EXCEEDED {lr_max:.1e}. Freshly-initialized heads (phy_*) barely move at this "
                "rate while the pretrained flow head still fine-tunes -- this alone can pin L_proto at "
                "log(K) and leave L_dist drifting. Check warmup_steps < num_train_steps."
            )

    # --- 2. Label supply -----------------------------------------------------------------------
    if has("num_valid"):
        vals = _finite(series["num_valid"])
        mean_valid = statistics.mean(vals)
        zero_steps = sum(1 for v in vals if v == 0)
        add("")
        add(
            f"labels     : {mean_valid:.2f} labeled frames/step (min={min(vals):.0f} max={max(vals):.0f});"
            f" {zero_steps}/{len(vals)} steps had NONE"
        )
        add("             (masked metrics below are averaged over LABELED steps only -- an all-masked")
        add("              step logs 0 by construction and would otherwise drag them toward 0)")
        if mean_valid < 8:
            verdicts.append(
                f"LABEL STARVATION: only {mean_valid:.1f} labeled frames/step reach L_dist and L_proto "
                "(43.6% of kept frames are unlabeled). The KL is estimated from a handful of samples, so its "
                "noise floor swamps the signal. Raise batch_size."
            )
    elif has("frac_valid"):
        add("")
        add(f"labels     : frac_valid mean={statistics.mean(_finite(series['frac_valid'])):.3f}")

    # --- 3. Gradient delivery ------------------------------------------------------------------
    gnorm_keys = sorted(k for k in series if k.startswith("gnorm_"))
    if gnorm_keys:
        add("")
        add("grad norms per module (mean over run):")
        for key in gnorm_keys:
            vals = _finite(series[key])
            if not vals:
                continue
            mean_g = statistics.mean(vals)
            add(f"    {key[6:]:<18} {mean_g:.3e}")
            if mean_g < 1e-9:
                verdicts.append(
                    f"NO GRADIENT reaches {key[6:]} (mean {mean_g:.1e}). Its loss cannot move it: the "
                    "module is disconnected from the objective or frozen by the freeze filter."
                )

    # --- 4. The three failure suspects ---------------------------------------------------------
    add("")
    add("physical branch:")

    if has("z_phy_cos"):
        cos_first, cos_last = first_last("z_phy_cos")
        add(f"    z_phy_cos          {cos_first:+.4f} -> {cos_last:+.4f}   (mean pairwise cosine; ~1 = collapsed)")
        if cos_last > _COLLAPSE_COSINE:
            verdicts.append(
                f"z_phy COLLAPSED (pairwise cosine {cos_last:.3f}): every sample maps to the same "
                "direction, so z_phy carries no per-sample information and NO downstream head can "
                "discriminate, regardless of LR or training length. Suspect the FVLMoE hidden or the "
                "L2 normalization washing out the signal."
            )
        else:
            add(f"      -> z_phy is spread (cos {cos_last:.3f} < {_COLLAPSE_COSINE}); not a collapse.")

    if has("z_phy_norm"):
        _, zn = first_last("z_phy_norm")
        add(f"    z_phy_norm         {zn:.4f}                (L2-normalized => must be ~1.0)")

    if has("loss_dist") and has("loss_dist_baseline"):
        d_first, d_last = first_last("loss_dist")
        b_first, b_last = first_last("loss_dist_baseline")
        ratio = d_last / (b_last + 1e-9)
        add(f"    loss_dist          {d_first:.4f} -> {d_last:.4f}")
        add(f"    loss_dist_baseline {b_first:.4f} -> {b_last:.4f}   (constant predictor: mu=0, sigma=1 normalized)")
        add(f"      -> ratio vs trivial = {ratio:.3f}   (<1 beats a constant; ~1 learned nothing; >1 worse)")
        if ratio > 1.0:
            verdicts.append(
                f"L_dist is WORSE than a constant predictor (ratio {ratio:.2f}). The head is not fitting "
                "the labels. If gradients are healthy and z_phy is spread, suspect the LR/warmup or that "
                "z_phy's input distribution is being reshaped by the flow gradient faster than the head "
                "can track."
            )
        elif ratio > 1.0 - _FLAT_RELATIVE_CHANGE:
            verdicts.append(f"L_dist matches the trivial baseline (ratio {ratio:.2f}): the head learned nothing.")

    if has("loss_proto"):
        p_first, p_last = first_last("loss_proto")
        add(f"    loss_proto         {p_first:.4f} -> {p_last:.4f}")
        if has("lambda_proto"):
            lam = max(_finite(series["lambda_proto"]))
            add(f"    lambda_proto (max) {lam:.4f}")
            if lam == 0.0:
                verdicts.append("lambda_proto stayed 0: L_proto was never actually optimized (check the ramp).")
        # Compare against log(K) for the K implied by the run, if the accuracy hints at it.
        for k in (3, 4, 5, 6):
            if abs(p_last - math.log(k)) < _UNIFORM_TOLERANCE:
                verdicts.append(
                    f"L_proto is PINNED AT log({k})={math.log(k):.4f} (value {p_last:.4f}): the classifier "
                    "emits a uniform distribution, i.e. it has learned nothing at all. This is 'no learning', "
                    "NOT 'learned but wrong' -- the latter would show low entropy with low accuracy."
                )
                break
    if has("proto_entropy"):
        e_first, e_last = first_last("proto_entropy")
        add(f"    proto_entropy      {e_first:.4f} -> {e_last:.4f}   (H(p); ~log(K) = uniform = collapsed)")
    if has("proto_acc"):
        a_first, a_last = first_last("proto_acc")
        add(f"    proto_acc          {a_first:.4f} -> {a_last:.4f}")

    if has("g_phy_rel"):
        g_first, g_last = first_last("g_phy_rel")
        add(f"    g_phy_rel          {g_first:.4f} -> {g_last:.4f}   (||G_phy||/||G_fvl||)")
        if g_last < _NEGLIGIBLE_GUIDANCE:
            verdicts.append(
                f"G_phy is NEGLIGIBLE next to G_fvl (ratio {g_last:.3f}): the physical token is not moving "
                "the action head in any meaningful way, so the branch is decorative -- it would fail plan "
                "rule 3 ('must influence action generation') even if its own losses looked fine."
            )

    # --- 5. Per-dim attribution ----------------------------------------------------------------
    kl_keys = [k for k in ("kl_fx", "kl_fy", "kl_fz", "kl_tx", "kl_ty", "kl_tz") if has(k)]
    if kl_keys:
        add("")
        add("L_dist per wrench dim (final):")
        for key in kl_keys:
            _, v = first_last(key)
            add(f"    {key[3:]:<4} {v:9.4f}")
        worst = max(kl_keys, key=lambda k: _window_mean(_values(k), first=False))
        add(f"      -> dominated by {worst[3:]}")

    # --- 6. Headline losses + freeze integrity -------------------------------------------------
    add("")
    if has("loss_flow"):
        f_first, f_last = first_last("loss_flow")
        add(f"loss_flow  : {f_first:.4f} -> {f_last:.4f}")
        if f_last > f_first:
            verdicts.append("loss_flow did NOT decrease: the run has a problem beyond the physical branch.")
    if has("loss"):
        t_first, t_last = first_last("loss")
        add(f"loss       : {t_first:.4f} -> {t_last:.4f}")
    if has("param_norm"):
        vals = _finite(series["param_norm"])
        drift = (max(vals) - min(vals)) / (abs(vals[0]) + 1e-9)
        add(f"param_norm : {vals[0]:.4f} -> {vals[-1]:.4f}   relative drift={drift:.2e}")
        if drift > 1e-4:
            verdicts.append(
                f"param_norm drifted {drift:.1e} -- for a frozen-VLM config this should be ~1e-7 (bf16 "
                "rounding). The freeze filter may not be holding."
            )
        else:
            add("      -> frozen VLM intact (drift is bf16 rounding noise).")

    # --- 7. Verdicts ---------------------------------------------------------------------------
    missing = [k for k in _REQUIRED_DIAGNOSTICS if k not in series]

    add("")
    add("-" * 78)
    if missing:
        add(f"INCONCLUSIVE: {len(missing)} diagnostic(s) absent from this run's metrics.csv:")
        add(f"    {', '.join(missing)}")
        add("    This run predates them (or the physical branch was off). The checks that depend on")
        add("    them did NOT run -- absence of findings below is NOT evidence of health. Re-run to")
        add("    get a conclusive diagnosis.")
        add("")
    if verdicts:
        add(f"FINDINGS ({len(verdicts)}):")
        for i, v in enumerate(verdicts, 1):
            add(f"  {i}. {v}")
    elif missing:
        add("FINDINGS: none from the checks that could run (see INCONCLUSIVE above).")
    else:
        add("FINDINGS: none -- every diagnostic ran and none tripped. The physical branch looks healthy.")
    add("=" * 78)
    return "\n".join(out)


def write_summary(metrics_path: pathlib.Path, out_path: pathlib.Path, *, config_line: str = "") -> str:
    text = build_summary(metrics_path, config_line=config_line)
    out_path.write_text(text + "\n")
    return text
