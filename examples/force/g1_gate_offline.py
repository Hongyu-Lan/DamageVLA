"""G1 offline gate (ICLR2027-DraftVLA/notes/eval_logging_spec.md §2): after contact, does the
distribution head separate the empty and the filled carton on HELD-OUT demonstrations?

This is the go/no-go before spending real-robot trials. It runs a trained checkpoint through the
exact serving path -- policy_config.create_trained_policy -> Policy.infer (bfloat16, checkpoint norm
stats, DraftVLAInputs/DraftVLAOutputs) -- over the frames of held-out episodes, and records per frame
what the server returns: the action chunk and, for a model with the physical branch, the readouts
safe_force_distribution [mu_hat, sigma_hat], prototype_probs and z_phy. Carton episodes are aligned
on the moment the fingers stop closing and the two conditions are compared before and after it.

It also runs on the arms WITHOUT the branch (`pi0_draftvla_task12_forcevla`, `..._noforce`). Those
have no distribution head, so no mu_hat/KL exists for them; what is comparable across all three arms
offline is the ACTION: error against the demonstrated action on held-out frames (Table app-offline
"Action MSE") and, the behavioural surrogate of Q1/Q2, whether the predicted gripper target after
contact differs between the empty and the filled carton -- the two look identical, so a policy that
does not read contact must predict the same closure for both.

Observations are rebuilt exactly as convert_draftvla_data_to_lerobot.py builds training frames:
  - state(7)          = tcp position + rotation vector + measured gripper width
  - contact_input(57) = draftvla_contact.contact_input_57 with the per-episode prepare-window zero
                        (the no-force arm's transforms ignore it)
  - images            = record's image_crop_box / wrist_image_crop_box applied to the ORIGINAL
                        1080p frame, then LANCZOS resize to 224x224 (postprocess/apply_image_crop_*.py).
                        The local post-processed copy carries no images, hence --images-root.
  - prompt            = the record's prompt

Outputs (--out-dir):
  frames.csv        one row per evaluated frame (readouts, targets, per-frame KL, actions, alignment)
  z_phy.npy         [N, 128] tokens in frames.csv row order (physical-branch arms only; §4.2)
  summary.json      per-episode / per-condition windows, action metrics, separation, verdict, definitions
  g1_gate.png       mu_hat / sigma_hat (or gripper targets) and measured grip vs time from closure stop

Usage:
  uv run examples/force/g1_gate_offline.py --config-name pi0_draftvla_task12 \
      --checkpoint-dir checkpoints/pi0_draftvla_task12/task12_full/9999 --out-dir examples/force/g1_gate_20260921
Add --all-val for every held-out episode with local frames, or --max-frames-per-episode 5 for a
pipeline smoke test.
"""

import argparse
import csv
import json
import pathlib
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import draftvla_contact as contact  # noqa: E402

from openpi.policies import policy_config as _policy_config  # noqa: E402
import openpi.training.config as _config  # noqa: E402

_HERE = pathlib.Path(__file__).resolve()
_REPO = _HERE.parents[2]
_DEFAULT_RECORDS_ROOT = _REPO.parent / "DamageVLA_training_post_process_20260916"
_DEFAULT_IMAGES_ROOT = _REPO.parents[1] / "ros" / "joystick_ur_ws" / "pi0_training_logs"
_DEFAULT_LABELS_DIR = _REPO.parent / "task12_provenance" / "prototype_metadata_task12_trainonly"
_DEFAULT_VAL_LIST = _HERE.parent / "val_episodes_20260917.txt"

IMAGE_SIZE = 224  # postprocess/apply_image_crop_20260911.py SIZE
CONTACT_STAGES = ("grasp", "lift", "translate", "place")
READOUT_KEYS = ("safe_force_distribution", "prototype_probs", "z_phy")
ACTION_NAMES = ("vx", "vy", "vz", "wx", "wy", "wz", "grip")
FPS = 10.0

# Alignment definitions (eval_logging_spec.md §9 asks for these to be fixed up front and recorded).
CLOSURE_STOP_TOL_M = 0.3e-3  # |dw| below this for CLOSURE_STOP_FRAMES consecutive frames = fingers stopped
CLOSURE_STOP_FRAMES = 3
GRIP_ONSET_MULT = 10.0  # grip > 10 x sigma floor (0.051 V) for GRIP_ONSET_FRAMES frames, aperture < 45 mm
GRIP_ONSET_FRAMES = 3
GRIP_ONSET_MAX_APERTURE_M = 0.045
PRE_WINDOW = (-30, -5)  # frames relative to closure stop: "before contact"
POST_WINDOW = (5, 40)  # frames relative to closure stop: "after contact"


def _read_episode_list(path: pathlib.Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def _load_group_table(labels_dir: pathlib.Path) -> dict[tuple[str, str], dict]:
    meta = json.loads((labels_dir / "group_metadata.json").read_text())
    return {(g["condition"], g["stage_group"]): g for g in meta["groups"]}


def _crop_resize(path: pathlib.Path, box: list[int]) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.size == (IMAGE_SIZE, IMAGE_SIZE):
        return np.asarray(image)  # already post-processed
    return np.asarray(image.crop(tuple(int(v) for v in box)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS))


def _gaussian_kl(mu_gt: float, sigma_gt: float, mu_pred: float, sigma_pred: float) -> float:
    """KL(N(gt) || N(pred)) for scalars -- physical.diagonal_gaussian_kl_per_dim, in numpy."""
    sigma_gt = max(sigma_gt, 1e-4)
    sigma_pred = max(sigma_pred, 1e-4)
    return float(np.log(sigma_pred / sigma_gt) + (sigma_gt**2 + (mu_gt - mu_pred) ** 2) / (2.0 * sigma_pred**2) - 0.5)


CLOSURE_FINAL_TOL_M = 1.0e-3  # "closed": within 1 mm of the tightest width reached in the contact stages


def _closure_stop(width: np.ndarray, first_grasp: int, last_contact: int) -> int | None:
    """First frame at/after the grasp annotation where the fingers have reached their final width.

    Demonstrations use staged closure (the operator pauses mid-way), so "width stopped changing for a
    few frames" fires on the first pause, still in free air. Anchoring on the tightest width of the
    contact stages instead is pause-proof: the first frame within CLOSURE_FINAL_TOL_M of it that also
    holds still for CLOSURE_STOP_FRAMES frames.
    """
    w_min = float(np.min(width[first_grasp : last_contact + 1]))
    for i in range(max(first_grasp, 1), len(width) - CLOSURE_STOP_FRAMES):
        window = width[i : i + CLOSURE_STOP_FRAMES + 1]
        if width[i] <= w_min + CLOSURE_FINAL_TOL_M and np.all(np.abs(np.diff(window)) < CLOSURE_STOP_TOL_M):
            return i
    return None


def _grip_onset(grip: np.ndarray, width: np.ndarray) -> int | None:
    thr = GRIP_ONSET_MULT * contact.GRIP_SIGMA_FLOOR
    hits = (grip > thr) & (width < GRIP_ONSET_MAX_APERTURE_M)
    for i in range(len(hits) - GRIP_ONSET_FRAMES + 1):
        if hits[i : i + GRIP_ONSET_FRAMES].all():
            return i
    return None


def _select_frames(records: list[dict], prepare_frames: int, max_frames: int | None) -> list[int]:
    stages = [r.get("stage") for r in records]
    contact_idx = [i for i, s in enumerate(stages) if s in CONTACT_STAGES]
    if not contact_idx:
        return []
    first = contact_idx[0]
    lead = [i for i in range(max(0, first - prepare_frames), first) if stages[i] == "prepare"]
    chosen = lead + contact_idx
    if max_frames is not None:
        chosen = chosen[:max_frames]
    return chosen


def evaluate_episode(
    policy, episode, records_root, images_root, groups, *, expect_phy, label_prior, prepare_frames, max_frames, log
):
    rec_dir = records_root / episode
    img_dir = images_root / episode
    records = contact.load_records(rec_dir)
    if not (img_dir / records[0]["image_path"]).exists():
        raise FileNotFoundError(f"{episode}: no frames under {img_dir}")
    # The post-processed records and the original frames must be the same episode, frame for frame.
    if (img_dir / "observations.jsonl").exists():
        raw = contact.load_records(img_dir)
        if len(raw) != len(records):
            raise ValueError(f"{episode}: {len(records)} post-processed records vs {len(raw)} raw frames")
        step = max(1, len(records) // 8)
        for a, b in zip(records[::step], raw[::step], strict=True):
            if abs(float(a["stamp_sec"]) - float(b["stamp_sec"])) > 1e-6:
                raise ValueError(f"{episode}: stamp mismatch at index {a['index']} between records and frames")

    condition = contact.episode_condition(records, episode)
    zeroing = contact.tactile_zeroing(records, episode)
    chosen = _select_frames(records, prepare_frames, max_frames)

    width = np.asarray([r["gripper_width"] for r in records], dtype=np.float64)
    grip_all = np.asarray([contact.grip_scalar(*contact.zeroed_tactile(r, zeroing)) for r in records])
    stages = [r.get("stage") for r in records]
    first_grasp = next(i for i, s in enumerate(stages) if s in CONTACT_STAGES)
    last_contact = max(i for i, s in enumerate(stages) if s in CONTACT_STAGES)
    t_stop = _closure_stop(width, first_grasp, last_contact)
    t_grip = _grip_onset(grip_all, width)

    rows, tokens = [], []
    t_start = time.monotonic()
    for n, i in enumerate(chosen):
        r = records[i]
        tcp = r["tcp_pose"]
        obs = {
            "observation/image": _crop_resize(img_dir / r["image_path"], r["image_crop_box"]),
            "observation/wrist_image": _crop_resize(img_dir / r["wrist_image_path"], r["wrist_image_crop_box"]),
            "observation/state": np.asarray(
                [*tcp["position_xyz"], *tcp["rotation_vector"], r["gripper_width"]], dtype=np.float32
            ),
            "observation/contact_input": contact.contact_input_57(r, zeroing, episode),
            "prompt": r["prompt"],
        }
        out = policy.infer(obs)
        has_phy = all(k in out for k in READOUT_KEYS)
        if expect_phy and not has_phy:
            raise RuntimeError("policy response lacks the physical readouts: the served source predates sample_actions_with_physical")

        stage_group = contact.STAGE_TO_GROUP.get(r.get("stage"))
        g = groups.get((condition, stage_group)) if stage_group else None
        actions = np.asarray(out["actions"], dtype=np.float64)  # (8, 7)
        demo = np.asarray(r["action_7"], dtype=np.float64)
        row = {
            "episode": episode,
            "condition": condition,
            "index": int(r["index"]),
            "stage": r.get("stage"),
            "stage_group": stage_group,
            "t_from_closure_stop": (i - t_stop) if t_stop is not None else None,
            "t_from_grip_onset": (i - t_grip) if t_grip is not None else None,
            "t_from_grasp_start": i - first_grasp,
            "gripper_width_m": float(r["gripper_width"]),
            "grip_measured": float(grip_all[i]),
            "mu_target": g["mu"] if g else None,
            "sigma_target": g["sigma"] if g else None,
            "proto_target": g["prototype"] if g else None,
            # Behaviour, comparable across all arms: the chunk's first row and its tightest gripper target.
            **{f"action_{name}_row0": float(actions[0, k]) for k, name in enumerate(ACTION_NAMES)},
            **{f"demo_{name}": float(demo[k]) for k, name in enumerate(ACTION_NAMES)},
            "action_gripper_target_min_chunk_m": float(actions[:, 6].min()),
            "action_sq_err_row0": float(np.mean((actions[0] - demo) ** 2)),
        }
        if has_phy:
            mu_hat, sigma_hat = (float(v) for v in out["safe_force_distribution"])
            probs = np.asarray(out["prototype_probs"], dtype=np.float64)
            row.update(
                {
                    "mu_hat": mu_hat,
                    "sigma_hat": sigma_hat,
                    "kl": _gaussian_kl(g["mu"], g["sigma"], mu_hat, sigma_hat) if g else None,
                    "kl_baseline": _gaussian_kl(g["mu"], g["sigma"], *label_prior) if g else None,
                    "proto_argmax": int(probs.argmax()),
                    **{f"proto_p{k}": float(p) for k, p in enumerate(probs)},
                }
            )
            tokens.append(np.asarray(out["z_phy"], dtype=np.float32))
        else:
            row.update({"mu_hat": None, "sigma_hat": None, "kl": None, "kl_baseline": None, "proto_argmax": None})
        rows.append(row)
        if n % 50 == 0 or n == len(chosen) - 1:
            phy_txt = f"mu_hat={row['mu_hat']:.3f} sigma_hat={row['sigma_hat']:.3f}" if has_phy else "no physical branch"
            log(f"  {episode} [{condition}] frame {n + 1}/{len(chosen)}  {phy_txt}  grip_tgt={actions[0, 6] * 1e3:.1f}mm  ({time.monotonic() - t_start:.0f}s)")
    meta = {
        "condition": condition,
        "closure_stop": t_stop,
        "grip_onset": t_grip,
        "first_grasp": first_grasp,
        "n_frames": len(chosen),
        "zero_window": zeroing.window_size,
        "taxel_noise": zeroing.noise,
    }
    return rows, tokens, meta


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _window_stats(rows, window):
    sel = [r for r in rows if r["t_from_closure_stop"] is not None and window[0] <= r["t_from_closure_stop"] <= window[1]]
    if not sel:
        return None
    mus = [r["mu_hat"] for r in sel if r["mu_hat"] is not None]
    return {
        "n": len(sel),
        "mu_hat_mean": float(np.mean(mus)) if mus else None,
        "mu_hat_std": float(np.std(mus)) if mus else None,
        "sigma_hat_mean": _mean([r["sigma_hat"] for r in sel]),
        "grip_measured_mean": _mean([r["grip_measured"] for r in sel]),
        "gripper_target_row0_mean_m": _mean([r["action_grip_row0"] for r in sel]),
        "gripper_target_min_chunk_mean_m": _mean([r["action_gripper_target_min_chunk_m"] for r in sel]),
        "demo_gripper_target_mean_m": _mean([r["demo_grip"] for r in sel]),
    }


def summarize(rows, episodes_meta, groups, out_dir: pathlib.Path, checkpoint: str, has_phy: bool, log):
    per_ep = {}
    for ep, meta in episodes_meta.items():
        ep_rows = [r for r in rows if r["episode"] == ep]
        stat = {
            "condition": meta["condition"],
            "alignment": meta,
            "pre": _window_stats(ep_rows, PRE_WINDOW),
            "post": _window_stats(ep_rows, POST_WINDOW),
            # Action fit against the demonstrated action, raw units (m/s, rad/s, m), first chunk row.
            "action_mse_row0": _mean([r["action_sq_err_row0"] for r in ep_rows]),
            "gripper_target_rmse_mm": float(
                1e3 * np.sqrt(np.mean([(r["action_grip_row0"] - r["demo_grip"]) ** 2 for r in ep_rows]))
            ),
        }
        for sg in contact.STAGE_GROUPS:
            sel = [r for r in ep_rows if r["stage_group"] == sg]
            stat[f"{sg}_action_mse_row0"] = _mean([r["action_sq_err_row0"] for r in sel])
            if has_phy:
                sel_kl = [r for r in sel if r["kl"] is not None]
                if sel_kl:
                    stat[f"{sg}_kl"] = float(np.mean([r["kl"] for r in sel_kl]))
                    stat[f"{sg}_kl_baseline"] = float(np.mean([r["kl_baseline"] for r in sel_kl]))
                    stat[f"{sg}_proto_acc"] = float(np.mean([r["proto_argmax"] == r["proto_target"] for r in sel_kl]))
                    stat[f"{sg}_mu_hat_mean"] = float(np.mean([r["mu_hat"] for r in sel_kl]))
        per_ep[ep] = stat

    definitions = {
        "closure_stop": (
            f"first frame >= grasp annotation within {CLOSURE_FINAL_TOL_M * 1e3:.0f} mm of the tightest width of the "
            f"contact stages and with |dw| < {CLOSURE_STOP_TOL_M * 1e3:.1f} mm for {CLOSURE_STOP_FRAMES} frames"
        ),
        "grip_onset": f"grip > {GRIP_ONSET_MULT:.0f} x {contact.GRIP_SIGMA_FLOOR} V for {GRIP_ONSET_FRAMES} frames and aperture < {GRIP_ONSET_MAX_APERTURE_M * 1e3:.0f} mm",
        "pre_window_frames": list(PRE_WINDOW),
        "post_window_frames": list(POST_WINDOW),
        "action_mse_row0": "mean over frames and the 7 action dims of (predicted chunk row 0 - demonstrated action_7)^2, raw units",
    }
    verdict = {"gate": "G1 carton separation after contact", "has_distribution_head": has_phy, "definitions": definitions}
    carton = {c: [ep for ep, s in per_ep.items() if s["condition"] == c] for c in contact.CARTON_CONDITIONS}
    if all(carton.values()):

        def window_values(cond, window, key):
            return [per_ep[ep][window][key] for ep in carton[cond] if per_ep[ep][window] and per_ep[ep][window][key] is not None]

        # Behavioural separation (all arms): predicted gripper target after contact, empty vs full.
        tgt_e = window_values("carton_empty", "post", "gripper_target_row0_mean_m")
        tgt_f = window_values("carton_full", "post", "gripper_target_row0_mean_m")
        demo_e = window_values("carton_empty", "post", "demo_gripper_target_mean_m")
        demo_f = window_values("carton_full", "post", "demo_gripper_target_mean_m")
        verdict["post_contact_gripper_target_mm"] = {
            "empty_episode_means": [1e3 * v for v in tgt_e],
            "full_episode_means": [1e3 * v for v in tgt_f],
            "gap_full_minus_empty": (1e3 * (np.mean(tgt_f) - np.mean(tgt_e))) if tgt_e and tgt_f else None,
            "demo_gap_full_minus_empty": (1e3 * (np.mean(demo_f) - np.mean(demo_e))) if demo_e and demo_f else None,
        }
        if has_phy:
            post_e, post_f = window_values("carton_empty", "post", "mu_hat_mean"), window_values("carton_full", "post", "mu_hat_mean")
            pre_e, pre_f = window_values("carton_empty", "pre", "mu_hat_mean"), window_values("carton_full", "pre", "mu_hat_mean")
            gap_target = groups[("carton_full", "grasp")]["mu"] - groups[("carton_empty", "grasp")]["mu"]
            post_gap = float(np.mean(post_f) - np.mean(post_e)) if post_e and post_f else None
            pre_gap = float(np.mean(pre_f) - np.mean(pre_e)) if pre_e and pre_f else None
            separated = bool(post_e and post_f and min(post_f) > max(post_e))
            verdict.update(
                {
                    "target_gap_grasp": gap_target,
                    "post_contact": {
                        "empty_episode_means": post_e,
                        "full_episode_means": post_f,
                        "gap": post_gap,
                        "gap_over_target_gap": (post_gap / gap_target) if post_gap is not None else None,
                        "episodes_non_overlapping": separated,
                    },
                    "pre_contact": {"empty_episode_means": pre_e, "full_episode_means": pre_f, "gap": pre_gap},
                }
            )
            passed = post_gap is not None and post_gap > 0.5 * gap_target and separated
            verdict["result"] = "PASS" if passed else "FAIL"
            verdict["rule"] = (
                "PASS iff post-contact mean(mu_hat|full) - mean(mu_hat|empty) > 0.5 x target gap AND every full "
                "episode's post-contact mean exceeds every empty episode's"
            )
        else:
            verdict["result"] = "n/a (no distribution head on this arm; see post_contact_gripper_target_mm)"
    summary = {"checkpoint": checkpoint, "has_distribution_head": has_phy, "episodes": per_ep, "verdict": verdict}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    def fmt(v, w=7, p=3):
        return f"{v:{w}.{p}f}" if isinstance(v, (int, float)) and v is not None else f"{'--':>{w}}"

    log("\n=== per-episode ===")
    for ep, s in per_ep.items():
        pre = s["pre"]["mu_hat_mean"] if s["pre"] else None
        post = s["post"]["mu_hat_mean"] if s["post"] else None
        tgt = s["post"]["gripper_target_row0_mean_m"] if s["post"] else None
        demo = s["post"]["demo_gripper_target_mean_m"] if s["post"] else None
        line = f"{ep[-26:]:28s} {s['condition']:13s} act-MSE {fmt(s['action_mse_row0'], 8, 5)}  grip-tgt RMSE {fmt(s['gripper_target_rmse_mm'], 5, 1)}mm"
        line += f"  post grip-tgt {fmt(1e3 * tgt if tgt is not None else None, 5, 1)}mm (demo {fmt(1e3 * demo if demo is not None else None, 5, 1)}mm)"
        if has_phy:
            line += (
                f" | mu_hat pre {fmt(pre)} post {fmt(post)}  KL grasp {fmt(s.get('grasp_kl'))} (base {fmt(s.get('grasp_kl_baseline'))})"
                f" hold {fmt(s.get('hold_kl'))} (base {fmt(s.get('hold_kl_baseline'))})  proto acc g/h "
                f"{fmt(s.get('grasp_proto_acc'), 4, 2)}/{fmt(s.get('hold_proto_acc'), 4, 2)}"
            )
        log(line)
    log(f"\n=== G1 verdict: {verdict.get('result')} ===")
    log(json.dumps({k: v for k, v in verdict.items() if k not in ('definitions', 'gate', 'rule')}, indent=1))
    return summary


def plot(rows, groups, out_dir: pathlib.Path, title: str, has_phy: bool):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"carton_empty": "tab:blue", "carton_full": "tab:red"}
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for ep in sorted({r["episode"] for r in rows}):
        ep_rows = [r for r in rows if r["episode"] == ep and r["t_from_closure_stop"] is not None]
        if not ep_rows:
            continue
        cond = ep_rows[0]["condition"]
        t = np.asarray([r["t_from_closure_stop"] for r in ep_rows]) / FPS
        c = colors.get(cond, "tab:gray")
        if has_phy:
            axes[0].plot(t, [r["mu_hat"] for r in ep_rows], color=c, alpha=0.8, label=f"{cond} {ep[-6:]}")
            axes[1].plot(t, [r["sigma_hat"] for r in ep_rows], color=c, alpha=0.8)
        else:
            axes[0].plot(t, [1e3 * r["action_grip_row0"] for r in ep_rows], color=c, alpha=0.8, label=f"{cond} {ep[-6:]} predicted")
            axes[0].plot(t, [1e3 * r["demo_grip"] for r in ep_rows], color=c, alpha=0.4, ls="--")
            axes[1].plot(t, [1e3 * r["gripper_width_m"] for r in ep_rows], color=c, alpha=0.8)
        axes[2].plot(t, [r["grip_measured"] for r in ep_rows], color=c, alpha=0.8)
    if has_phy:
        for cond, c in colors.items():
            for sg, ls in (("grasp", "--"), ("hold", ":")):
                g = groups.get((cond, sg))
                if g:
                    axes[0].axhline(g["mu"], color=c, ls=ls, lw=1, alpha=0.6)
                    axes[1].axhline(g["sigma"], color=c, ls=ls, lw=1, alpha=0.6)
        axes[0].set_ylabel("mu_hat (V)")
        axes[1].set_ylabel("sigma_hat (V)")
        sub = "dashed: group target grasp, dotted: hold"
    else:
        axes[0].set_ylabel("gripper target (mm)\nsolid predicted, dashed demo")
        axes[1].set_ylabel("measured width (mm)")
        sub = "no distribution head on this arm"
    for ax in axes:
        ax.axvline(0, color="k", lw=0.8)
        ax.grid(alpha=0.3)
    axes[2].set_ylabel("measured grip (V, demo)")
    axes[2].set_xlabel("time from closure stop (s)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].set_title(f"{title}\n({sub})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "g1_gate.png", dpi=130)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", default="pi0_draftvla_task12")
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--records-root", default=str(_DEFAULT_RECORDS_ROOT), help="post-processed episodes (stage, zeroed wrench, crop boxes)")
    p.add_argument("--images-root", default=str(_DEFAULT_IMAGES_ROOT), help="original frames; cropped+resized here")
    p.add_argument("--labels-dir", default=str(_DEFAULT_LABELS_DIR))
    p.add_argument("--val-list", default=str(_DEFAULT_VAL_LIST))
    p.add_argument("--episodes", nargs="*", default=None, help="explicit episodes; default = held-out carton episodes")
    p.add_argument("--all-val", action="store_true", help="every held-out episode that has local frames")
    p.add_argument("--prepare-frames", type=int, default=30, help="prepare frames kept before the first grasp frame")
    p.add_argument("--max-frames-per-episode", type=int, default=None)
    p.add_argument("--num-steps", type=int, default=10, help="flow integration steps (serving default 10)")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = (out_dir / "run.log").open("a")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    records_root, images_root = pathlib.Path(args.records_root), pathlib.Path(args.images_root)
    val = _read_episode_list(pathlib.Path(args.val_list))
    if args.episodes:
        episodes = args.episodes
    elif args.all_val:
        episodes = [e for e in val if (images_root / e / "rgb").exists()]
    else:
        episodes = [e for e in val if "carton" in e]
    log(f"config: {args.config_name}\ncheckpoint: {args.checkpoint_dir}\nepisodes ({len(episodes)}): {episodes}")

    groups = _load_group_table(pathlib.Path(args.labels_dir))
    config = _config.get_config(args.config_name)
    expect_phy = bool(config.model.phy_enabled)
    # loss_dist_baseline semantics: the constant predictor mu_norm=0, sigma_norm=1 in label space.
    label_prior = (float(config.model.phy_label_mean[0]), float(config.model.phy_label_scale[0]))

    t0 = time.monotonic()
    policy = _policy_config.create_trained_policy(config, args.checkpoint_dir, sample_kwargs={"num_steps": args.num_steps})
    log(f"policy loaded in {time.monotonic() - t0:.0f}s  (physical branch: {expect_phy})")

    rows, tokens, episodes_meta = [], [], {}
    for ep in episodes:
        ep_rows, ep_tokens, meta = evaluate_episode(
            policy, ep, records_root, images_root, groups,
            expect_phy=expect_phy, label_prior=label_prior,
            prepare_frames=args.prepare_frames, max_frames=args.max_frames_per_episode, log=log,
        )
        rows += ep_rows
        tokens += ep_tokens
        episodes_meta[ep] = meta
        log(f"  {ep}: closure_stop={meta['closure_stop']} grip_onset={meta['grip_onset']} first_grasp={meta['first_grasp']}")

    fieldnames = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k))
    with (out_dir / "frames.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    has_phy = bool(tokens)
    if has_phy:
        np.save(out_dir / "z_phy.npy", np.stack(tokens))
    summarize(rows, episodes_meta, groups, out_dir, args.checkpoint_dir, has_phy, log)
    try:
        title = f"G1 gate -- held-out cartons, {args.config_name} @ {pathlib.Path(args.checkpoint_dir).name}"
        plot(rows, groups, out_dir, title, has_phy)
        log(f"wrote {out_dir / 'g1_gate.png'}")
    except Exception as e:  # noqa: BLE001 - the numbers matter more than the picture
        log(f"plot failed: {e!r}")
    log(f"wrote {out_dir / 'frames.csv'}{', z_phy.npy' if has_phy else ''}, summary.json  (total {time.monotonic() - t0:.0f}s)")


if __name__ == "__main__":
    main()
