"""Single source of truth for the DraftVLA contact-input contract (todo_training_contract.md, Plan B).

Everything that both the label builder (build_safe_group_prototypes.py) and the LeRobot converter
(convert_draftvla_data_to_lerobot.py) must agree on lives here, so the two can never drift apart:

  - the 8 physical conditions and the 16 (task, condition, stage_group) supervision groups;
  - the episode denylist (contract §0b, decided 2026-09-17);
  - per-episode tactile zeroing (contract §C);
  - the scalar grip signal (contract §B) and the 57-D contact input (contract §A);
  - the descriptor features (contract §D): contact area, centre-of-pressure row, stiffness.

Zeroing and descriptor formulas are ALIGNED 1:1 with outlines/analyze_groups.py (the script that
produced the K-scan in data_analysis_20260916.md), reconciled 2026-09-17:
  - zero window: frames with stage=='prepare' AND max|cmd_speed_l| < 1e-6, first 15, min 5;
    baseline = MEAN over the window (not necessarily contiguous). All 83 episodes have 15.
  - per-episode taxel noise = std of the window residuals; contact threshold = max(5*noise, 1e-4).
  - area = per-frame (count_left + count_right) / 2 above the threshold.
  - CoP = intensity-weighted mean ROW index (arange(25)//5) of the combined positive residuals,
    over frames whose total exceeds the threshold.
  - stiffness = lstsq slope of grip vs closure (w_max_in_grasp - w) over the grasp stage;
    needs >= 4 grasp frames and closure ptp > 1e-4.
Group (mu, sigma) labels do NOT depend on these features.
"""

import dataclasses
import json
import pathlib

import numpy as np

TASK = "pick_place"
# 6 fruit/vegetable categories + the 2 carton load conditions = 8 conditions (contract §E).
FRUIT_CONDITIONS = ("apple", "banana", "cucumber", "kiwi", "potato", "tomato")
CARTON_CONDITIONS = ("carton_empty", "carton_full")
CONDITIONS = FRUIT_CONDITIONS + CARTON_CONDITIONS

# Stage annotation keeps the four contact stages; the distribution target pools them into two
# stage groups (contract §0b: F_stage for mu is 0.11 -- the force barely changes once closed).
CONTACT_STAGES = ("grasp", "lift", "translate", "place")
STAGE_TO_GROUP = {"grasp": "grasp", "lift": "hold", "translate": "hold", "place": "hold"}
STAGE_GROUPS = ("grasp", "hold")

GROUP_KEYS = tuple((TASK, cond, sg) for cond in CONDITIONS for sg in STAGE_GROUPS)  # 16 groups
GROUP_INDEX = {key: i for i, key in enumerate(GROUP_KEYS)}

# Grip-scalar noise floor from data_analysis_20260916.md (single-taxel noise 0.00069 * sqrt(50),
# rounded up); the sigma labels are clamped to it. The per-episode CONTACT threshold is derived
# from that episode's own zero-window noise (analyze_groups.py), not from this constant.
GRIP_SIGMA_FLOOR = 0.0051

# analyze_groups.py: stage=='prepare' AND all-zero command, first 15 such frames, at least 5.
CMD_STILL_THRESHOLD = 1e-6
ZERO_WINDOW_MIN = 5
ZERO_WINDOW_MAX = 15

CONTACT_INPUT_DIM = 57  # left_data(25) + right_data(25) + gripper_width(1) + force_torque_zeroed(6)
CONTACT_INPUT_SIGNAL = "tactile_data_zeroed_50_plus_gripper_width_plus_force_torque_zeroed"

# Episode denylist (contract §0b + data_analysis §6; banana 234727 confirmed 2026-09-17:
# hold-stage grip is negative (-0.645) at a normal aperture -- a baseline/sensor fault, not a light
# grasp). Denylisted episodes are excluded from the group statistics AND never converted.
DEFAULT_EXCLUDE = (
    "pi0_train_20260916_015938_carton_02_full",  # 3xMAD outlier; 25/25 taxels saturated; session warmup
    "pi0_train_20260911_002313",  # cucumber, 3xMAD outlier (closed to 21.9 mm)
    "pi0_train_20260911_003740",  # cucumber, 3xMAD outlier (closed to 24.3 mm)
    "pi0_train_20260910_234336",  # kiwi, 3xMAD outlier
    "pi0_train_20260903_234527",  # banana, hold grip ~ noise (0.016): never actually gripped
    "pi0_train_20260903_234727",  # banana, hold grip NEGATIVE (-0.645): sensor/baseline fault
)


def load_records(episode_dir: pathlib.Path) -> list[dict]:
    with (episode_dir / "observations.jsonl").open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    records.sort(key=lambda r: r["index"])
    return records


def episode_condition(records: list[dict], episode_name: str) -> str:
    """Resolve the physical condition (contract §E). Carton episodes carry episode_context on every
    frame; fruit episodes predate that field, so the condition is parsed from the fixed prompt."""
    ctx = records[0].get("episode_context")
    if ctx:
        load = ctx.get("load_condition")
        if load not in ("empty", "full"):
            raise ValueError(f"{episode_name}: episode_context.load_condition = {load!r}")
        return f"carton_{load}"
    prompt = records[0].get("prompt", "")
    for fruit in FRUIT_CONDITIONS:
        if f"grasp the {fruit} " in prompt:
            return fruit
    raise ValueError(f"{episode_name}: cannot resolve condition from prompt {prompt!r}")


@dataclasses.dataclass(frozen=True)
class TactileZeroing:
    left0: np.ndarray  # [25]
    right0: np.ndarray  # [25]
    window_indices: tuple[int, ...]  # frame indices used (need not be contiguous)
    noise: float  # std of the window residuals (both fingers) -- per-episode taxel noise

    @property
    def window_size(self) -> int:
        return len(self.window_indices)

    @property
    def contact_threshold(self) -> float:
        """Per-taxel contact threshold (analyze_groups.py): max(5 * noise, 1e-4)."""
        return max(5.0 * self.noise, 1e-4)


def tactile_zeroing(records: list[dict], episode_name: str) -> TactileZeroing:
    """Per-episode fingertip baseline (analyze_groups.py `episode_features`): MEAN of left/right
    `_data` over the first (up to) 15 frames with stage=='prepare' AND an all-zero cmd_speed_l.
    The frames need not be contiguous; all 83 episodes have the full 15."""
    idx, left_rows, right_rows = [], [], []
    for i, r in enumerate(records):
        cmd = r.get("cmd_speed_l") or [0.0] * 6
        if r.get("stage") != "prepare" or max(abs(float(x)) for x in cmd) >= CMD_STILL_THRESHOLD:
            continue
        tv = r["tactile_voltage_signals"]
        idx.append(i)
        left_rows.append(np.asarray(tv["left_data"], dtype=np.float64))
        right_rows.append(np.asarray(tv["right_data"], dtype=np.float64))
        if len(idx) >= ZERO_WINDOW_MAX:
            break
    if len(idx) < ZERO_WINDOW_MIN:
        raise ValueError(
            f"{episode_name}: only {len(idx)} still prepare frame(s) "
            f"(need >= {ZERO_WINDOW_MIN}) -- cannot zero the fingertip baseline; exclude this episode."
        )
    left = np.stack(left_rows)
    right = np.stack(right_rows)
    left0 = left.mean(axis=0)
    right0 = right.mean(axis=0)
    noise = float(np.std(np.concatenate([left - left0, right - right0])))
    return TactileZeroing(left0=left0, right0=right0, window_indices=tuple(idx), noise=noise)


def zeroed_tactile(record: dict, zeroing: TactileZeroing) -> tuple[np.ndarray, np.ndarray]:
    tv = record["tactile_voltage_signals"]
    left = np.asarray(tv["left_data"], dtype=np.float64) - zeroing.left0
    right = np.asarray(tv["right_data"], dtype=np.float64) - zeroing.right0
    return left, right


def grip_scalar(left_zeroed: np.ndarray, right_zeroed: np.ndarray) -> float:
    """Contract §B: grip_t = 0.5 * (sum(left_data_zeroed) + sum(right_data_zeroed))."""
    return float(0.5 * (left_zeroed.sum() + right_zeroed.sum()))


def contact_input_57(record: dict, zeroing: TactileZeroing, episode_name: str) -> np.ndarray:
    """Contract §A: [left_data_zeroed(25), right_data_zeroed(25), gripper_width(1), force_torque_zeroed(6)]."""
    left, right = zeroed_tactile(record, zeroing)
    if left.shape != (25,) or right.shape != (25,):
        raise ValueError(f"{episode_name} frame {record.get('index')}: taxel shape {left.shape}/{right.shape}")
    ft = np.asarray(record["force_torque_zeroed"], dtype=np.float64)
    if ft.shape != (6,):
        raise ValueError(f"{episode_name} frame {record.get('index')}: force_torque_zeroed shape {ft.shape}")
    x = np.concatenate([left, right, [float(record["gripper_width"])], ft]).astype(np.float32)
    if x.shape != (CONTACT_INPUT_DIM,) or not np.all(np.isfinite(x)):
        raise ValueError(f"{episode_name} frame {record.get('index')}: bad contact input")
    return x


# --- Descriptor features (contract §D) --------------------------------------------------------
# Formulas 1:1 with outlines/analyze_groups.py (reconciled 2026-09-17).

_TAXEL_ROW = (np.arange(25) // 5).astype(np.float64)  # row index 0..4 of each taxel (5x5 grid)


def contact_area(left_zeroed: np.ndarray, right_zeroed: np.ndarray, threshold: float) -> float:
    """Per-finger count of taxels above the per-episode threshold, AVERAGED over the two fingers."""
    return 0.5 * (float((left_zeroed > threshold).sum()) + float((right_zeroed > threshold).sum()))


def cop_row(left_zeroed: np.ndarray, right_zeroed: np.ndarray, threshold: float) -> float | None:
    """Intensity-weighted mean row index of the combined positive residuals (both fingers summed).
    None when the total intensity does not exceed the threshold (no meaningful contact)."""
    v = np.clip(left_zeroed, 0.0, None) + np.clip(right_zeroed, 0.0, None)
    total = float(v.sum())
    if total <= threshold:
        return None
    return float(v @ _TAXEL_ROW / total)


def episode_stiffness(grip: np.ndarray, gripper_width: np.ndarray, stages: np.ndarray) -> float | None:
    """Least-squares slope of grip (V) vs closure (m) over the grasp stage, closure measured from
    the WIDEST aperture within the stage. Needs >= 4 grasp frames and closure ptp > 1e-4 m."""
    m = stages == "grasp"
    if m.sum() < 4:
        return None
    closure = gripper_width[m].max() - gripper_width[m]
    if np.ptp(closure) <= 1e-4:
        return None
    a = np.vstack([closure, np.ones_like(closure)]).T
    return float(np.linalg.lstsq(a, grip[m], rcond=None)[0][0])
