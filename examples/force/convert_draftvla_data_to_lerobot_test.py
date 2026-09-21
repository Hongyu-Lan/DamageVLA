"""v2 action contract of convert_draftvla_data_to_lerobot.py: gripper button + action_loss_weight.

Pure-function tests run everywhere; the data-backed test runs only where the 2026-09-16 post-process
batch is present and pins the counts the whole v2 plan rests on (558 close / 719 open / 34,272 hold,
39% down-weighted over the 62 training episodes).
"""

import glob
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import convert_draftvla_data_to_lerobot as conv  # noqa: E402


def _rec(target, width, stage="prepare", speed=(0.0,) * 6, action_7=None):
    a7 = list(action_7) if action_7 is not None else [*speed, target]
    return {
        "index": 0,
        "gripper_action_target": target,
        "gripper_width": width,
        "stage": stage,
        "cmd_speed_l": list(speed),
        "action_7": a7,
    }


def test_button_close_open_hold_are_the_three_anchors():
    assert conv.gripper_button_action(_rec(0.004, 0.0636), "ep") == conv.GRIPPER_CLOSE
    assert conv.gripper_button_action(_rec(0.065, 0.0500), "ep") == conv.GRIPPER_OPEN
    # no button held: the recorded target is a readback of the width (may lag by up to ~1.7 mm)
    assert conv.gripper_button_action(_rec(0.0320, 0.0320), "ep") == conv.GRIPPER_HOLD
    assert conv.gripper_button_action(_rec(0.05902, 0.05830), "ep") == conv.GRIPPER_HOLD


def test_button_rejects_a_target_that_is_neither_anchor_nor_readback():
    with pytest.raises(ValueError, match="neither a button anchor nor a readback"):
        conv.gripper_button_action(_rec(0.030, 0.045), "ep")


def test_fully_open_readback_is_hold_not_open():
    # The gripper's mechanical maximum is 64.3 mm; the open BUTTON writes exactly 65.0 mm.
    assert conv.gripper_button_action(_rec(0.0643, 0.0643), "ep") == conv.GRIPPER_HOLD


def test_action_loss_weight_rules():
    still = (0.0,) * 6
    moving = (0.0, 0.0, -0.03, 0.0, 0.0, 0.0)
    # reset: always idle, moving or not
    assert conv.action_loss_weight(_rec(0.064, 0.064, "reset", moving), conv.GRIPPER_HOLD) == conv.ACTION_WEIGHT_IDLE
    # prepare pause (still, no button): idle
    assert conv.action_loss_weight(_rec(0.064, 0.064, "prepare", still), conv.GRIPPER_HOLD) == conv.ACTION_WEIGHT_IDLE
    # prepare while descending: full
    assert conv.action_loss_weight(_rec(0.064, 0.064, "prepare", moving), conv.GRIPPER_HOLD) == conv.ACTION_WEIGHT_FULL
    # grasp: the arm is still while the gripper closes -- MUST stay full (that stillness is the label)
    assert conv.action_loss_weight(_rec(0.004, 0.050, "grasp", still), conv.GRIPPER_CLOSE) == conv.ACTION_WEIGHT_FULL
    assert conv.action_loss_weight(_rec(0.032, 0.032, "grasp", still), conv.GRIPPER_HOLD) == conv.ACTION_WEIGHT_FULL
    # lift / translate / place: full
    for stage in ("lift", "translate", "place"):
        assert conv.action_loss_weight(_rec(0.032, 0.032, stage, still), conv.GRIPPER_HOLD) == conv.ACTION_WEIGHT_FULL


def test_frame_action_7_replaces_only_dim_6():
    rec = _rec(0.004, 0.0636, "grasp", action_7=[0.01, -0.02, -0.03, 0.0, 0.0, 0.001, 0.004])
    action, weight = conv.frame_action_7(rec, "ep")
    np.testing.assert_allclose(action[:6], [0.01, -0.02, -0.03, 0.0, 0.0, 0.001])
    assert action[6] == conv.GRIPPER_CLOSE
    assert action.dtype == np.float32 and action.shape == (7,)
    assert weight == conv.ACTION_WEIGHT_FULL


_START_POSE = pathlib.Path(
    "/home/lar-ur/vla/DamageVLA/deployments/artifacts/task12-continuous-acceptance/training-start-pose.json"
)


@pytest.mark.skipif(
    not conv.DEFAULT_DATA_DIR.is_dir() or not _START_POSE.is_file(),
    reason="2026-09-16 post-process batch or the 62-episode train list is not on this machine",
)
def test_train_split_counts_match_the_v2_plan():
    episodes = json.loads(_START_POSE.read_text())["episodes"]
    counts = {conv.GRIPPER_CLOSE: 0, conv.GRIPPER_HOLD: 0, conv.GRIPPER_OPEN: 0}
    idle = total = 0
    for ep in episodes:
        files = sorted(glob.glob(str(conv.DEFAULT_DATA_DIR / ep / "*.jsonl")))
        assert files, ep
        for line in open(files[0]):
            if not line.strip():
                continue
            action, weight = conv.frame_action_7(json.loads(line), ep)
            counts[float(action[6])] += 1
            idle += weight == conv.ACTION_WEIGHT_IDLE
            total += 1
    assert total == 35549
    assert counts == {conv.GRIPPER_CLOSE: 558, conv.GRIPPER_HOLD: 34272, conv.GRIPPER_OPEN: 719}
    assert round(idle / total, 2) == 0.39
