"""Minimal websocket client for the force-aware pi0 policy (ForceVLA-style).

Start the policy server first (in a separate terminal), e.g. for the M1 (force-token) config:

  uv run scripts/serve_policy.py policy:checkpoint \\
    --policy.config=pi0_force_token \\
    --policy.dir=checkpoints/pi0_force_token/force_m1/2999

Then run this client:

  uv run examples/force/main.py

By default it sends the current DraftVLA contract, a 57-D ``observation/contact_input`` =
``[left_data_zeroed(25), right_data_zeroed(25), gripper_width(1), force_torque_zeroed(6)]`` (see
``draftvla_contact.py``). Pass ``--no-draftvla`` for legacy ``pi0_force_*`` checkpoints that expect
the flange ``observation/force_torque`` (6). On a real robot, replace ``_random_observation`` with live
sensor reads.

A DraftVLA server answers with ``actions`` (8, 7) plus the physical-branch readouts of the same
forward pass: ``safe_force_distribution`` [mu_hat, sigma_hat], ``prototype_probs`` [K=6] and
``z_phy`` [128]. They are diagnostics to log per frame (notes/eval_logging_spec.md §4), never control
inputs; ForceVLA / no-force checkpoints return ``actions`` only.
"""

import dataclasses
import logging

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    """Command line arguments."""

    host: str = "0.0.0.0"
    port: int | None = 8000
    api_key: str | None = None
    num_steps: int = 5
    prompt: str = "grasp the banana"
    draftvla: bool = True


def _random_observation(prompt: str, *, draftvla: bool) -> dict:
    # The state and contact layouts MUST match the relevant conversion script:
    #   state   = [tcp_position_xyz(3), tcp_rotation_vector(3), gripper_width(1)]  -> (7,)
    #   DraftVLA contact_input = [left_data_zeroed(25), right_data_zeroed(25),
    #                             gripper_width(1), force_torque_zeroed(6)]        -> (57,)
    #     (a real robot client must apply the SAME per-episode fingertip zeroing --
    #      see examples/force/draftvla_contact.py)
    #   legacy wrench = [fx, fy, fz, tx, ty, tz]                                   -> (6,)
    observation = {
        "observation/state": np.random.rand(7),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": prompt,
    }
    wrench_key = "observation/contact_input" if draftvla else "observation/force_torque"
    observation[wrench_key] = np.random.rand(57 if draftvla else 6)
    return observation


def main(args: Args) -> None:
    policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port, api_key=args.api_key)
    logger.info(f"Server metadata: {policy.get_server_metadata()}")

    for step in range(args.num_steps):
        action = policy.infer(_random_observation(args.prompt, draftvla=args.draftvla))
        actions = np.asarray(action["actions"])
        logger.info(f"[step {step}] action chunk shape: {actions.shape}")  # expected (action_horizon, 7)
        # Present on a DraftVLA (physical branch) server only; absent on ForceVLA / no-force arms.
        readouts = {k: np.asarray(action[k]).shape for k in ("safe_force_distribution", "prototype_probs", "z_phy") if k in action}
        if readouts:
            logger.info(f"[step {step}] physical readouts: {readouts}")  # expected (2,), (6,), (128,)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
