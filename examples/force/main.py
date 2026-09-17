"""Minimal websocket client for the force-aware pi0 policy (ForceVLA-style).

Start the policy server first (in a separate terminal), e.g. for the M1 (force-token) config:

  uv run scripts/serve_policy.py policy:checkpoint \\
    --policy.config=pi0_force_token \\
    --policy.dir=checkpoints/pi0_force_token/force_m1/2999

Then run this client:

  uv run examples/force/main.py

By default it sends the current DraftVLA contract, a 57-D ``observation/contact_input`` containing
``[two-finger mean(6), signed half-difference(6)]``. Pass
``--no-draftvla`` for legacy ``pi0_force_*`` checkpoints that expect flange
``observation/force_torque``. On a real robot, replace ``_random_observation`` with live sensor reads.
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
