"""Physical Interaction Token (DraftVLA): z_phy and its three heads.

Implements `outlines/draftvla_plan.md` (v2.1) §6.2. The FVLMoE's contextualized force token is
projected into a compact physical token `z_phy`, which feeds three branches:

  1. `PhysicalActionProjector`   -> G_phy [b, T, d_act], added to the action hidden states as
     *additive guidance* alongside ForceVLA's own G_fvl (plan rule 3). This is what makes the
     physical branch affect action generation rather than being an auxiliary classifier.
  2. `SafeForceDistributionHead` -> the stage-level safe mean-wrench distribution (mu, sigma), [b, 6] each.
  3. `PrototypeClassifier`       -> soft prototype logits [b, K].

Normalized label space (plan §6.2)
----------------------------------
The head predicts the distribution in a *normalized* space and de-normalizes on the way out. This
is a change of variable on the wrench itself, x' = (x - m) / s, so:

    mu'    = (mu - m) / s          sigma' = sigma / s

Both use the SAME per-dim scale `s`, which is what makes the Gaussian KL provably identical in the
two spaces (KL is invariant under a per-dim affine change of variable). Using different scales for
mu and sigma -- e.g. the per-column std of the 12-vector -- silently changes the objective; do not.

The benefit is *conditioning*, not loss reweighting: raw labels span mu_fy ~ 0.001 to mu_tx ~ 136,
so one step of a shared learning rate on the head's last layer means wildly different things per
column. With s = the label's own median sigma (also the natural preconditioner, since
dKL/dmu_pred = (mu_pred - mu_gt) / sigma_pred^2), normalized mu lands within ~[-3.5, 3.2] and
normalized sigma within ~[0.16, 4.3] on this dataset.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at

# Floor for sigma in the KL, from plan v1 §8.2. It never binds on this dataset (the smallest label
# sigma is 0.2864); it is a guard against a degenerate predicted or dummy sigma.
SIGMA_FLOOR = 1e-4


class PhysicalProjector(nnx.Module):
    """FVLMoE contextualized force token -> z_phy (plan §6.2)."""

    def __init__(self, d_vlm: int, d_phy: int, *, hidden: int = 512, rngs: nnx.Rngs):
        self.fc_in = nnx.Linear(d_vlm, hidden, rngs=rngs)
        self.norm = nnx.LayerNorm(hidden, rngs=rngs)
        self.fc_out = nnx.Linear(hidden, d_phy, rngs=rngs)

    @at.typecheck
    def __call__(self, fused_force_token: at.Float[at.Array, "b d"]) -> at.Float[at.Array, "b p"]:
        x = jax.nn.gelu(self.fc_in(fused_force_token))
        x = self.norm(x)
        x = self.fc_out(x)
        # L2 normalize: z_phy is a direction, keeping the three downstream heads on a common scale.
        return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)


class PhysicalActionProjector(nnx.Module):
    """z_phy -> per-horizon additive action guidance G_phy (plan §6.2 / rule 3)."""

    def __init__(self, d_phy: int, d_act: int, action_horizon: int, *, hidden: int = 512, rngs: nnx.Rngs):
        self.action_horizon = action_horizon
        self.d_act = d_act
        self.fc_in = nnx.Linear(d_phy, hidden, rngs=rngs)
        self.fc_out = nnx.Linear(hidden, action_horizon * d_act, rngs=rngs)

    @at.typecheck
    def __call__(self, z_phy: at.Float[at.Array, "b p"]) -> at.Float[at.Array, "b t d"]:
        x = jax.nn.gelu(self.fc_in(z_phy))
        x = self.fc_out(x)
        return x.reshape(x.shape[0], self.action_horizon, self.d_act)


class SafeForceDistributionHead(nnx.Module):
    """z_phy -> stage-level safe wrench distribution, in NORMALIZED label space (plan §6.2).

    Returns `(mu_norm, sigma_norm)`, each [b, 6]. Stage-level only: the prediction is [b, 12], never
    a true temporal [b, T, 12] (plan rule 4), and the loss must never be summed over T (rule 5).
    """

    def __init__(self, d_phy: int, safe_force_dim: int = 6, *, hidden: int = 256, rngs: nnx.Rngs):
        self.safe_force_dim = safe_force_dim
        self.fc_in = nnx.Linear(d_phy, hidden, rngs=rngs)
        self.fc_out = nnx.Linear(hidden, 2 * safe_force_dim, rngs=rngs)

    @at.typecheck
    def __call__(self, z_phy: at.Float[at.Array, "b p"]) -> tuple[at.Float[at.Array, "b f"], at.Float[at.Array, "b f"]]:
        x = jax.nn.gelu(self.fc_in(z_phy))
        raw = self.fc_out(x)
        mu_norm = raw[:, : self.safe_force_dim]
        sigma_norm = jax.nn.softplus(raw[:, self.safe_force_dim :]) + 1e-6
        return mu_norm, sigma_norm


class PrototypeClassifier(nnx.Module):
    """z_phy -> soft prototype logits [b, K] (plan §6.2).

    A plain classifier over *fixed offline* prototypes: the K-means centers are metadata, never
    trainable parameters, and there are no learnable latent prototype vectors (plan rule 8).
    """

    def __init__(self, d_phy: int, num_prototypes: int, *, rngs: nnx.Rngs):
        self.fc_in = nnx.Linear(d_phy, d_phy, rngs=rngs)
        self.fc_out = nnx.Linear(d_phy, num_prototypes, rngs=rngs)

    @at.typecheck
    def __call__(self, z_phy: at.Float[at.Array, "b p"]) -> at.Float[at.Array, "b k"]:
        return self.fc_out(jax.nn.gelu(self.fc_in(z_phy)))


@at.typecheck
def diagonal_gaussian_kl_per_dim(
    mu_gt: at.Float[at.Array, "b f"],
    sigma_gt: at.Float[at.Array, "b f"],
    mu_pred: at.Float[at.Array, "b f"],
    sigma_pred: at.Float[at.Array, "b f"],
) -> at.Float[at.Array, "b f"]:
    """KL(p_gt || p_pred) per wrench dimension, un-reduced (plan §7.2).

    Kept separate from `diagonal_gaussian_kl` so diagnostics can attribute the loss to a dimension
    without recomputing it.
    """
    sigma_gt = jnp.maximum(sigma_gt, SIGMA_FLOOR)
    sigma_pred = jnp.maximum(sigma_pred, SIGMA_FLOOR)
    return jnp.log(sigma_pred / sigma_gt) + (sigma_gt**2 + (mu_gt - mu_pred) ** 2) / (2.0 * sigma_pred**2) - 0.5


@at.typecheck
def diagonal_gaussian_kl(
    mu_gt: at.Float[at.Array, "b f"],
    sigma_gt: at.Float[at.Array, "b f"],
    mu_pred: at.Float[at.Array, "b f"],
    sigma_pred: at.Float[at.Array, "b f"],
) -> at.Float[at.Array, " b"]:
    """KL(p_gt || p_pred) for diagonal Gaussians, averaged over the 6 wrench dims (plan §7.2).

    Forward KL: mode-covering, and equivalent to moment matching. Returns per-batch-element values
    so the caller can mask before reducing.
    """
    return jnp.mean(diagonal_gaussian_kl_per_dim(mu_gt, sigma_gt, mu_pred, sigma_pred), axis=-1)


@at.typecheck
def mean_pairwise_cosine(z: at.Float[at.Array, "b p"]) -> at.Float[at.Array, ""]:
    """Mean off-diagonal cosine similarity within the batch -- the z_phy collapse detector.

    `z_phy` is L2-normalized, so this is just the mean off-diagonal Gram entry. ~1.0 means every
    sample maps to the same direction: the token carries no per-sample information and the heads
    downstream of it cannot discriminate no matter how long they train. Near 0 means spread.
    """
    b = z.shape[0]
    gram = z @ z.T
    off_diagonal_sum = jnp.sum(gram) - jnp.trace(gram)
    return off_diagonal_sum / jnp.maximum(b * (b - 1), 1)


@at.typecheck
def categorical_entropy(logits: at.Float[at.Array, "b k"]) -> at.Float[at.Array, " b"]:
    """H(softmax(logits)) per batch element -- the prototype-collapse detector.

    At ~log(K) the classifier is emitting a uniform distribution, i.e. it has learned nothing. That
    is distinguishable from "learned but wrong", which shows up as low entropy + low accuracy.
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1)


@at.typecheck
def masked_mean(values: at.Float[at.Array, " b"], valid: at.Bool[at.Array, " b"]) -> at.Float[at.Array, ""]:
    """Mean of `values` over `valid` only, safe when nothing is valid (plan §7.2).

    43.6% of kept frames carry no physical label, so the effective batch varies per step and can be empty.
    """
    valid_f = valid.astype(values.dtype)
    return jnp.sum(values * valid_f) / jnp.maximum(jnp.sum(valid_f), 1.0)


@at.typecheck
def soft_label_cross_entropy(
    logits: at.Float[at.Array, "b k"], targets: at.Float[at.Array, "b k"]
) -> at.Float[at.Array, " b"]:
    """-sum(y * log_softmax(logits)), per batch element (plan §7.3)."""
    return -jnp.sum(targets * jax.nn.log_softmax(logits, axis=-1), axis=-1)


@at.typecheck
def target_entropy(targets: at.Float[at.Array, "b k"]) -> at.Float[at.Array, " b"]:
    """H(y): the soft cross-entropy's lower bound (plan §7.3).

    L_proto is bounded below by the target's own entropy, which grows with tau_q -- so comparing raw
    L_proto across a tau ablation compares nothing. Always log `L_proto - H(y)` too.
    """
    return -jnp.sum(targets * jnp.log(jnp.clip(targets, 1e-9, 1.0)), axis=-1)
