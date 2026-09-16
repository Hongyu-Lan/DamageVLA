# Damage-Aware ForceVLA with Stage-Level Safe Interaction Prototypes
## Implementation Specification — v2.1

> **Revision note.** v2 supersedes v1 (preserved verbatim at `outlines/draftvla_plan_v1.md`).
> v1 was written before the `data/draftVLA_training_20260715` dataset existed and before the
> ForceVLA extension landed in this repo. Every number, field name, shape, and line reference below
> has been checked against the data on disk and the code in `src/openpi/`.
> v2.1 folds in an adversarial review of v2; **every claim it falsified is marked ⚠︎ REVIEW-FIX**.
> Last updated: 2026-07-17.

> **Implementation override (2026-08-25):** the current training requested by the dataset owner uses
> `concat(0.5 × (left_estimated + right_estimated), 0.5 × (left_estimated - right_estimated))` from
> `tactile_estimated_wrenches` as the 12-D model input, stored in LeRobot as `gripper_wrench` under
> repo `draftvla/fruits_tactile`. The safe-distribution target remains the aggregate mean-wrench
> `[mu×6, sigma×6]` (12-D); input `force_dim=12` and output `safe_force_dim=6` are intentionally
> independent. This supersedes locked decision D-A and the flange de-bias/deployment sections below,
> which are retained as the historical v2.1 rationale. The model now requires the tactile estimator
> at deployment and no longer makes the v2.1 claim of inferring an unmeasured fingertip wrench from
> flange F/T.

---

## 0. What changed from v1, and why

| # | v1 said | v2.1 says | Reason |
|---|---|---|---|
| 1 | `current_force [6]`, source unspecified | `force_torque` (UR flange F/T), **de-biased (mandatory)**; labels stay tactile | Decision D-A. Input modality ≠ supervised modality **by design** (§1). De-bias is a *leakage fix*, not a nicety (§4.3.2) |
| 2 | PyTorch pseudocode | JAX / Flax NNX inside openpi | Decision D-B. The force path in this repo is JAX-only |
| 3 | `N_group = 16`, but `Q.shape = [36, 12]`, `Y_target_all.shape = [36, 4]` | `N_group = 16` everywhere; `Q [16,12]`, `Y [16,4]` | v1 was internally inconsistent. Data confirms 16 |
| 4 | Loads `soft_prototype_targets_all` + indexes by `group_id` | Reads per-frame `soft_prototype_target [4]` directly | Labels are already per-frame in the jsonl. Removes the `group_id = -1` footgun (§4.2) |
| 5 | No masking | `supervision_valid` masks `L_dist`/`L_proto`; `sigma_gt` guarded **before** the KL | 53% of frames carry no physical label; a zeroed `sigma_gt` inside the KL is a `log(x/0)` trap (§7.2) |
| 6 | KL and heads in raw units | Heads predict in normalized label space, **one shared scale per physical dim** | Conditioning, not loss balance — and the shared scale is what preserves the objective (§6.2) |
| 7 | Silent on failed / damaged episodes | 5 episodes excluded from **every** loss, enforced by a converter denylist | All 29 episodes carry labels — masking on `prototype_supervision_valid` alone does **not** exclude them (§4.1) |
| 8 | `D_vlm` ambiguous | `D_vlm = paligemma width`; tensor tapped **before** `out_proj` | `FVLMoE.__call__` returns `out_proj(x)` at action-expert width (`fvlmoe.py:113`) |
| 9 | No train/val split, no leakage handling | Episode-level split; labels rebuilt train-only; input-side leak closed | Two independent leaks — label-side (§4.3.1) and F/T-bias-side (§4.3.2) |
| 10 | Ablations = 3 loss variants | Adds the **oracle-group** arm and an honest limitations section | Needed to know whether the physical branch is more than fruit recognition (§8.3, §9) |

**Locked decisions** (confirmed 2026-07-17):

- **D-A — Force input** = `force_torque` (UR5e flange F/T), de-biased. Tactile is supervision only.
- **D-B — Framework** = JAX / Flax NNX, extending `pi0_force_fvlmoe`.
- **D-C — No fragility labels.** Rule 6 stands. `outlines/task_definition.md` §Methodology (fragility
  prompt, damage threshold, contrastive safety-margin loss) is **superseded** and must be updated to match.
- **D-D — `L_proto` is a core loss**, `lambda_proto = 0.05`, per v1 §8.5.

---

## 1. Goal and the claim we are actually making

Extend ForceVLA with a compact **Physical Interaction Token** `z_phy` that learns safe manipulation
patterns from (1) stage-level safe 6D wrench distributions and (2) soft prototype targets obtained by
clustering those distributions. No manually defined fragility, hardness, stiffness, compliance, damage
score, force limit, or safety upper bound.

Decision D-A sharpens the claim into something testable:

> At inference the robot has **no tactile reading of the fingertips** — only two RGB views, the language
> instruction, the arm state, and the arm's flange F/T. The model must **infer the fingertip contact
> wrench distribution it cannot measure**, and condition its actions on that inference.

That is `task_definition.md`'s force-perception → force-reasoning step as a proposition rather than a
slogan, and it makes the tactile sensor a **data-collection instrument, not a deployment dependency** — a
real deployment advantage worth stating in the paper.

Be precise about what `z_phy` can and cannot encode — read §9 before writing any abstract.

Backbone unchanged: pretrained VLM, projected 6D force token, FVLMoE fusion, flow-matching action head.
The physical branch must affect action generation, not only act as an auxiliary classifier.

---

## 2. Core design

```text
Vision + Language + Current flange wrench
        ↓
FVLMoE contextualized force token   h[:, -1, :]
        ↓
Physical Interaction Token z_phy
        ├── action guidance for the flow action head   (G_phy)
        ├── stage-level safe force distribution        (mu_pred, sigma_pred)  [tactile space]
        └── safe-interaction prototype prediction      (proto_logits)
```

---

## 3. Symbols and shapes

```text
B        batch size
T        action chunk horizon = 8         (action_horizon in the pi0_force_* configs)
N_vl     number of VLM prefix tokens (images + language), = prefix_out.shape[1]
D_vlm    FVLMoE hidden width = paligemma_config.width
D_act    action expert width  = action_expert_config.width
D_phy    physical token dim = 128
K        number of prototypes = 4
N_group  16   (1 task × 4 fruits × 4 stages)
```

```text
state                 [B, 7]        TCP xyz(3) + rotation_vector(3) + gripper_width(1)
force                 [B, 6]        flange F/T, de-biased then z-scored
gt_action             [B, T, 7]     tcp linear vel(3) + angular vel(3) + gripper target(1)

gt_safe_distribution  [B, 12]       tactile two-finger mean; [mu×6, sigma×6]; sigma is std, not var
soft_prototype_target [B, K]
supervision_valid     [B]           bool
group_id              [B]           0..15, or -1; supervision-only, logging only

h                     [B, N_vl+1, D_vlm]   FVLMoE hidden, BEFORE out_proj
fused_force_token     [B, D_vlm]           = h[:, -1, :]
z_phy                 [B, D_phy]
G_fvl, G_phy          [B, T, D_act]

action_pred           [B, T, 7]
mu_pred, sigma_pred   [B, 6]
proto_logits/probs    [B, K]
```

⚠︎ REVIEW-FIX: v1 and v2 used `H_fvl`, `h`, `N`, and `N_vl` for two tensors. There is exactly one
hidden tensor, named `h`, with `N_vl + 1` tokens. `H_fvl` is retired.

Safe-distribution order: `[mu_fx, mu_fy, mu_fz, mu_tx, mu_ty, mu_tz, sigma_fx, …, sigma_tz]`.

---

## 4. The dataset — verified facts

`data/draftVLA_training_20260715/`, 29 episodes, **22 398 frames**, ~7.8 GB.
Task: UR5e pick-and-place, prompt `"grasp the {fruit} from the table and place it into the box"`.
Fruits: potato ×7, orange ×7, pear ×10, banana ×5.

⚠︎ REVIEW-FIX **Sample rate: 9.99 Hz nominal, 8.70 Hz effective** (median across episodes;
`1/median(dt)` = 9.99, `n_frames/duration` = 8.70). The stream is a 10 Hz sensor with ~13% dropped
frames. README's "~8.2 Hz" is the effective figure, roughly. Anything defined in *seconds* must be
converted with care — §5.1 is therefore defined in **frames**.

Images in `rgb/` and `wrist/` are **already cropped to 224×224** — the converter must not re-crop.
Originals are in `rgb_full/` (720×1280) and `wrist_full/` (540×960).

Stage histogram: `prepare 8268 · grasp 2602 · lift 2281 · translate 3106 · place 2635 · reset 3506`.
Physical supervision: 10 624 frames valid, 11 774 masked.

### 4.1 Episodes: the denylist is not optional

**5 episodes are excluded**: `20260714_103759` (banana, damaged during operation) and 4 grasp failures —
`20260712_124735`, `20260712_151030`, `20260712_152829` (potato), `20260714_093626` (pear).

⚠︎ REVIEW-FIX **All 29 episodes carry non-null physical labels**, including all five excluded ones:

```text
pi0_train_20260712_124735  n=534  supervision_valid=300
pi0_train_20260714_093626  n=772  supervision_valid=506
pi0_train_20260714_103759  n=765  supervision_valid=367   # the damaged banana
```

The correct statement is that **24 episodes *contributed to* the label computation**, not that 24 carry
labels. This matters concretely: masking on `prototype_supervision_valid` alone does **not** exclude the
damaged or failed episodes — the converter needs an **explicit episode denylist** (§5). Without it, §7.6
is silently violated and the policy imitates the damage event.

Per-fruit valid episode counts: **banana 4, orange 7, pear 9, potato 4**. Thin, and §9 says so.

### 4.2 The offline label pipeline is DONE and verified — but its artifacts are missing

Per-frame labels already exist in every `observations.jsonl`: `group_id`,
`group_task`/`group_fruit`/`group_stage`, `gt_safe_distribution [12]`, `soft_prototype_target [4]`
(τ_q = 0.1), `prototype_supervision_valid`.

**Verified.** Recomputing `(banana, grasp)` from the tactile two-finger mean over the 24 contributing
episodes reproduces the shipped label bit-for-bit (and the all-29 recompute does **not** — error up to
7.4e+01, independently confirming the exclusion list):

```text
mu    = [-1.314, 0.001, 0.298, 3.993, 8.413, 17.868]
sigma = [ 1.0299, 0.4697, 0.4776, 6.3998, 20.3495, 14.8662]
```

**Because the soft targets are already per-frame, v1 §6's table lookup is unnecessary.** Read
`soft_prototype_target` straight from the row. This deletes rule 9's `group_id = -1` indexing trap from
the training path — `group_id` is needed only for per-group logging.

**Missing from disk** (`DATASET_README_zh.md:30-40,232-239` describe them; they do not exist):
`prototype_metadata/`, `prototype_metadata_tau05/`, `prototype_metadata_tau10/`, and all eight annotation
scripts (`build_safe_group_prototypes.py`, `add_group_prototype_labels.py`, …).

Consequence: **the K/τ ablations in v1 §6.4 are unrunnable**, and so is §4.3.1's leakage fix. Recovering
or rewriting `build_safe_group_prototypes.py` is a **prerequisite**. Risk is low — the recompute above
proves the recipe is fully specified: group = (task, fruit, stage) over `{grasp, lift, translate, place}`;
signal = per-dim mean of `tactile_estimated_wrenches.left_estimated` and `.right_estimated`; population
std (ddof=0).

Descriptor / K-means recipe, unchanged from v1 §6:
`q_g = concat(norm(mu_g), norm(log(clamp(sigma_g, 1e-4))))`, robust per-dim `(x - median)/(IQR + 1e-6)`,
K-means K=4 seed=0, `Y = softmax(-mean_sq_dist / tau_q)`.

Clusters (τ=0.1 targets are effectively one-hot):

| Prototype | Member groups | Reading |
|---|---|---|
| 0 | banana ×4 stages, orange ×4 stages | light stable contact |
| 2 | pear ×4 stages, potato/grasp | hard-object squeeze |
| 3 | potato/lift, potato/translate | heavy load in flight |
| 1 | potato/place | heavy load unloading |

Only `potato/grasp` is meaningfully soft at τ=0.1 (`[0.077, 0, 0.923, 0]`).

⚠︎ **This table describes the SHIPPED (24-episode) labels only — the train-only rebuild re-partitions
it.** Rebuilding from the 20 training episodes of §8.1 yields a different, stable partition: potato
collapses into a single prototype and pear splits by phase (`{pear/grasp, pear/lift}` vs
`{pear/translate, pear/place}`), against the shipped `{pear ×4 + potato/grasp}` / `{potato/lift,
translate}` / `{potato/place}`. Both are verified global optima (stable across n_init 5→50), so this
is not a seed artifact — holding out one episode per fruit genuinely changes the cluster structure.

Two consequences. (1) Any prose quoting the cluster membership above must be regenerated against the
labels the model actually trains on; the shipped table is a description of the data, not of the
experiment. (2) This is §9.4's scale limitation biting concretely, and it is itself a result worth
reporting: prototypes that move when one episode per fruit is held out are not a stable structure
that would survive new fruit instances. Check the rebuilt partition before writing any claim about
what the prototypes "mean".

### 4.3 Two independent leaks

#### 4.3.1 Label-side

The shipped `gt_safe_distribution`, K-means centers, and normalizer stats were computed over **all 24**
contributing episodes. Any episode held out for validation contributed to the group statistics it will be
scored against, and to the cluster centers. `DATASET_README_zh.md` §6.2 already requires
"training-set-only statistics"; the shipped tables do not honor that across a split.

**Required:** rebuild labels from the **20 training episodes only** (§8.1), written to a separate label
set rather than overwriting the shipped `observations.jsonl`.

#### 4.3.2 ⚠︎ REVIEW-FIX Input-side: raw flange F/T leaks fruit identity outright

The F/T sensor drifts across a recording session, and fruits were recorded in contiguous time blocks. The
per-episode **rest `fx`** (median over the first 10 `prepare` frames, free space, no contact) separates
fruit **perfectly within each session**:

```text
session 20260712   orange: -8.53 -8.50 -8.48 -8.43 -8.09 -6.81 -6.64
                   potato: -5.79 -5.39 -1.55 -1.34            → threshold -6.2 separates perfectly
session 20260714   banana: -11.30 -9.52 -9.37 -8.64
                   pear:    -7.70 -7.36 -7.00 -5.93 -4.81 -4.12 -3.31 -1.51 -0.66
                                                              → threshold -8.00 separates perfectly
```

A model reading raw `force_torque` can identify the fruit from the sensor's DC offset alone, without
looking at the image. Every claim in §1 and §9.1 about *inferring* contact from *vision* would be
unfalsifiable. **This is why §5.1's de-bias is mandatory** — subtracting the episode's own rest baseline
zeroes exactly this offset. It also reframes the `raw-force` ablation (§8.3): that arm is a **leak probe**,
not a cost/benefit question.

Residual pose-dependent gravity survives de-biasing, but it is a function of arm pose, not of fruit, so it
does not leak identity.

### 4.4 README errors to not propagate

1. **README §9.2 claims tactile `fz` is identically zero. It is not** — 8 974 of 18 870 frames (counted
   over the 24 contributing episodes) have `fz ≠ 0`. It is *zero-inflated* (~52% exact zeros), which
   makes a Gaussian on that dim misspecified but does not make it a dead feature.
2. **`force_torque` is not de-biased** (README §9.1 is right): gripper weight + sensor bias, `fx ≈ -12 N`
   at rest. See §5.1 and §4.3.2.
3. **Rate is 10 Hz nominal / 8.7 Hz effective**, not 8.2 Hz (§4).

---

## 5. Data pipeline (openpi)

Converter: extend `examples/force/convert_force_data_to_lerobot.py` → new repo_id `draftvla/fruits`.
It takes an explicit **episode denylist** (§4.1); denylisted episodes are not written at all.

| Key | Source | Shape |
|---|---|---|
| `observation/image` | `rgb/*.jpg` (already 224²) | (224,224,3) |
| `observation/wrist_image` | `wrist/*.jpg` (already 224²) | (224,224,3) |
| `observation/state` | `tcp_pose.position_xyz` + `tcp_pose.rotation_vector` + `gripper_width` | (7,) |
| `observation/force_torque` | `force_torque`, de-biased per §5.1 | (6,) |
| `actions` | `action_7` | (7,) |
| `prompt` | `prompt` | str |
| `gt_safe_distribution` | `gt_safe_distribution`; `mu := 0`, **`sigma := 1`** when null | (12,) |
| `soft_prototype_target` | `soft_prototype_target`, uniform `1/K` when null | (4,) |
| `supervision_valid` | `prototype_supervision_valid` | () bool |
| `group_id` | `group_id` (**logging only**) | () int32 |

⚠︎ REVIEW-FIX Null `sigma` becomes **1, not 0**. A zeroed `sigma_gt` makes the KL's `log(σ_pred/σ_gt)`
term `log(x/0) = +inf`, and `0 * inf = NaN` — masking *after* the reduction cannot rescue it, so the whole
`L_total` is NaN at step 0. The dummy must be a valid distribution. Never NaN, and **never a zero sigma**.
§7.2 adds a second, independent guard; both ship.

### 5.1 De-biasing the flange F/T — mandatory

Baseline `b_e` = per-episode **median of `force_torque` over the first 10 `prepare` frames** (~1 s at
10 Hz nominal; defined in frames because the effective rate varies, §4). Require ≥5 such frames, else
fall back to the whole `prepare` segment; assert at conversion. Emit `force_torque := raw - b_e`.

**Rationale** (⚠︎ REVIEW-FIX — v2 argued this backwards): de-biasing removes the **constant per-episode
sensor offset**, which is precisely the fruit-identity leak of §4.3.2. It does *not* remove the
pose-dependent gravity component — a single constant cannot — and it does not need to: that component
tracks arm orientation, not fruit, so the model is welcome to learn it. v2's claim that "z-scoring won't
remove a pose-dependent bias, therefore de-bias" was a non-sequitur; the real argument is the leak.

**Deployment contract:** at inference the robot records ≥10 free-space F/T frames at episode start and
subtracts the same baseline before sending `observation/force_torque`. Document in
`examples/force/README.md`; enforce in the client example.

Normalization: `force` and `state` keep openpi's z-score path via `compute_norm_stats.py` (the `"force"`
stats key is already gated on `model.force_aware`).

### 5.2 Routing supervision to the loss — **Decision D1, open**

`data_loader.py:540` yields exactly `(Observation.from_dict(batch), batch["actions"])`; other keys are
dropped. The four supervision arrays need a route.

- **Option A (recommended).** Yield `(observation, actions, aux)` with `aux: dict[str, Array]` = `{}` for
  every existing config. Touches `data_loader.py:540`, the train step in `scripts/train.py`, and
  `BaseModel.compute_loss` (`model.py:281`) gains `aux: dict | None = None`, so `pi0_fast` and every other
  model are untouched. Keeps `Observation` = model inputs only, which is what makes rule 9 mechanically
  enforceable.
- **Option B.** Add optional trailing fields to `Observation`, as `force` was added. Smaller diff, but it
  puts `group_id` inside the model's input PyTree — what rule 9 forbids — and the guarantee decays to a
  code-review promise.

Resolve before implementation starts.

---

## 6. Model (JAX / Flax NNX)

### 6.1 Changes to existing files

**`src/openpi/models/fvlmoe.py`** — `__call__` returns `out_proj(x)` at `d_out = D_act` (`fvlmoe.py:113`).
Add `return_hidden: bool = False` to also return `h [B, N_vl+1, D_vlm]` **before** `out_proj`.
`fused_force_token = h[:, -1, :]` — the last **token** (the appended force token, which has attended to
every prefix position), not the last feature dimension. `D_vlm = paligemma_config.width` (`pi0.py:114`).

**`src/openpi/models/pi0.py`** — ⚠︎ REVIEW-FIX v2's line attributions were wrong and named only one of the
**two** injection sites. Correct map:

| Site | Line | What |
|---|---|---|
| `embed_suffix` | `160` | M1 token path only (`180–185`). **No M2 code here** — v2 wrongly placed the M2 branch in this function |
| `compute_loss` | `217` | M2 fusion at `245`; **training injection at `250`**: `action_hidden = action_hidden + fused[:, -self.action_horizon:, :]` |
| `sample_actions` | `256` | M2 fusion at `285`; inference injection at `319` |

`G_phy` must be added at **both** `250` and `319`. Following v2 literally would have added it only at
inference — **the physical branch would never have trained.** The correct statement is
`action_hidden += G_fvl + G_phy` at each site. (v1 §7.3's `G_total = G_fvl + G_phy + S_suffix` does not
map onto this code at all: openpi has no separate `S_suffix` addend.)

As in the current M2 path, `G_phy` depends only on the prefix and the force reading, not on the denoising
step — compute it once outside the loop in `sample_actions`.

**`src/openpi/models/pi0_config.py`** — new flags, all defaulting off: `phy_enabled: bool = False`,
`phy_dim: int = 128`, `phy_num_prototypes: int = 4`, `lambda_dist: float = 1.0`,
`lambda_proto: float = 0.05`, `lambda_nll: float = 0.0`, `use_force_nll: bool = False`, plus the label
normalizer stats. `get_freeze_filter()` needs no change — the new modules sit outside the VLM.

**`src/openpi/training/weight_loaders.py`** — extend `extra_missing_regex` to cover the new `phy_*`
parameters, exactly as `force_proj` / `fvlmoe` are handled today.

### 6.2 New file `src/openpi/models/physical.py`

```text
PhysicalProjector:          Linear(D_vlm→512) → GELU → LayerNorm(512) → Linear(512→D_phy) → L2 normalize
PhysicalActionProjector:    Linear(D_phy→512) → GELU → Linear(512→T*D_act) → reshape(B,T,D_act)
SafeForceDistributionHead:  Linear(D_phy→256) → GELU → Linear(256→12)
PrototypeClassifier:        Linear(D_phy→D_phy) → GELU → Linear(D_phy→K)
```

**Why normalize — and the two ways to get it wrong.**

An earlier draft held that the raw-unit KL is dominated by torque (label scales run from
`sigma_fy = 0.2864` to `sigma_tx ≈ 127`). **That is wrong.** Gaussian KL is invariant under a per-dim
affine change of variable: scaling `mu` and `sigma` by the same `s_d` leaves `log(σ_q/σ_p)` and
`(σ_p² + Δμ²)/(2σ_q²)` unchanged. The loss is already scale-free.

⚠︎ REVIEW-FIX **But v2's own normalizer broke that invariance.** v2 defined `(m, s)` as the per-dim
mean/std of the **12-vector**, giving `mu_d` and `sigma_d` *different* scales (`s_mu(ty) = 13.49` vs
`s_sigma(ty) = 33.07`, a 2.45× gap) — which silently changes the objective:

```text
raw KL            = 0.3472
v2's normalizer   = 0.5576     # a different loss
one shared s_d    = 0.3472     # what invariance actually requires
```

**Correct normalizer.** One scale per *physical* dim `d ∈ {fx…tz}`, shared by `mu_d` and `sigma_d`, from
the training split only. This is a change of variable on the wrench itself, `x'_d = (x_d - m_d)/s_d`, so
`mu'_d = (mu_d - m_d)/s_d` and `sigma'_d = sigma_d/s_d` — shift applies to `mu` only, and the KL is
provably unchanged:

```text
m_d = mean over training groups of mu_gt[d]
s_d = median over training groups of sigma_gt[d]      # the label's own physical scale for that dim

head predicts     mu_norm [B,6], raw_sigma [B,6]
sigma_norm      = softplus(raw_sigma) + 1e-6
L_dist          = KL in normalized space               # identical to raw space, better conditioned
mu_pred         = m + s * mu_norm                      # de-normalize for logging / serving only
sigma_pred      = s * sigma_norm
```

**The benefit is conditioning, and the honest version of that argument is:** `∂KL/∂mu_pred =
(mu_pred − mu_gt)/σ_pred²`, so the natural per-dim preconditioner is the label `σ_d` — hence `s_d = median
label sigma`, not the across-group std of `mu_d`. With `s_d` chosen this way, `sigma_norm ≈ 1` and
`mu_norm` lands within roughly ±1.5 for every dim, so a shared learning rate on the head's last layer
means the same thing in every column. Note this repo trains with Adam (`config.py:537`), whose update is
approximately scale-free, so the raw-space gradient disparity is *partly* normalized away already;
normalization here buys conditioning of the head's output space, not a loss reweighting. Do not oversell it.

The `1e-4` clamps from v1 §8.2 never bind on this data (smallest label sigma = **0.2864**, `sigma_fy`,
banana/translate) — keep them as guards.

### 6.3 Output dictionary

```python
{
    "action_pred":  [B, T, 7],
    "mu_pred":      [B, 6],    # de-normalized, raw tactile units
    "sigma_pred":   [B, 6],    # de-normalized, raw tactile units
    "z_phy":        [B, D_phy],
    "proto_logits": [B, K],
    "proto_probs":  [B, K],
}
```

⚠︎ REVIEW-FIX v1/v2 emitted `safe_force_distribution [B,12]` *and* `mu_pred`/`sigma_pred` — the same
numbers twice, two places to drift. The `[12]` form is now **derived at the serving boundary only**
(`concat(mu_pred, sigma_pred)`), never carried in the model output. `safe_force_distribution_chunk
[B, T, 12]` is a broadcast for API convenience; it must never multiply the distribution loss by `T`.

---

## 7. Losses

### 7.1 `L_flow`

Unchanged pi0 flow-matching objective over all `[B, T, 7]` action tokens. Computed on every frame of a
training episode, including `prepare` and `reset` — the policy must act there too.

### 7.2 `L_dist` — masked, with the sigma guard

```text
valid      = supervision_valid                                  # [B]
sigma_safe = jnp.where(valid[:, None], sigma_gt_n, 1.0)         # guard BEFORE the KL
mu_safe    = jnp.where(valid[:, None], mu_gt_n, 0.0)
kl_b       = diagonal_gaussian_kl(mu_safe, sigma_safe, mu_norm, sigma_norm)   # [B], mean over 6 dims
L_dist     = sum(valid * kl_b) / max(sum(valid), 1)
```

⚠︎ REVIEW-FIX The `where` is not redundant with §5's dummy — it is the guard that makes `L_dist`
correct **regardless** of what the converter wrote. Reducing first and masking second yields
`0 * inf = NaN` whenever a `sigma_gt` reaches the KL as 0. Note the gradient in that scenario is
coincidentally 0, so a test that checks only "finite gradients" **half-passes while the loss is NaN** —
§10.5 therefore asserts on the loss value too.

`KL(p_gt ‖ p_pred)` — forward, mode-covering, equivalent to moment matching. Right direction; keep it.
~53% of frames are masked, so the effective batch for this loss is about half of `B` and varies per
batch; the `max(·, 1)` guard matters.

### 7.3 `L_proto` — masked

```text
y_target = soft_prototype_target                  # [B, K], per-frame, no table lookup
L_proto  = sum(valid * -(y_target * log_softmax(proto_logits)).sum(-1)) / max(sum(valid), 1)
```

Always log **`L_proto − H(y_target)`** alongside `L_proto`. The soft cross-entropy is lower-bounded by
the target's own entropy, which grows with τ — comparing raw `L_proto` across a τ ablation compares
nothing (`DATASET_README_zh.md` §9.7).

### 7.4 `L_nll` — optional, off, and largely redundant

Disabled for every baseline (`lambda_nll = 0`). For the record: minimizing `KL(p_gt ‖ p_pred)` *is*
expected NLL under `p_gt` up to a constant, so `L_nll` is the same objective with empirical chunk samples
in place of group moments. It also needs new plumbing — future force over the horizon `[B,T,6]` is not
currently a LeRobot feature. Defer.

### 7.5 Total and schedule

```text
L_total = L_flow + lambda_dist * L_dist + lambda_proto * L_proto
lambda_dist = 1.0    lambda_proto = 0.05    lambda_nll = 0.0

Phase 1: L_flow + lambda_dist * L_dist
Phase 2: linearly ramp lambda_proto 0 → 0.05
Phase 3: joint fine-tune
```

### 7.6 Training episodes

Train on the **20 training episodes of the 24 contributing ones**. The 4 grasp failures teach the policy
to fail; the damaged banana (`20260714_103759`) teaches it to damage. Excluded from every loss **via the
converter denylist** (§4.1/§5) — masking alone will not do it, since these episodes carry valid labels.
The damaged episode is retained out-of-band for diagnostics: it is the dataset's only recorded damage
event (§9). Ablation `train-on-failures` quantifies the choice.

---

## 8. Evaluation

### 8.1 Split

Episode-level, stratified by fruit — **4 validation episodes, 20 training**:

| Fruit | Contributing eps | Val episode |
|---|---|---|
| banana | 4 | `pi0_train_20260714_103538` |
| orange | 7 | `pi0_train_20260712_162950` |
| pear | 9 | `pi0_train_20260714_095938` |
| potato | 4 | `pi0_train_20260712_153929` |

Labels **rebuilt from the 20 training episodes only** (§4.3.1). This leaves banana and potato groups with
**3 episodes each** to estimate a 6-D mean and std. Those sigmas are noise-dominated; §9 owns it.

Held-out-*fruit* generalization is **not testable** with 4 fruits — removing any fruit deletes at least
one prototype (banana/orange share cluster 0; potato alone spans clusters 1 and 3). Do not claim it.

### 8.2 Offline metrics (what this data can actually measure)

- Val `L_flow`; action MSE per dim, split by stage.
- Val `L_dist`; per-dim `mu`/`sigma` MAE in raw tactile units; per-group breakdown (16 rows).
- Prototype top-1 accuracy and `L_proto − H(y_target)`.
- `z_phy` structure: t-SNE / linear probe for fruit and stage — tests §9.1 directly.

### 8.3 Ablations

| Arm | Config | Question |
|---|---|---|
| `no-force` | `pi0_force_baseline` | does force help at all? |
| `forcevla` | `pi0_force_fvlmoe` (today's) | does the physical branch add to plain ForceVLA? |
| `+dist` | `+ L_dist` | v1 §14 |
| `full` | `+ L_dist + L_proto` | v1 §14 |
| `no-guidance` | full, `G_phy` detached | ⚠︎ see caveat below |
| **`oracle-group`** | `z_phy` := embedding of the GT 16-way one-hot | **upper bound. If `full ≈ oracle`, the branch is fruit+stage recognition and nothing more** |
| `tactile-input` | force token := tactile two-finger mean | upper bound on D-A: what does the flange F/T cost us? |
| `raw-force` | **leak probe** (§4.3.2): no de-bias | if this *wins*, it is reading fruit ID off the sensor's DC offset, not doing force reasoning. Expect it to win. Report as a leak measurement, never as a method result |
| `train-on-failures` | include the 5 denylisted episodes | §7.6 |
| K/τ sweep | `K ∈ {3,4,5,6}`, `τ_q ∈ {0.05, 0.1, 0.2}` | v1 §6.4 — **blocked on §4.2 rebuild** |

⚠︎ REVIEW-FIX **The `no-guidance` arm is weaker than v1/v2 advertised.** `z_phy` derives from
`h[:, -1, :]`, the force token — which is *already inside* the `fused[:, -action_horizon:, :]` slice that
forms `G_fvl` (`pi0.py:250`, `:285`). Detaching `G_phy` does not remove that token's influence on the
action head, so the arm does **not** isolate "gradient shaping only". It measures only the *additional*
`z_phy → action` pathway on top of an already force-conditioned baseline. State it that way, or build a
genuine variant that also drops the force token from the `G_fvl` slice.

`oracle-group` and `tactile-input` are the two arms that make this a paper rather than a leaderboard entry.

### 8.4 Online metrics — the headline numbers are not offline-derivable

`task_definition.md` asks for success rate, damage rate, and SDR = success/damage. **The offline dataset
contains exactly one damaged episode.** Damage rate and SDR are **not measurable offline at all**, and no
val loss substitutes. They require real-robot trials. Before any such run, fix:

- a written damage criterion (e.g. bruise area from a fixed post-trial photo, ≥2 blind raters,
  inter-rater agreement reported);
- ≥10 trials per fruit per arm, on **fruit instances not in the training set**;
- arms: `no-force`, `forcevla`, `full` at minimum;
- the SDR plot from `task_definition.md`: damage on x, success rate on y.

Plan this now — it gates the contribution, and 4 fruits × 3 arms × 10 trials is ~120 physical trials plus
fruit procurement.

---

## 9. Limitations to state, not hide

1. **The safe-distribution head is a 16-way lookup keyed by recognized (fruit, stage).**
   `gt_safe_distribution` is constant within a group, so the Bayes-optimal predictor given (fruit, stage)
   *is* the group label. `L_dist ≈ 0` proves only that the model recognizes fruit and phase from pixels.
   That is not nothing — inferring an unmeasured fingertip wrench distribution from vision + flange F/T is
   a real inference — but it is not instance-level force reasoning, and the ripeness/size variation that
   `task_definition.md` §任务难点 calls the core difficulty is **not captured by any label in this
   dataset**. The `oracle-group` ablation measures exactly this gap. Report it.
   (This claim is only meaningful once §4.3.2's leak is closed — with raw force, the fruit half is
   readable off the sensor bias and even `oracle-group` would not bound it.)

2. ⚠︎ REVIEW-FIX **The label sigmas mean different things per stage — v2 over-generalized from
   `translate`.** Within-episode share of total variance, recomputed across all 4 fruits × 6 dims:

   | Stage | min | median | max | Reading |
   |---|---|---|---|---|
   | `grasp` | 0.262 | **0.749** | 0.945 | genuine within-interaction variation |
   | `place` | 0.135 | **0.559** | 0.807 | genuine, more spread |
   | `lift` | 0.001 | **0.045** | 0.561 | mixed (banana 0.03–0.54, pear fz 0.56; orange/potato ≈0) |
   | `translate` | 0.000 | **0.002** | 0.038 | almost entirely between-episode offset |

   So the honest statement — and it is **better news than v2 claimed** — is that during steady transport
   (`translate`, and mostly `lift`) the fingertip wrench is near-constant within an episode and the group
   sigma is dominated by episode-to-episode offsets (grasp pose, fruit placement, calibration drift). But
   at `grasp` and `place` — **the two damage-relevant contact events** — the majority of the variance is
   genuinely within-episode, so sigma there is a real interaction band. v2's "for every translate group
   0.00–0.03 / most lift groups under 0.1 / grasp and place 0.25–0.95" quoted single favorable dims and
   was wrong on three counts (potato/translate fx = 0.038; banana/lift fx = 0.54; orange/place fy = 0.135).
   Consider weighting `L_dist` toward `grasp`/`place`, or reporting per-stage `L_dist` separately.

3. **`L_proto` is close to redundant with `L_dist`** (D-D keeps it as a core loss anyway). τ=0.1 targets
   are one-hot except `potato/grasp`, and clusters track fruit identity. The prototype label is a
   deterministic coarsening of the group that `L_dist` already supervises at full resolution. If `+dist`
   and `full` are within noise, say so plainly rather than tuning `lambda_proto` until a gap appears. The
   K/τ sweep is the honest way to give `L_proto` a chance to matter — another reason §4.2's rebuild is a
   prerequisite.

4. **Scale.** 24 usable episodes, 4 fruits, 3–9 episodes per group, one damage event. Banana and potato
   group statistics rest on 3 training episodes.

5. **Tactile `fz` is zero-inflated** (~52% exact zeros, §4.4) — a Gaussian on that dim is misspecified.
   Kept for fidelity to the shipped labels; flag it if `fz` shows anomalous per-dim MAE.

---

## 10. Definition of done

Implementation is complete when **all** of the following hold:

1. `ruff check . && ruff format --check .` clean; `uv run pytest --strict-markers -m "not manual"` green.
2. **No regression:** with `force_aware=False`, outputs bit-identical to vanilla π₀; with
   `phy_enabled=False`, bit-identical to today's `pi0_force_fvlmoe`. Asserted by test, not by eye.
3. **Rule 9 enforced by a test:** `group_id` is provably absent from every model input PyTree. Option A
   (§5.2) makes this structural.
4. Shape tests for every §3 tensor; `fvlmoe(..., return_hidden=True)` gives `h [B, N_vl+1, D_vlm]`.
5. **Mask test:** an all-`False` `supervision_valid` batch yields `L_dist == L_proto == 0` **and finite
   `L_total`** — assert on the loss *value*, not only on gradients (§7.2's half-passing trap).
6. **Both-injection-sites test:** with `phy_enabled=True`, `G_phy` has nonzero gradient under
   `compute_loss` (`pi0.py:250`), not only under `sample_actions` (`pi0.py:319`). This is the test that
   catches v2's bug.
7. **KL-invariance test:** normalized-space `L_dist` equals raw-space `L_dist` to float tolerance — the
   property v2's normalizer silently broke (§6.2).
8. **Denylist test:** no frame from the 5 excluded episodes reaches any loss (§4.1).
9. **De-bias test:** every converted episode's rest baseline is ≈0 after de-biasing, and the §4.3.2
   rest-`fx` fruit separation is destroyed.
10. **Label-parity test:** the rebuilt `build_safe_group_prototypes.py` reproduces the shipped
    `gt_safe_distribution` bit-for-bit over all 24 contributing episodes — §4.2's hand check, automated as
    the regression gate on the rebuild.
11. `compute_norm_stats.py` emits `force` stats for the new config; smoke train (200 steps, 1 episode)
    drops the loss and holds VLM `param_norm` constant, per `examples/force/run_smoke_repro.sh`.
12. End-to-end serving returns an `(8, 7)` action chunk **and** a `[12]` safe distribution, verified
    against a live `serve_policy.py`.
13. `examples/force/README.md` documents the F/T de-bias deployment contract (§5.1).
14. `outlines/task_definition.md` updated to match decision D-C.

**Review gate.** Before the full training run, a second model reviews this spec and the diff against it
(per `CLAUDE.md` Policy), specifically checking: the mask + sigma guard, the shared-scale KL, both
injection sites, `group_id` isolation, the denylist, and that §9's limitations are reflected in whatever
text claims results.

---

## 11. Implementation rules (revised from v1 §12)

1. `fused_force_token = h[:, -1, :]` where `h` is the FVLMoE hidden **before** `out_proj`; the last
   **token**, not the last feature dimension. `D_vlm = paligemma_config.width`.
2. Do not use cross-attention to derive `z_phy`.
3. `z_phy` influences actions through additive guidance — `action_hidden += G_fvl + G_phy` — at **both**
   `pi0.py:250` (training) and `pi0.py:319` (inference). Missing the first silently disables the branch.
4. The safe-distribution head predicts stage-level `[B, 12]`, never a true temporal `[B, T, 12]`.
5. Never duplicate a stage label over `T` and sum `T` repeated KL losses.
6. No force limits, upper-bound losses, damage thresholds, fragility labels, or hand-designed material
   labels. (Decision D-C — this rule beats `task_definition.md`.)
7. Prototype descriptors use only `[normalize(mu), normalize(log(sigma))]`.
8. K-means centers and soft target tables are fixed offline metadata, never trainable parameters.
9. `group_id` is supervision-only and must never reach the model. Enforced by §10.3.
10. Force and torque dims are normalized separately — in the **descriptor** (§4.2) and in the **head's
    output space** (§6.2), with **one shared scale per physical dim across `mu` and `sigma`**. Not as a KL
    reweighting; the KL is already scale-invariant, and an unshared scale silently changes the objective.
11. Build safe-distribution labels from successful, non-damaging demonstrations only — and from the
    **training split only** (§4.3.1). Exclusion is enforced by a converter denylist, not by label masking.
12. Start with `K=4`, `tau_q=0.10`, `D_phy=128`, `lambda_proto=0.05`, `T=8`.
13. Mask `L_dist`/`L_proto` with `supervision_valid`, and guard `sigma_gt` with a `where` **before** the
    KL. Null labels are never NaN and **`sigma` is never 0** — a zero sigma is the NaN.
14. De-bias `force_torque` per episode (§5.1). This is a leakage fix (§4.3.2), not a preprocessing
    preference.
15. Log predicted mu/sigma, prototype probs, `L_proto − H(y_target)`, and per-group **and per-stage** losses.
16. Every ablation in §8.3 ships, including `oracle-group` — it is the arm that tests §9.1.
