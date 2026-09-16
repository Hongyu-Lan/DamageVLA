# Damage-Aware ForceVLA with Stage-Level Safe Interaction Prototypes
## Implementation Specification for Codex

## 1. Goal

Extend ForceVLA with a compact **Physical Interaction Token** that learns safe manipulation patterns from:

1. stage-level safe 6D wrench distributions;
2. soft prototype targets obtained by clustering those safe distributions.

Do **not** use manually defined fragility, hardness, stiffness, compliance, damage scores, force limits, or safety upper-bound labels.

Keep the original ForceVLA backbone:

- pretrained VLM;
- projected 6D force token;
- FVLMoE multimodal fusion;
- flow-matching action head.

The physical branch must directly affect action generation, not function only as an auxiliary classifier.

---

## 2. Core Design

```text
Vision + Language + Current Wrench
        ↓
FVLMoE contextualized force token
        ↓
Physical Interaction Token z_phy
        ├── action guidance for the flow action head
        ├── stage-level safe force distribution prediction
        └── safe-interaction prototype prediction
```

The physical token should implicitly encode how delicately the current object and interaction should be handled, without an explicit fragility label.

---

## 3. Symbols and Tensor Shapes

```text
B       batch size
T       action chunk horizon
D_vlm   FVLMoE hidden dimension
D_act   action expert hidden dimension
D_phy   physical interaction token dimension, recommended 128
K       number of fixed safe-interaction prototypes, default 4
N_group number of (task, fruit, stage) groups
```

Example dataset:

```text
1 tasks × 4 fruits × 4 stages = N_group = 16
```

Main tensors:

```text
state                 [B, 7]
current_force         [B, 6]
gt_action             [B, T, 7]
gt_safe_distribution  [B, 12]
group_id              [B]

H_fvl                 [B, N_vl + 1, D_vlm]
fused_force_token     [B, D_vlm]
z_phy                 [B, D_phy]

action_pred           [B, T, 7]
safe_dist_pred        [B, 12]
mu_pred               [B, 6]
sigma_pred            [B, 6]

proto_logits          [B, K]
proto_probs           [B, K]
soft_proto_target     [B, K]
```

The safe distribution order is:

```text
[mu_fx, mu_fy, mu_fz, mu_tx, mu_ty, mu_tz,
 sigma_fx, sigma_fy, sigma_fz, sigma_tx, sigma_ty, sigma_tz]
```

---

## 4. Training Inputs and Labels

### Model inputs

Each policy window provides:

```text
RGB(base)
RGB(wrist)
Language instruction
Current State [7]
Current Force [6]
```

### Supervision metadata

Each policy window provides:

```text
gt_action               [T, 7]
gt_safe_distribution    [12]
group_id                scalar integer in [0, N_group - 1]
```

Optional diagnostic data:

```text
gt_force                [T, 6] or [6]
```

`group_id` is only for retrieving fixed prototype supervision. Never feed it to the VLM, FVLMoE, policy, or any prediction head.

---

## 5. Offline Stage-Level Safe Distribution Labels

Define:

```text
group = (task, fruit, stage)
```

Examples:

```text
(pick_place, banana, lift)
(pick_place, banana, translate)
(pick_place, banana, place)
```

For each group, use only successful and non-damaging demonstrations. Collect all 6D wrench readings within the stage:

```text
wrench = [Fx, Fy, Fz, Tx, Ty, Tz]
```

Compute per-dimension statistics:

```text
mu_g       [6]
sigma_g    [6]
```

Construct:

```text
safe_distribution_g = concat(mu_g, sigma_g)    # [12]
```

Every policy window from that group receives the same `[12]` label.

Do not repeat this stage label over T and sum duplicate losses.

---

## 6. Offline Safe-Interaction Prototype Construction

### 6.1 Descriptor construction

For each group:

```text
q_g = concat(
    normalize(mu_g),                       # [6]
    normalize(log(clamp(sigma_g, 1e-4)))  # [6]
)                                          # [12]
```

Stack:

```text
Q = [q_1, ..., q_N_group]                 # [N_group, 12]
```

For the example dataset:

```text
Q.shape = [36, 12]
```

### 6.2 Normalization

Force and torque have different units. Normalize each descriptor dimension independently, with training-set-only statistics.

Recommended robust normalization:

```text
x_norm = (x - median(x)) / (IQR(x) + 1e-6)
```

Constants:

```text
sigma_min = 1e-4
epsilon = 1e-6
```

### 6.3 K-means

Run K-means once, offline, over Q:

```text
K = 4
centers C: [K, 12] = [4, 12]
```

Centers are fixed descriptor-space metadata, not neural parameters.

### 6.4 Soft prototype target table

```python
delta = Q[:, None, :] - C[None, :, :]     # [N_group, K, 12]
dist_sq = (delta ** 2).mean(dim=-1)        # [N_group, K]

tau_q = 0.10
Y_target_all = softmax(-dist_sq / tau_q, dim=-1)
# [N_group, K]
```

For the example:

```text
Y_target_all.shape = [36, 4]
```

Search in ablations:

```text
K in {3, 4, 5, 6}
tau_q in {0.05, 0.10, 0.20}
```

Save:

```text
safe_distribution_all.pt        [N_group, 12]
prototype_centers.pt            [K, 12]
soft_prototype_targets_all.pt   [N_group, K]
normalizer_stats.pt
group_metadata.json
```

---

## 7. Online Model Architecture

### 7.1 Base ForceVLA encoding

```text
Dual-view RGB + Language
        ↓
pretrained VLM
        ↓
VL tokens

Current 6D wrench
        ↓
force projection
        ↓
force token

VL tokens + force token
        ↓
FVLMoE
        ↓
H_fvl [B, N_vl + 1, D_vlm]
```

Use the final FVLMoE token as the contextualized force token:

```python
fused_force_token = H_fvl[:, -1, :]   # [B, D_vlm]
```

This is the last token, not the last feature dimension.

### 7.2 Physical Interaction Token

```python
z_phy = PhysicalProjector(fused_force_token)   # [B, D_phy]
```

Recommended projector:

```text
Linear(D_vlm -> 512)
GELU
LayerNorm(512)
Linear(512 -> D_phy)
L2 Normalize
```

Recommended:

```text
D_phy = 128
```

The same token is used by all three branches below.

### 7.3 Physical Action Guidance Branch

```python
G_phy = PhysicalActionProjector(z_phy)   # [B, T, D_act]
```

Recommended:

```text
Linear(D_phy -> 512)
GELU
Linear(512 -> T * D_act)
reshape(B, T, D_act)
```

Let:

```text
G_fvl      original ForceVLA guidance from FVLMoE, [B, T, D_act]
S_suffix   original state + noisy-action feature, [B, T, D_act]
```

Use:

```python
G_total = G_fvl + G_phy + S_suffix
```

Feed `G_total` to the unchanged flow action expert:

```text
action_pred: [B, T, 7]
```

### 7.4 Stage-Level Safe Force Distribution Head

Predict only one stage-level distribution:

```python
safe_dist_raw = SafeForceDistributionHead(z_phy)   # [B, 12]
mu_pred = safe_dist_raw[:, :6]                     # [B, 6]
raw_sigma = safe_dist_raw[:, 6:]                   # [B, 6]
sigma_pred = softplus(raw_sigma) + 1e-6             # [B, 6]
```

Recommended:

```text
Linear(D_phy -> 256)
GELU
Linear(256 -> 12)
```

For an external chunk-aligned API only:

```python
safe_dist_chunk = safe_dist_raw[:, None, :].expand(-1, T, -1)
# [B, T, 12]
```

Do not use the broadcast form to multiply the distribution loss by T.

### 7.5 Prototype Classifier

Do not use learnable latent prototype vectors.

```python
proto_logits = PrototypeClassifier(z_phy)  # [B, K]
proto_probs = softmax(proto_logits, dim=-1)
```

Recommended:

```text
Linear(D_phy -> D_phy)
GELU
Linear(D_phy -> K)
```

---

## 8. Loss Functions

### 8.1 Flow action loss

Keep the original ForceVLA / pi0 flow-matching objective unchanged:

```text
L_flow
```

It supervises all `[B, T, 7]` action tokens, not only the first action.

### 8.2 Safe distribution KL loss

Target:

```text
p_gt = Normal(mu_gt, diag(sigma_gt^2))
```

Prediction:

```text
p_pred = Normal(mu_pred, diag(sigma_pred^2))
```

Use diagonal Gaussian KL divergence:

```python
def diagonal_gaussian_kl(mu_gt, sigma_gt, mu_pred, sigma_pred):
    sigma_gt = sigma_gt.clamp_min(1e-4)
    sigma_pred = sigma_pred.clamp_min(1e-4)

    kl = (
        torch.log(sigma_pred / sigma_gt)
        + (sigma_gt.pow(2) + (mu_gt - mu_pred).pow(2))
          / (2.0 * sigma_pred.pow(2))
        - 0.5
    )
    return kl.mean()
```

Shapes:

```text
mu_gt, sigma_gt:        [B, 6]
mu_pred, sigma_pred:    [B, 6]
```

### 8.3 Soft prototype loss

```python
y_target = soft_prototype_targets_all[group_id]   # [B, K]
log_probs = log_softmax(proto_logits, dim=-1)
L_proto = -(y_target * log_probs).sum(dim=-1).mean()
```

Equivalent to:

```text
KL(y_target || proto_probs)
```

Complexity is `O(B*K)`. No pairwise distance matrix, memory bank, or sampled positives/negatives is required.

### 8.4 Optional successful-force NLL calibration

This is optional and should be disabled for the first baseline.

For successful future force values:

```text
gt_force: [B, T, 6]
```

broadcast the stage prediction:

```python
mu_expand = mu_pred[:, None, :]          # [B, 1, 6]
sigma_expand = sigma_pred[:, None, :]    # [B, 1, 6]

L_nll = (
    0.5 * ((gt_force - mu_expand) / sigma_expand).pow(2)
    + torch.log(sigma_expand)
).mean()
```

Use this only for successful, non-damaging trajectories.

### 8.5 Total loss

Default:

```text
L_total =
    L_flow
    + lambda_dist * L_dist
    + lambda_proto * L_proto
```

Optional:

```text
L_total =
    L_flow
    + lambda_dist * L_dist
    + lambda_proto * L_proto
    + lambda_nll * L_nll
```

Initial weights:

```text
lambda_dist  = 1.0
lambda_proto = 0.05
lambda_nll   = 0.0 initially
```

Training schedule:

```text
Phase 1:
    train L_flow + lambda_dist * L_dist

Phase 2:
    linearly ramp lambda_proto from 0 to 0.05

Phase 3:
    jointly fine-tune enabled losses
```

---

## 9. Training Step Pseudocode

```python
def training_step(batch, model, soft_prototype_targets_all, cfg):
    rgb_base = batch["rgb_base"]
    rgb_wrist = batch["rgb_wrist"]
    language = batch["language"]
    state = batch["state"]                    # [B, 7]
    current_force = batch["current_force"]    # [B, 6]

    gt_action = batch["gt_action"]            # [B, T, 7]
    gt_safe_dist = batch["gt_safe_distribution"]  # [B, 12]
    group_id = batch["group_id"]              # [B]
    gt_force = batch.get("gt_force", None)    # optional [B, T, 6]

    outputs = model(
        rgb_base=rgb_base,
        rgb_wrist=rgb_wrist,
        language=language,
        state=state,
        current_force=current_force,
        gt_action_for_flow=gt_action,
    )

    mu_pred = outputs["mu_pred"]               # [B, 6]
    sigma_pred = outputs["sigma_pred"]         # [B, 6]
    proto_logits = outputs["proto_logits"]     # [B, K]

    mu_gt = gt_safe_dist[:, :6]                # [B, 6]
    sigma_gt = gt_safe_dist[:, 6:]             # [B, 6]
    y_target = soft_prototype_targets_all[group_id]  # [B, K]

    loss_flow = compute_original_flow_loss(outputs, gt_action)
    loss_dist = diagonal_gaussian_kl(mu_gt, sigma_gt, mu_pred, sigma_pred)
    loss_proto = soft_label_cross_entropy(proto_logits, y_target)

    loss = (
        loss_flow
        + cfg.lambda_dist * loss_dist
        + cfg.lambda_proto * loss_proto
    )

    if cfg.use_force_nll and gt_force is not None:
        loss_nll = stage_distribution_nll(gt_force, mu_pred, sigma_pred)
        loss = loss + cfg.lambda_nll * loss_nll
    else:
        loss_nll = None

    return {
        "loss": loss,
        "loss_flow": loss_flow.detach(),
        "loss_dist": loss_dist.detach(),
        "loss_proto": loss_proto.detach(),
        "loss_nll": None if loss_nll is None else loss_nll.detach(),
    }
```

---

## 10. Inference

### Inputs

```text
RGB(base)
RGB(wrist)
Language
Current State [7]
Current Force [6]
```

### Forward path

```text
RGB + Language -> VLM -> VL tokens
Current force -> force projection -> force token
VL tokens + force token -> FVLMoE
last fused force token -> z_phy

z_phy -> physical action guidance
z_phy -> stage-level safe distribution [12]
z_phy -> optional prototype probabilities [K]

G_total -> original flow action head -> action chunk [T, 7]
```

### Required outputs

```text
action_chunk:             [T, 7]
safe_force_distribution:  [12]
```

Optional chunk-shaped API output:

```text
safe_force_distribution_chunk: [T, 12]
```

created only by broadcasting the stage-level prediction.

Optional diagnostics:

```text
prototype_probs: [K]
physical_token:  [D_phy]
```

Inference does not require:

```text
gt_action
gt_safe_distribution
gt_force
group_id
soft prototype target
K-means centers
```

---

## 11. Required Model Output Dictionary

```python
{
    "action_pred": Tensor[B, T, 7],
    "safe_force_distribution": Tensor[B, 12],
    "mu_pred": Tensor[B, 6],
    "sigma_pred": Tensor[B, 6],
    "z_phy": Tensor[B, D_phy],
    "proto_logits": Tensor[B, K],
    "proto_probs": Tensor[B, K],
}
```

---

## 12. Implementation Rules

1. Use `H_fvl[:, -1, :]` as the contextualized force token.
2. Do not use cross-attention to derive `z_phy`.
3. The physical token must influence action generation through additive guidance.
4. The safe distribution head predicts stage-level `[B, 12]`, not a true temporal `[B, T, 12]`.
5. Do not duplicate a stage label over T and sum T repeated KL losses.
6. Do not use force limits, upper-bound losses, damage thresholds, fragility labels, or hand-designed material labels.
7. Prototype descriptors use only normalized safe-distribution mean and log-standard-deviation: `[mu, log(sigma)]`.
8. K-means centers and soft target tables are fixed offline metadata, not trainable parameters.
9. `group_id` is supervision-only and must never be a model input.
10. Normalize force and torque dimensions separately.
11. Build safe-distribution labels from successful, non-damaging demonstrations only.
12. Start with `K=4`, `tau_q=0.10`, `D_phy=128`, and `lambda_proto=0.05`.
13. Log predicted mean/std, prototype probabilities, and per-group losses during training.
14. Evaluate ablations: without `L_dist`, without `L_proto`, and without physical action guidance.
