# 实施说明：消融开关 B1 / B2 / B3

定于 2026-09-22。**给谁看**：在 Leonardo 上改这个仓库代码的人。
照着这份文件改，不要自行发挥；每一处改动都标了「为什么」，动机不成立的改法在 §6 列了。

**先回流。** 本机训练仓 `/home/lar-ur/vla/DamageVLA_task12_training` 是 Leonardo 副本**和部署镜像**的唯一源头。
在 Leonardo 上改完必须把 diff 送回本机仓，否则这三条 arm 有 checkpoint 但没有能服务它们的镜像，上不了机器人。

---

## 0. 不变量（先读这条，它是验收的总判据）

> **两个新字段的默认值必须让现有三条 arm 的行为逐比特不变。**
> `phy_guidance=True`、`phy_source_vl=False` 就是 PiVLA。把某条 arm 改回我们的方法 =
> 把这两个值设回默认（或删掉那两行），**不需要动 `pi0_draftvla_task12_v2` 的任何一行**。

改完之后这条必须成立：`pi0_draftvla_task12_v2` 的配置文件一个字符都没变，而且
`uv run pytest src/openpi/models/physical_test.py src/openpi/models/pi0_test.py -q` 全绿。

---

## 1. 主线

| | 步骤 | 判据 | 不过怎么办 |
|---|---|---|---|
| 1 | `pi0_config.py` 加两个字段（§3.1） | 默认值 = PiVLA | — |
| 2 | `pi0.py` 改 `_force_guidance` 和两个调用点（§3.2） | 见 §4 测试 1 | 默认路径变了就是改错了 |
| 3 | `pi0.py` 改 `probe_features`（§3.3） | 见 §4 测试 5 | 漏了会让离线探针读到模型根本没用的 token |
| 4 | `config.py` 加三个 TrainConfig（§3.4） | `uv run python -c "import openpi.training.config"` 不报错 | — |
| 5 | 三个 conf + `run_train.sh` + `submit_leonardo.sh`（§3.5–3.7） | 启动时打印的命令里 λ 和 MODE 对得上 | — |
| 6 | **新增 5 个测试，全绿**（§4） | 全绿 | **这是 go/no-go 闸门，不绿不准提交 10k** |
| 7 | 三条 200 步 smoke（§5） | 见 §5 的表 | 按表里的「不及格表现」回查 |
| 8 | 提交三条 10k 作业 | — | — |

第 6 步是唯一的硬闸门：一条开关接错，浪费的是三个 GPU-day 外加一轮真机排期。

---

## 2. 三条 arm 的定义

我们在 ForceVLA† 之上加的东西可以拆成三件可独立开关的：**模块/容量**、**物理监督**、**直接通路 $G_{phy}$**。
三条消融各关掉一件（B3 关的是第四件：表征的信息来源）。

| arm | 配置名 | 唯一改动 | 隔离出什么 |
|---|---|---|---|
| **B1** architecture-only | `pi0_draftvla_task12_v2_archonly` | `lambda_dist=0.0, lambda_proto=0.0` | **容量**。模块和 $G_{phy}$ 全留着，只是没有物理监督 |
| **B2** no extra guidance | `pi0_draftvla_task12_v2_noguid` | `phy_guidance=False` | **通路**。监督照旧，只是 $z_{phy}$ 不再直接进动作 |
| **B3** vision-language token | `pi0_draftvla_task12_v2_vltoken` | `phy_source_vl=True` | **信息来源**。$z_{phy}$ 改从融合前的 VL 前缀读 |

三条都从 `pi0_draftvla_task12_v2` 复制，**除上表的「唯一改动」外一个字段都不许变**——
包括 `phy_action_gain_init=13.0`、原型 ramp、`phy_label_mean/scale`、数据集、freeze filter、
`batch_size=16`、`num_train_steps=10_000`。有第二处差异，这条 arm 就什么都证明不了。

**B1 不需要任何代码改动**，`lambda_dist` / `lambda_proto` 已经是现成字段。代码只为 B2 和 B3 改。

---

## 3. 代码改动

### 3.1 `src/openpi/models/pi0_config.py`

在 `phy_action_gain_init` 之后、`phy_label_mean` 之前插入：

```python
    # === Ablation switches (2026-09-22). THE DEFAULTS ARE PiVLA. ===
    # Leaving both untouched reproduces the full method bit-for-bit, so the three main arms need no
    # edit; setting a switch back to its default here is the whole "restore PiVLA" operation.
    #
    # B2 "no extra guidance": False drops G_phy from the action path. z_phy, both heads and both
    # auxiliary losses stay on -- the supervision still reaches the action expert through the shared
    # FVLMoE (G_fvl), so this variant subtracts the direct pathway alone, not the supervision.
    phy_guidance: bool = True
    # B3 "vision-language token": True reads z_phy from the last VALID VLM prefix token, taken
    # BEFORE FVLMoE fusion, instead of the fused force token. FVLMoE and G_fvl are untouched, so
    # force still reaches the action expert; only the origin of the representation moves. Any
    # post-fusion token would already have attended to the force token and prove nothing.
    phy_source_vl: bool = False
```

在 `__post_init__` 的第一个 `if` 之前加一条守卫，防止把开关设在没有物理分支的 arm 上而悄无声息：

```python
        if (not self.phy_guidance or self.phy_source_vl) and not self.phy_enabled:
            raise ValueError(
                "phy_guidance / phy_source_vl are ablations OF the physical branch and require "
                f"phy_enabled=True (got phy_enabled={self.phy_enabled})."
            )
```

### 3.2 `src/openpi/models/pi0.py` — `_force_guidance`（唯一的注入点）

这个方法被训练和推理**共用**，所以两个开关各自只改这一处，不存在两边漂移。

**签名加一个参数**（B3 要用 prefix mask 找最后一个有效 token）：

```python
    def _force_guidance(
        self,
        prefix_out: at.Float[at.Array, "b n d"],
        force: at.Float[at.Array, "b f"],
        prefix_mask: at.Bool[at.Array, "b n"] | None = None,
    ) -> tuple[at.Float[at.Array, "b t da"], dict[str, at.Array] | None]:
```

**方法体**，把现在的

```python
        fused, hidden = self.fvlmoe(prefix_out, force_token, return_hidden=True)
        g_fvl = fused[:, -self.action_horizon :, :]
        z_phy = self.phy_proj(hidden[:, -1, :])
        g_phy = self.phy_action_proj(z_phy)
```

改成

```python
        fused, hidden = self.fvlmoe(prefix_out, force_token, return_hidden=True)
        g_fvl = fused[:, -self.action_horizon :, :]
        if self.phy_source_vl:
            # B3: the last VALID prefix token, pre-fusion. `hidden[:, -1]` would be the fused force
            # token (the thing we are ablating away), and `prefix_out[:, -1]` is often padding --
            # the prefix is padded to a fixed length, so the index must come from the mask.
            if prefix_mask is None:
                raise ValueError("phy_source_vl=True requires prefix_mask at every _force_guidance call site.")
            last = jnp.sum(prefix_mask, axis=1) - 1
            phy_src = jnp.take_along_axis(prefix_out, last[:, None, None], axis=1)[:, 0]
        else:
            phy_src = hidden[:, -1, :]
        z_phy = self.phy_proj(phy_src)
```

再把 `g_phy` 的计算和相加改成条件的：

```python
        if self.phy_guidance:
            g_phy = self.phy_action_proj(z_phy)
            g_phy_rms = jnp.sqrt(jnp.mean(g_phy.astype(jnp.float32) ** 2))
            guidance = g_fvl + g_phy.astype(g_fvl.dtype)
        else:
            # B2: the module stays INSTANTIATED so the parameter tree -- and hence the checkpoint
            # layout and the capacity argument against B1 -- matches the full model exactly. Its
            # output is simply never added, so it receives no gradient. Report 0 so `g_phy_rel`
            # reads "pathway off" in the logs instead of a frozen-at-init number.
            g_phy_rms = jnp.zeros((), jnp.float32)
            guidance = g_fvl
```

`phy` 字典里 `"g_phy_rms": g_phy_rms`（不再是 `jnp.sqrt(...)` 的内联式），
末尾 `return g_fvl + g_phy.astype(g_fvl.dtype), phy` 改成 `return guidance, phy`。

**两个调用点都补上 `prefix_mask`**（两处的作用域里都已经有这个变量，不用另算）：

- `_flow_loss_and_phy` 里：`guidance, phy = self._force_guidance(prefix_out, observation.force, prefix_mask)`
- `_sample_actions_and_physical` 里：同样加第三个实参。

### 3.3 `src/openpi/models/pi0.py` — `probe_features`

离线探针必须读模型**真正在用**的那个 token，否则 B3 的 checkpoint 上探出来的 `z_phy` 是模型从未见过的东西。
把

```python
            if self.phy_enabled:
                feats["z_phy"] = self.phy_proj(fused).astype(jnp.float32)
```

改成

```python
            if self.phy_enabled:
                if self.phy_source_vl:
                    last_idx = jnp.sum(prefix_mask, axis=1) - 1
                    phy_src = jnp.take_along_axis(prefix_out, last_idx[:, None, None], axis=1)[:, 0]
                else:
                    phy_src = fused
                feats["z_phy"] = self.phy_proj(phy_src).astype(jnp.float32)
```

（保持模型 dtype 再转 float32，不要拿已经转成 float32 的 `feats["vl_prefix_last"]` 去喂 `phy_proj`。）

### 3.4 `src/openpi/training/config.py` — 三个 TrainConfig

紧跟在 `pi0_draftvla_task12_v2_noforce` 之后插入。**逐字复制 `pi0_draftvla_task12_v2`**，
只改 `name` 和下表的那一个字段：

| 新 name | 相对 v2 的唯一差异 |
|---|---|
| `pi0_draftvla_task12_v2_archonly` | `lambda_dist=0.0,` `lambda_proto=0.0,` |
| `pi0_draftvla_task12_v2_noguid` | `phy_guidance=False,` |
| `pi0_draftvla_task12_v2_vltoken` | `phy_source_vl=True,` |

注意：

- `freeze_filter=` 里那份 `Pi0Config(...)` 副本**原样照抄，不要加新开关**。它只按参数树路径匹配，
  开关对它没有意义；三条 arm 的参数树本来就和 full 一致。
- `weight_loader` 的 `extra_missing_regex` 原样保留 `phy_action_proj`——B2 里这个模块仍然存在。
- `phy_proto_ramp_start/steps` 在 B1 里保持 1000/2000 不变：λ 已经是 0，ramp 乘出来还是 0，
  但改了它就多出一处差异。

每条加一行注释说明它隔离什么，照 §2 的表写。

### 3.5 三个 conf 文件

从 `train_config_task12_v2.conf` 复制，文件名 `train_config_task12_v2_{archonly,noguid,vltoken}.conf`，
只改三处：

| 文件 | `MODE=` | `EXP_NAME=` | 其他 |
|---|---|---|---|
| `..._archonly.conf` | `task12_v2_archonly` | `task12_v2_archonly_full` | **`LAMBDA_DIST=0`、`LAMBDA_PROTO=0`** |
| `..._noguid.conf` | `task12_v2_noguid` | `task12_v2_noguid_full` | λ 保持 1.0 / 0.05 |
| `..._vltoken.conf` | `task12_v2_vltoken` | `task12_v2_vltoken_full` | λ 保持 1.0 / 0.05 |

conf 里的 λ 必须和 `config.py` 里写的一致。启动时脚本会打印完整命令，**扫一眼确认 `--model.lambda-dist`
是你要的值**——conf 会覆盖 `config.py`。两个新开关**不走 conf**，只在 `config.py` 里定义（理由见 §6）。

### 3.6 `run_train.sh`

两处都要加，漏第二处会让 conf 里的 λ 被静默忽略：

1. `MODE -> CONFIG` 的 `case` 里加三行：
   ```bash
   task12_v2_archonly)  CONFIG=pi0_draftvla_task12_v2_archonly ;;
   task12_v2_noguid)    CONFIG=pi0_draftvla_task12_v2_noguid ;;
   task12_v2_vltoken)   CONFIG=pi0_draftvla_task12_v2_vltoken ;;
   ```
   以及 `*)` 分支的错误信息里补上这三个名字。
2. FVLMoE 超参的 `case "$MODE" in ... task12_v2|task12_v2_forcevla)` 那一行，和物理超参的
   `if [ "$MODE" = "task12" ] || ... || [ "$MODE" = "task12_v2" ]` 那一行，**都要把三个新 MODE 加进去**。

### 3.7 `submit_leonardo.sh`

加三个 mode，参数和 `full_task12_v2` 完全一样（`WALLTIME=24:00:00`、`NUM_GPUS=1`、`NUM_CPUS=16`、`MEMORY=120G`），
并更新 `Usage:` 那一行：

```bash
  full_task12_v2_archonly)
    CONFIG_FILE=train_config_task12_v2_archonly.conf
    WALLTIME=24:00:00
    NUM_GPUS=1
    NUM_CPUS=16
    MEMORY=120G
    ;;
```
（`noguid` / `vltoken` 同理。）

---

## 4. 必须新增的 5 个测试

写进 `src/openpi/models/physical_test.py`，照现有的
`test_gain_default_preserves_legacy_behavior` 和 `test_phy_disabled_is_bit_identical_to_forcevla` 的写法。

| # | 测试 | 断言 |
|---|---|---|
| 1 | `test_ablation_switch_defaults_are_bit_identical_to_pivla` | 同一 rng seed 下，显式传 `phy_guidance=True, phy_source_vl=False` 的模型和不传的模型，`_force_guidance` 输出**逐比特相同** |
| 2 | `test_no_guidance_removes_g_phy_but_keeps_the_supervision` | `phy_guidance=False` 时：返回的 guidance 与同权重模型的 `g_fvl` 逐比特相同；`g_phy_rms == 0`；**`phy_action_proj` 的梯度为 0，而 `phy_proj` / `phy_dist_head` 的梯度非 0** |
| 3 | `test_vl_source_reads_the_last_valid_prefix_token_not_padding` | 构造一个 prefix 有 padding 的 batch；`z_phy == phy_proj(prefix_out[按 mask 取的最后一个])`，且 **≠** `phy_proj(hidden[:, -1])`，也 ≠ `phy_proj(prefix_out[:, -1])` |
| 4 | `test_vl_source_z_phy_is_independent_of_the_force_input` | `phy_source_vl=True` 时，扰动 `observation.force`，`z_phy` **完全不变**，而 `g_fvl` 变了 |
| 5 | `test_probe_features_z_phy_follows_the_source_switch` | `probe_features()["z_phy"]` 在两种开关下分别等于 `_force_guidance` 内部用的那一个 |

**测试 4 是最要紧的一个**：B3 的全部意义就是「这个表征看不见力」。如果实现里不小心读到了任何融合后的量，
只有这条测试抓得住，而其他四条都会绿。

跑：`uv run pytest src/openpi/models/physical_test.py src/openpi/models/pi0_test.py -q`，**全绿才继续**。
特别确认这两条老测试没红：`test_phy_disabled_is_bit_identical_to_forcevla`、`test_g_phy_has_gradient_at_the_training_site`。

---

## 5. 200 步 smoke 的验收

三条各跑一次 200 步（把 smoke conf 的 `MODE` / `NUM_TRAIN_STEPS` 改一下即可），对照下表读日志：

| arm | 应该看到 | 不及格的表现 → 查什么 |
|---|---|---|
| **B1** archonly | `loss_flow` 和 full 同量级；`loss_dist` **不下降**（它没有梯度）；`g_phy_rel` 非零 | `loss_dist` 在降 → λ 没生效，查打印出来的 `--model.lambda-dist` |
| **B2** noguid | `g_phy_rel` **严格等于 0**；`loss_dist` 正常下降 | `g_phy_rel ≠ 0` → 开关没接上；`loss_dist` 不降 → 误把监督也删了 |
| **B3** vltoken | `loss_dist` 会降，但应**明显高于** full 同步数；`z_phy_cos` 不趋近 1 | `z_phy_cos → 1` → 表征塌缩，查是不是取到了 padding token |

三条的 `action_weight_mean` 应该和 v2 full 完全一致（同一份数据）。不一致说明数据集用错了。

通过后提交 10k：

```bash
bash submit_leonardo.sh full_task12_v2_archonly
bash submit_leonardo.sh full_task12_v2_noguid
bash submit_leonardo.sh full_task12_v2_vltoken
```

三条互相独立，可并行排队。训完每条都要过 **G1 离线闸门**
（`examples/force/g1_gate_offline.py`，判据见 `notes/eval_logging_spec.md` §2），
再决定哪几条值得花真机时间。

---

## 6. 明确不做的事

- **不做 oracle-group**（2026-09-22 决定）。
- **不要为了 B2 把 `phy_action_gain_init` 设成 0**。`gain` 是可学习参数，梯度会把它拉回来——
  那是个初始化，不是开关。必须从加法里真正拿掉。
- **不要物理删除 `phy_action_proj` 或两个头**来实现 B1/B2。参数量一变，B1 就不再是干净的容量对照，
  「A2→B1 的差 = 参数买到的」这句话就没法说了。
- **不要用 `hidden[:, -1]` 或 `prefix_out[:, -1]` 实现 B3**。前者是融合后的力 token（正是要消融掉的东西），
  后者多半是 padding。必须按 mask 取最后一个有效 token。
- **不要把两个新开关做成 conf 变量 / CLI 覆盖**。它们是 arm 的身份，不是超参；放 `config.py` 里
  一条 arm 一行，既不用碰 tyro 的布尔 flag 拼写，也让「这条 arm 到底是什么」只有一个地方可查。
- **不要改 `pi0_draftvla_task12_v2` / `_v2_forcevla` / `_v2_noforce` 的任何一行。**

---

## 7. 结果出来时怎么读（先写在这里，免得事后解释）

- **B1**：安全成功率若退回 ForceVLA† 附近 → 参数本身不买账，改进来自监督。这是预期结果。
- **B2**：预期签名是**分布误差几乎不变、安全成功率掉**——表征学到了却到不了行为。
  两个指标必须分开报，否则这一行读不出来。
- **B3**：预期 Task 1 与 full 持平（类别可见）、Task 2 垮（空/满看不出来）。

**B3 有一个必须提前说清的坑。** 2026-09-21 的探针发现：**VL 前缀的最后一个 token 在接触之前就能以 0.93 的
准确率分出 carton_01（空）和 carton_02（满）**——两个纸盒实例外观可分，示教里存在实例泄漏
（`examples/force/probe_20260921/`）。探针是事后拟合、梯度不回传主干，没能利用这个泄漏；
**B3 是端到端训练的，很可能把它吃干净**，于是「VL token 在 Task 2 上也行」，看起来像我们的主张被证伪，
实际上只是泄漏。

对策，**现在就定，不要等结果出来再解释**：

1. B3 的真机评估只在**未见印花**的纸盒上做（评估协议本来就这么要求）；
2. 离线分布误差也要报在未见实例的留出 episode 上；
3. 论文里写明训练分布中存在这个实例混淆。
