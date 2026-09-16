> **已被 `todo_training_contract.md` 取代（2026-09-16）。**
> A 节的 12→13 维方案作废：输入改为原始触觉电压 + 开度 + 末端 wrench（57 维），
> 目标改为一维标量。Task 2 分组和 split-first 两节已并入新文件。此文件仅留作记录。

# 需求：力 token 加入夹爪开度 + Task 2（纸杯）支持

记录于 2026-09-14。两项改动：A 是把夹爪开度放进力输入；B 是让分组标签支持 Task 2 的纸杯。
两项都要在最终训练前完成，且必须同样应用到基线 `pi0_draftvla_forcevla`。

---

## 0. 背景与已定的决定

**为什么要加开度。** 区分 Task 2 两种条件（空杯 / 满杯）的物理量是**刚度**，而刚度只能由
力和开度**一起**确定：同样一个力读数，可能是轻捏刚性物体，也可能是用力压软物体。现在
`force` 只有 12 维指尖 wrench，开度虽然在 7 维 `state` 里，但 `state` 只进动作专家，
**不进 `z_phy`**，所以物理分支在抓取阶段拿不到可辨识刚度的信息。

两种杯子外形尺寸相同，接触起始开度一致，所以单帧的 (力, 开度) 就能区分，不需要时序历史。

**为什么用实测开度 `gripper_width`，不用指令值。** 抓取时指令值会跳到全闭（实测数据里
`gripper_action_target=0.0040`），而实测开度停在物体把手指挡住的位置（`gripper_width=0.0557`）。
带物体信息的是实测值。指令值还是动作的一部分，放进观测会让模型读到自己的动作。

**监督目标不变。** 物理分支学的仍然是**成功示教的力分布**（`gt_safe_distribution` =
mu(6) + sigma(6)，`safe_force_dim=6`）。标定扫描测出的打滑/损坏区间**不进训练**，只用于：
验证示教落在区间内、计算违规率、证明 Task 2 两种条件的区间不重叠。

**fz 保留**（六维不变），尽管它在 20260911 批次里有约 60% 的帧恒为 0。论文里会说明。

---

## A. 力向量 12 维 → 13 维

新的输入布局：`[mean_wrench(6), signed_half_difference(6), gripper_width(1)]`。

### A.1 `examples/force/convert_draftvla_data_to_lerobot.py`

1. `_two_finger_mean_difference()`：末尾拼上 `float(record["gripper_width"])`，返回 shape `(13,)`；
   函数名和 docstring 一并改（建议改成 `_force_input_vector`）；有限性检查要覆盖新加的这一维。
2. feature spec 里的 `"gripper_wrench"`：`"shape": (12,)` → `(13,)`；
   `"names"` 末尾加 `"gripper_width"`。
3. 常量 `FORCE_INPUT_SIGNAL` 的字符串加上开度（例如
   `tactile_estimated_wrench_mean_plus_signed_half_difference_plus_gripper_width`）。
4. **不要动** `state`：它第 7 维仍然是 `gripper_width`。这个重复是有意的——`state` 走动作专家，
   `gripper_wrench` 走物理分支，两条通路。

### A.2 `src/openpi/training/config.py`

把 `force_dim=12` 全部改成 `13`。注意每个 config 出现**两次**（`model=` 一次，
`freeze_filter=` 里重建 `Pi0Config` 再一次），涉及：

- `pi0_draftvla`
- `pi0_draftvla_20260905`
- `pi0_draftvla_20260905_20260911`
- `pi0_draftvla_forcevla` ← **基线必须一起改，否则两者输入不同，比较无效**

不要改：`pi0_draftvla_noforce`（没有力输入）、`pi0_force_*`（旧的 6 维法兰契约）。

### A.3 其余需要同步的位置

| 文件 | 改动 |
|---|---|
| `src/openpi/policies/draftvla_policy.py` | `make_draftvla_example()` 里 `np.random.rand(12)` → `13`；文件头部注释里的输入契约说明。`DraftVLAInputs` 本身是透传，不用改 |
| `examples/force/main.py` | `_random_observation()` 里 `np.random.rand(12 if draftvla else 6)` → `13`；上方注释里的 wrench 布局；文件头 docstring 的契约说明 |
| `examples/force/validate_draftvla_data.py` | L129 打印的输入契约文字；建议再加一条检查：每帧 `gripper_width` 存在且有限（L100 那个 12 是标签维度，**不要动**） |
| `src/openpi/models/physical_test.py` | 硬编码的 12：`_config(force_dim=12)`、`model.force_proj.in_features == 12`、`"force": np.zeros((1, 12))`；测试名 `test_12d_force_input_keeps_12d_safe_distribution_target` 也改。**`gt_safe_distribution` 的 12 保持不变** |
| `src/openpi/models/pi0_config.py` | `safe_force_dim` 上方注释里写着 "DraftVLA conditions on [mean, difference] (12-D)"，文字更新 |
| `examples/force/README.md` | 输入契约一节 |

不用改：`fvlmoe.py`、`physical.py`（`force_proj` 会跟随 `force_dim`）；量纲不用手动缩放，
`compute_norm_stats.py` 对 `force` 做 z-score，米和牛顿会被拉到同一尺度。

### A.4 数据与训练

1. 用**新的** `DATASET_REPO_ID`（例如加 `_ap13` 后缀）重新转换，**不要覆盖**服务器上已经合并好的
   20260905 + 20260911 数据集。
2. `uv run scripts/compute_norm_stats.py --config-name <config>`（`force` 的统计量维度变了）。
3. 从 `pi0_base` 重新训练 `pi0_draftvla_*` **和** `pi0_draftvla_forcevla`。
   **旧 checkpoint 不能续用**：`force_proj.in_features` 从 12 变 13。

### A.5 验收标准

- `uv run pytest src/openpi/models/physical_test.py src/openpi/models/pi0_test.py -q` 全绿。
- 转换后抽查若干帧：`gripper_wrench[12] == state[6]`（都是实测开度），且数据集 feature 的 shape 是 13。
- norm stats 里 `force` 的均值/方差是 13 维。
- 200 步 smoke 跑通，`g_phy_rel` 非零，`loss_dist` 下降。

---

## B. Task 2（纸杯）支持

### B.1 分组不能再靠 prompt 识别

`add_group_prototype_labels.py` 现在用 `fruit_from_prompt()` 从 prompt 里匹配水果名来确定组。
**空杯和满杯的 prompt 完全相同**，这条路直接失效，而且这正是任务设计的要点。

要求：采集时就把条件写进每条 episode（建议每帧一个字符串字段，例如
`group_condition ∈ {cup_empty, cup_filled, cup_half}`，或者 episode 目录下一个 meta 文件），
标注脚本从这个字段读条件，**不要从 prompt 推断**。

### B.2 组的定义从 fruit 泛化成 condition

group key 从 `(task, fruit, stage)` 改成 `(task, condition, stage)`，Task 1 的 condition 就是
水果类别，Task 2 的是装填量。涉及：

- `add_group_prototype_labels.py`：`FRUIT_FIELD`、`fruit_from_prompt()`、group key 组装。
- `build_safe_group_prototypes.py`（**两份副本**：post-process 目录里一份，
  `openpi_draftvla/examples/force/` 里一份）：`FRUITS`、`GROUP_KEYS`、`group_fruit`。
- 每帧 JSONL 字段 `group_fruit` 建议改名 `group_condition`（或保留旧名但语义改为 condition，
  需在两处副本里一致）。

### B.3 合并后重建一张 prototype 表

- 果蔬 6 类 × 4 阶段 = 24 组，加上纸杯 2 条件 × 4 阶段 = 8 组，合计 **32 组**。
- 重新选择 K（现在是 4），并更新 `train_config_*.conf` 里的 `EXPECTED_PROTOTYPE_GROUPS`。
- 半杯（中间装填量）**只做评估**：不进示教、不进 prototype 表、不进 group 统计。

### B.4 split-first：先划分，再建标签表

**现状：完全没有划分，全量训练。** `scripts/train.py` 没有任何验证逻辑，
`data_loader.py` 直接把整个 LeRobot 数据集拿去训练（上游 openpi 就没有 val split），
唯一被排除的是 denylist 里的失败 episode，那是在转换阶段剔除的，不是划分。
`train_config.conf` L105-112 的注释也写明当前配置不定义验证集。

**为什么必须做：**

1. **选不了 checkpoint。** 没有验证曲线，只能盲取最后一步。
2. **发现不了物理分支的记忆行为。** 分布目标在组内是常数，小 MLP 很容易背下来，
   而且可能背的是每条 episode 的个体特征（例如传感器偏置），训练集上的 `loss_dist` 看不出来。
3. **论文的消融表需要它。** "分布误差"那一列是 Q3 的核心证据（视觉-语言变体在 Task 2 上学不到分布）。
   在训练数据上算出来的数没有说服力。
4. **论文 data 一节已经写明**：统计量、归一化参数、聚类中心、软目标只在训练 episode 上拟合，
   划分以物体实例和采集时段为单位。

**划分原则：** 每个 condition（果蔬 6 类 + 纸杯 2 条件）留 1–2 条 episode，总量 15–20%；
按物体实例和采集时段分层，同一个物理实例不要同时出现在两边。
注意：泛化用的是评估专用物体（半杯、牛油果、西葫芦），和 episode 划分是两件事。

**操作步骤（脚本已有的参数都能用）：**

```bash
# 1) 建标签表：统计只用训练 episode，验证 episode 另外写 labels.jsonl sidecar
uv run examples/force/build_safe_group_prototypes.py \
    --data-dir <合并后的 post-process 目录> \
    --val-episodes <val_ep_1> --val-episodes <val_ep_2> ... \
    --k <重选后的 K> --tau-q 0.1 \
    --output-dir prototype_metadata_trainonly --write

# 2) 转训练集（只含训练 episode）
uv run examples/force/convert_draftvla_data_to_lerobot.py \
    --data-dir <同上> --repo-id draftvla/<name>_train_ap13 \
    --episodes <train_ep_1> --episodes <train_ep_2> ... \
    --labels-dir prototype_metadata_trainonly

# 3) 转验证集（只含验证 episode，标签同样来自训练集拟合的表）
uv run examples/force/convert_draftvla_data_to_lerobot.py \
    --data-dir <同上> --repo-id draftvla/<name>_val_ap13 \
    --episodes <val_ep_1> --episodes <val_ep_2> ... \
    --labels-dir prototype_metadata_trainonly

# 4) norm stats 只在训练集上算
uv run scripts/compute_norm_stats.py --config-name <train config>
```

训练 config 的 `repo_id` 指向 `_train_ap13`；基线 `pi0_draftvla_forcevla` 用同一个训练集。

**需要新增：离线评估脚本**（现在没有这个入口，约几十行）

- 建议路径：`examples/force/eval_draftvla_offline.py`。
- 输入：checkpoint 目录 + 验证集 repo_id + config 名。
- 做法：按训练同样的 transform 构造 batch，调 `model.compute_train_losses(...)`，
  但**只取指标不反传**；`supervision_valid` 掩码沿用。
- 输出指标：动作 MSE（`loss_flow`）、分布误差（`loss_dist`，即式 (6) 的 KL）、
  `proto_acc`、`loss_dist_baseline`（对照）、`g_phy_rel`，全部**按阶段分别报告**
  （grasp 阶段是瓶颈，见 C 节）。
- 频率：训练中每 500–1000 步跑一次，或训练结束后对保留的若干 checkpoint 各跑一次。
- **checkpoint 按验证集的动作 MSE + 分布误差选**，不要取最后一步；
  验证曲线放论文附录。

**验收标准：**

- `prototype_metadata_trainonly/` 里的 `included_episodes` 不包含任何验证 episode；
- 验证集 LeRobot 数据集的帧数 + 训练集帧数 = 合并后总帧数（减去 denylist）；
- 离线评估脚本在验证集上跑通，并给出逐阶段指标；
- 论文附录里贴出划分清单（哪些 episode 属于验证集，按 condition 分组）。

### B.5 标定

纸杯的每个条件都要测 f_min（打滑下界）和 f_max（损坏上界），**包括半杯**——违规率和泛化分析要用。
抓握力的定义是：两指平均指尖 wrench 的抓握法向分量（论文里要注明是哪一维）。

---

## C. 暂不处理，但要记住

- **grasp 阶段样本很少**：20260911 批次里 885 / 22,933 帧（3.9%），平均每条示教约 22 帧，
  而抓握力决策恰恰在这个阶段。必要时考虑给 `L_dist` 按阶段加权，或在采集时延长闭合过程。
- **消融分支还没有代码开关**：no-guidance（去掉 `G_phy`）、vision--language token、oracle-group。
  其中 vision--language 变体**必须从 FVLMoE 融合之前的 VLM 前缀取 token**，
  否则自注意力已经把力信息混进去，消融证明不了任何事。
  architecture-only、distribution-only、prototype-only 用现有的 `LAMBDA_DIST` / `LAMBDA_PROTO` 就能实现。
