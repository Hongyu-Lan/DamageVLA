# 需求：新的训练数据契约（方案 B）+ Task 2 支持

**本次改动的四份文档**（都在 `outlines/`，同步到服务器后一起读）：

| 文件 | 内容 |
|---|---|
| `todo_training_contract.md`（本文件） | 训练数据契约与逐文件改动清单，**照着它改代码** |
| `data_analysis_20260916.md` | 83 条 episode 的实测分析：各条件的力、方差来源、K 扫描、要排除的 episode |
| `collection_protocol.md` | 采集与标定协议（分段闭合、盲测、标定流程与元数据字段） |
| `analyze_groups.py` | 上述分析的脚本，数据更新后重跑：`python3 analyze_groups.py <果蔬目录...> --carton <纸盒目录>` |

2026-09-16 定稿，取代 `todo_aperture_and_task2.md` 里的 A 节（那份里的 12→13 维方案已作废，
Task 2 分组和 split-first 两节仍然有效，本文件也一并收录）。

所有改动必须**同样应用到基线 `pi0_draftvla_forcevla`**，否则两者输入不同，比较无效。

---

## 0. 已定的决定

| 项 | 决定 | 理由 |
|---|---|---|
| 输入 | 原始触觉电压 `_data`（2×25）+ 实测开度（1）+ 末端 wrench（6）= **57 维** | 估算指尖 wrench 误差大；原始电压信息最全，让网络自己学映射 |
| 估算指尖 wrench | **不再使用**（输入和目标都不用） | 6 轴解算不准，$f_z$ 有约 60% 的帧恒为 0 |
| 用 `_data` 还是 `_raw` | **`_data`** | 实测：`_raw` 未接触 35.68 → 接触 37.41（对比度 5%）；`_data` 0.01 → 1.74（对比度 170 倍）。逐 taxel 相关系数 1.000，`_data` 就是 `_raw` 减掉基线 |
| 目标 | **1 维标量**：两指归零后 `_data` 电压总和的均值，组级 $(\mu,\sigma)$ | 论文里的安全区间本来就是标量；一维标定可行，6 轴不可行；分布、区间、违规率进入同一空间 |
| 单位换算（电压→牛顿） | **训练前不做**，事后补或不补 | 共享尺度的仿射变换不改变高斯 KL。**例外见 G 节**：标定曲线若明显非线性，必须在建标签前套上单调映射 |
| 切向维度 | **暂不做** | 切向需求本质是物体重量，已由末端 wrench 进入输入；压心代理需单独验证，边际价值低 |
| 原型描述子 | 与目标解耦，用数据选特征（D 节） | 目标降成标量后，若描述子仍是 $[\mu,\log\sigma]$，原型退化成力的分箱 |

---

## 0b. 基于实测数据的决定（83 条 episode，8 个条件，详见 `data_analysis_20260916.md`）

| 决定 | 内容 | 依据 |
|---|---|---|
| **阶段** | 标注保留四个接触阶段，但**分布目标按两组统计：`grasp` 和 `hold`（lift+translate+place）** | mu 的 F_stage 只有 0.11，夹爪闭合到位后力基本不变；合并后每组样本翻倍 |
| **描述子** | `[mu, log sigma, 接触面积, 压心偏移]` 逐阶段组 + `刚度` 条件级 | 单特征 F_condition：压心 2.37、面积 1.30，而 mu 只有 0.44、log sigma 0.36。只用 [mu, log sigma] 聚类会把空盒和满盒的 hold 归进同一簇 |
| **K** | **K=6**（16 组时）。silhouette 在 K=4/5/6 完全持平（0.397–0.399）、稳定性都接近 1，两个指标选不出 K，按可解释性选：K=6 是最小的能让**满盒的持握单独成簇**的 K。K=4 也可接受且与现配置一致 | 见分析文档第 5 节，扫描表要进论文附录 |
| **sigma 下限** | 建标签时 `sigma_g` 钳制到不低于 **0.0051**（= 单 taxel 噪声 0.00069 × sqrt(50)） | 空盒示教力接近传感器分辨率，sigma 会小于测量噪声 |
| **统计量** | **保持均值和标准差**，不用中位数/MAD | 目标是高斯、损失是 KL，均值和标准差是高斯的最大似然估计，换稳健估计前后不一致，也不直观 |
| **排除规则** | 不改统计量，改纳入统计的数据（见下） | 一条离群 episode 就能让满盒 sigma 从 0.12 变成 1.20，两个条件的分布彻底重叠 |

### 排除规则（建标签时执行，被排除的进 denylist 并在附录报告数量）

**首选（标定齐全后）**：episode 的阶段组均值落在该条件标定区间 $[f_{\min}, f_{\max}]$ 之外就排除。
超上界 = 抓过头，本来就不是"安全示教"；低于下界 = 没夹住或传感器没接触好。

**临时规则（标定还没覆盖的条件）**，`analyze_groups.py` 已实现：

1. 组内 3×MAD 之外 → 抓过头；
2. 接触段超过 5×噪声的 taxel 少于 3 个 → 传感器没接触好（**`carton_empty` 豁免**，它本来就轻）；
3. 持握段均值低于 10×噪声 → 疑似没夹到（**`carton_empty` 豁免**）。

当前标记出的 episode：

| 条件 | episode | mu | 处理 |
|---|---|---|---|
| carton_full | `pi0_train_20260916_015938_carton_02_full` | 4.43 | 排除（25/25 taxel 饱和、闭合 44.7 mm） |
| cucumber | `pi0_train_20260911_002313` | 4.61 | 排除 |
| cucumber | `pi0_train_20260911_003740` | 4.94 | 排除 |
| kiwi | `pi0_train_20260910_234336` | 2.84 | 排除 |
| banana | `pi0_train_20260903_234527` | 0.016 | 已在 conf 的 `EXCLUDE_EPISODES` |
| banana | `pi0_train_20260903_234727` | 0.184 | 回看视频后定 |

三条空纸盒（`023258`/`023845`/`024043`，mu 0.001–0.003）**保留**：已核查传感器正常，
峰值仍是噪声的 7–17 倍，只是空盒太轻、碰到即可夹起。

**论文要补的一句**（limitations）：空盒条件的示教力接近传感器分辨率下限（信噪比 7–17 倍），
该条件的 sigma_g 主要反映测量噪声而非物理容差。

---

## 0c. 头号风险：物理引导的幅度（上一次训练的实测）

上一次 10,000 步的完整训练（`examples/force/train_draftvla_20260905_20260911_full.log`）里：

```
loss_flow  = 0.0276     占总损失 54%
loss_dist  = 0.0057     占 11%      (lambda=1)   对照 loss_dist_baseline = 0.6316
loss_proto = 0.3394     x0.05 = 0.0170  占 33%
g_phy_rel: step 0 = 0.0225  ->  step 9920 = 0.0279
```

两个结论：

1. **lambda_proto = 0.05 不小**。原型损失的数值比分布损失大 60 倍，实际贡献（0.017）反而超过分布项（0.0057），
   不需要提高。但 **ramp 建议从 4k–8k 提前到 1k–3k**：10k 步的训练里，原型损失前 40% 完全关闭、
   只有最后 20% 是满权重。
2. **分布任务太容易**：650 步时 `loss_dist` 就比平凡基线好 110 倍。组内目标是常数，认出是哪一组就赢了。
   降到 1 维目标后只会更容易。所以提高 `lambda_dist` 没有意义。

**真正的风险是第三点：`g_phy_rel` 跑完 10,000 步几乎没动（0.0225 -> 0.0279）。**

`g_phy_rel = ||G_phy|| / ||G_fvl||`，即物理引导相对 ForceVLA 引导的幅度。**3% 意味着物理分支几乎没有影响动作**，
方法实际上退化成"ForceVLA + 两个没人用的辅助头"，而消融表里"去掉 G_phy"那一行会显示没区别——等于自证无效。

原因：$z_{\mathrm{phy}}$ 被 L2 归一化（模长固定为 1），幅度只能由 `PhysicalActionProjector` 的权重决定；
而流匹配损失不需要它（G_fvl 已经带了力信息），所以没有梯度把它推大。

**建议改动（优先级高于任何超参调整）**：在 `physical.py` 的 `PhysicalActionProjector` 输出上加一个
**可学习的标量增益 alpha**，初始化到使 `g_phy_rel ≈ 0.3`。理由：

- 让模型自己决定要不要用这条通路；
- alpha 的最终值本身就是"物理表征对行为有多重要"的直接证据，可以写进论文；
- 如果训练后 alpha 被压回接近 0，那是一个诚实的负面结果，也必须知道；
- 论文里"推理时缩放物理引导"的分析正好复用同一个标量。

**训练时的头号监控指标就是 `g_phy_rel`**：跑完先看它。如果仍在 2–3%，先解决这个再谈其它，
否则后面的真机实验大概率白跑。

---

## A. 输入：57 维

新布局 `[left_data(25), right_data(25), gripper_width(1), flange_wrench(6)]`，全部在**去零之后**（见 C 节）。

| 文件 | 改动 |
|---|---|
| `examples/force/convert_draftvla_data_to_lerobot.py` | `_two_finger_mean_difference()` 整个换掉：读 `tactile_voltage_signals.{left,right}_data`、`gripper_width`、`force_torque`，按上面顺序拼成 57 维；feature spec 的 shape (12,)→(57,)，`names` 相应改；常量 `FORCE_INPUT_SIGNAL` 改名 |
| 同上 | 建议把 key `gripper_wrench` 改名为 `contact_input`（内容已经不是 wrench 了，论文里也叫 contact input）。改名要同步 `draftvla_policy.py`、`main.py`、README |
| `src/openpi/training/config.py` | `force_dim=12` → `57`，四个 draftvla 配置各出现两次（`model=` 和 `freeze_filter=`），**含 `pi0_draftvla_forcevla`** |
| `src/openpi/policies/draftvla_policy.py` | `make_draftvla_example()` 的随机向量 12 → 57；文件头契约说明 |
| `examples/force/main.py` | `_random_observation()` 12 → 57；注释里的布局；真机客户端要发同样的 57 维 |
| `src/openpi/models/pi0_config.py` | `safe_force_dim` 上方注释里"12-D"的说明 |

`fvlmoe.py`、`physical.py` 不用动，`force_proj` 跟随 `force_dim`。

---

## B. 目标：1 维

`gt_safe_distribution` 从 12 个数（μ×6, σ×6）变成 **2 个数**（μ, σ）。

标量定义（固定公式，建标签和算区间都用它）：

```
grip_t = 0.5 * ( sum_{i=1..25} left_data_zeroed[i, t] + sum_{i=1..25} right_data_zeroed[i, t] )
```

| 文件 | 改动 |
|---|---|
| `src/openpi/training/config.py` | `safe_force_dim=6` → `1`；`phy_label_mean` / `phy_label_scale` 改成单元素元组，数值由新标签重算 |
| `examples/force/convert_draftvla_data_to_lerobot.py` | `gt_safe_distribution` feature (12,)→(2,)；`NULL_SAFE_DISTRIBUTION` 改成 `[0.0, 1.0]`（σ 仍然不能为 0） |
| `src/openpi/models/pi0.py` | `dim_names` 从 `["fx",...,"tz"]` 改成 `["grip"]`（只影响日志键名） |
| `examples/force/validate_draftvla_data.py` | `len(safe) != 12` → `!= 2`；打印的契约说明 |
| `src/openpi/models/physical_test.py` | 硬编码的 12 和 6：`force_dim`、`safe_force_dim`、`force_proj.in_features`、`np.zeros((1,12))`、测试名 |
| `add_group_prototype_labels.py`（post-process） | 写出的 `gt_safe_distribution` 变成 2 个数 |

KL 那边不用改：`diagonal_gaussian_kl` 对任意维度成立，1 维时"除以 6"自然变成"除以 1"。

---

## C. 每个 episode 归零（目前**完全没有实现**，必须补）

论文附录声称"法兰和指尖通道都用每个 episode 开头的静止张开窗口归零"，但管线里没有任何地方做这件事：

- `_data` 只在驱动启动时去过一次基线，session 内的漂移没有处理；
- 末端 wrench 完全没去偏，静止时 fx 就有约 −9.5 N 的工具重力项。

要求：在转换脚本里，对每条 episode 取开头一段**夹爪张开且静止**的窗口（建议 `stage == "prepare"` 且 `cmd_speed_l` 全为 0 的前 N 帧，N 取 10–20），算均值后从该 episode 的
`_data`（50 维）和 `force_torque`（6 维）中减掉。窗口不足 N 帧的 episode 记为异常并排除。

末端 wrench 的重力投影随姿态变化，开头归零只消掉初始姿态那一份。**不做完整负载辨识**：运动轮廓固定，姿态项在两种条件之间是共同的，且 TCP 旋转向量本来就在 state 里。附录要写明这一点。

---

## D. 原型描述子：用数据选，不要拍脑袋

旧描述子 $\vq_g=R([\vmu_g;\log\boldsymbol\sigma_g])\in\mathbb R^{12}$ 建立在估算 wrench 上，随该信号一起作废。

**候选特征**（每个都能由记录数据用固定公式算出）：

| 特征 | 公式 | 物理含义 |
|---|---|---|
| $\mu_{\text{grip}}$ | 组内 grip 均值 | 该用多大力 |
| $\sigma_{\text{grip}}$ | 组内 grip 总体标准差 | 力的波动 |
| **刚度** | 闭合段 $dF/d(\text{开度})$ 的组内中位数 | **全篇的核心物理量，也是空盒/满盒的本质差异** |
| 接触面积 | 超过阈值的 taxel 个数（阈值要定并记录） | 软硬的直接代理：硬物点接触、软物面接触 |
| 压心偏移 | $\sum v_i y_i / \sum v_i$ | 切向利用率的代理（可选） |

沿用原来的稳健归一化（组间 median/IQR），特征数控制在 4–6 个。

**选择流程**（标签重建后做，约半小时）：

1. 每个候选特征算**组间方差 / 组内方差**，去掉不能区分组的；
2. **留一 episode 重算描述子**，看聚类分配是否稳定，不稳定的说明是噪声；
3. 看聚类是否**把外观无关但物理相似的组放到一起**（例如满盒抓取与土豆抓取），还是仅仅按力大小排序。

把"只用 $[\mu,\log\sigma]$"和"加上刚度、接触面积"两套描述子在这三个指标上对比，用结果决定。
如果三项都显示新描述子只是力的分箱，就如实承认原型分支冗余，交给"仅原型"消融实证回答。

---

## E. Task 2（纸盒）分组

1. **不能靠 prompt 识别条件**：空盒和满盒指令完全相同，`fruit_from_prompt()` 直接失效。
   采集时就要把条件写进数据（建议每帧 `group_condition ∈ {box_empty, box_filled, box_half}`）。
2. group key 从 `(task, fruit, stage)` 泛化为 `(task, condition, stage)`，涉及
   `add_group_prototype_labels.py` 和 `build_safe_group_prototypes.py`（**两份副本**：post-process 目录和
   `examples/force/` 各一份）里的 `FRUITS`、`GROUP_KEYS`、`group_fruit`。
3. 合并后 **16 组**（8 个条件 = 果蔬 6 类 + 纸盒 2 条件，各 2 个阶段组 grasp/hold，见 0b 节），
   K 按 0b 节重扫 4–10 后确定（本地 12 组时选出 K=4），更新
   `train_config_*.conf` 的 `EXPECTED_PROTOTYPE_GROUPS`。
4. 半盒只做评估：不进示教、不进 prototype 表、不进组统计。

---

## F. split-first：先划分，再建标签表

**现状：完全没有划分，全量训练。** `scripts/train.py` 没有验证逻辑，`data_loader.py` 直接把整个数据集拿去训。

**为什么必须做**：选不了 checkpoint；发现不了物理分支的记忆行为；论文消融表的"分布误差"那一列需要 held-out episode 才有意义；data 一节已经写明目标只在训练 episode 上拟合。

**划分原则**：每个 condition（果蔬 6 类 + 纸盒 2 条件）留 1–2 条，总量 15–20%，按物体实例和采集时段分层。

```bash
# 1) 建标签表：统计只用训练 episode，验证 episode 另写 sidecar
uv run examples/force/build_safe_group_prototypes.py \
    --data-dir <合并后的 post-process 目录> \
    --val-episodes <val_ep_1> --val-episodes <val_ep_2> ... \
    --k <重选后的 K> --tau-q 0.1 \
    --output-dir prototype_metadata_trainonly --write

# 2) 转训练集   3) 转验证集（两次，--episodes 指定各自的名单，--labels-dir 指向上面的输出）
# 4) norm stats 只在训练集上算
uv run scripts/compute_norm_stats.py --config-name <train config>
```

**需要新增：离线评估脚本**（`examples/force/eval_draftvla_offline.py`，约几十行）
输入 checkpoint + 验证集 repo_id + config；调 `model.compute_train_losses` 只取指标不反传；
输出动作 MSE、分布误差、`proto_acc`、`loss_dist_baseline`、`g_phy_rel`，**按阶段分别报告**；
训练中每 500–1000 步跑一次，**按验证指标选 checkpoint**，曲线放论文附录。

---

## G. 标定的线性度决定要不要加单调映射

滑落起始法给出 $S_{\text{slip}}$ 与 $mg$ 的关系（5–6 个质量，每个 3 次）。看线性度：

- 线性（$R^2$ 高）→ 直接用电压单位训练，牛顿换算事后补或不补；
- 明显非线性（触觉阵列大力端常见饱和）→ **在建标签之前**把拟合出的单调映射套到 grip 上，否则
  高力端的 $\sigma$ 被压缩，分布形状和真实力空间不一致。

安全区间和违规率必须和标签在同一个空间里。

---

## H. 验收标准

- `uv run pytest src/openpi/models/physical_test.py src/openpi/models/pi0_test.py -q` 全绿；
- 转换后抽查若干帧：输入第 51 维等于 `state[6]`（都是实测开度），feature shape 是 57；
- 未接触帧的 grip 标量接近 0（归零生效），抓取帧显著为正；
- `gt_safe_distribution` 是 2 个数，且组内恒定；
- norm stats 里 `force` 是 57 维；
- `prototype_metadata_trainonly/` 的 `included_episodes` 不含任何验证 episode；
- 训练集帧数 + 验证集帧数 = 合并后总帧数（减 denylist）；
- 200 步 smoke 跑通，`loss_dist` 下降、`g_phy_rel` 非零、`proto_entropy` 不卡在 log K。

---

## I. 暂不处理，但要记住

- **grasp 阶段样本少**：果蔬批次里只占 3.9%，纸盒采集已改为分段闭合来缓解；必要时给 `L_dist` 按阶段加权。
- **消融分支还没有代码开关**：no-guidance（去掉 `G_phy`）、vision--language token（**必须取 FVLMoE 融合之前的 VLM 前缀 token**）、oracle-group。architecture-only、distribution-only、prototype-only 用现有的 `LAMBDA_DIST` / `LAMBDA_PROTO` 即可。
- 论文里的"物理语义"定位调整见 `-ICLR2027-DraftVLA/notes/physical_semantics_framing.md`，等纸盒实验跑通再议。
