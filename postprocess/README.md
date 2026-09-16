# 后处理流水线（从原始采集到可转换的 observations.jsonl）

这些脚本原本只存在于各批数据目录里（`DamageVLA_training_post_process_*/`），
不在任何仓库里。但 `outlines/todo_training_contract.md` 里要改的东西有一半在这里，
所以收进仓库做版本管理。**脚本本身不含数据。**

| 文件 | 作用 |
|---|---|
| `postprocess_20260916_carton.py`（724 行，**当前流水线**） | 一条龙：有序宏阶段标注、法兰与触觉的阶段/未来块统计、组级安全分布与软原型、**逐 episode 的法兰力归零**（`force_torque_zeroed`）。纸盒批次还从 `episode_context` 读取 `load_condition` / `object_id` |
| `postprocess_20260911.py` | 同一套流水线的果蔬版本 |
| `apply_image_crop_*.py` | 1920×1080 中心裁剪到 1080×1080 再缩放 224 |
| `prepare_*_review.py` | 生成人工复查用的素材 |
| `validate_raw_*.py` / `validate_processed_*.py` | 采集后与处理后的校验 |

## 按新契约要改的地方（详见 `outlines/todo_training_contract.md`）

1. **分组**：group key 从 `(task, fruit, stage)` 改成 `(task, condition, stage)`。
   纸盒的 condition 用 `group_load_condition`（已有），水果用 `group_fruit`。
2. **阶段**：标注保留四个接触阶段，但**分布目标按 `grasp` / `hold`（lift+translate+place）两组统计**。
3. **目标降维**：`gt_safe_distribution` 从 12 个数（6 维 wrench 的 mu、sigma）改成 **2 个数**——
   标量 grip（两指归零 `_data` 电压总和的均值）的 mu 与 sigma。
4. **sigma 下限**：钳制到不低于 0.0051。
5. **排除规则**：3×MAD 离群、接触 taxel 少于 3 个、持握力近噪声（后两条对空纸盒豁免）。
6. **指尖归零**：目前只有法兰做了逐 episode 归零，指尖 `_data` 没有。
   要么在这里补一个 `tactile_data_zeroed` 字段，要么在转换脚本里做（二选一，别做两遍）。
7. **split-first**：组统计只能用训练 episode 拟合，验证 episode 走 sidecar 标签。

## 注意

`build_safe_group_prototypes.py` 有两份副本：一份在 `examples/force/`（已在仓库里），
一份历史上在 post-process 目录。两者的分组定义必须一致，改一处必须同步另一处。
