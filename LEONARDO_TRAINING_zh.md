# DamageVLA 在 Leonardo 上训练

## 推荐环境

本项目使用仓库锁定的 `uv` + `.venv`。不需要 Conda；当前登录节点也没有 Docker，集群提供的
Singularity 会给这次训练增加镜像构建与挂载步骤，因此暂不采用。环境和大型缓存都放在项目工作盘，
不会占用用户 home 的小配额。

## 第一次安装

在登录节点执行（不需要申请 GPU）：

```bash
cd /leonardo_work/IscrC_VLA/DamageVLA/openpi_draftvla
bash setup_leonardo.sh
```

脚本会加载 `python/3.11.7`，安装 `uv`，按 `uv.lock` 创建 `.venv` 和其中锁定的 CUDA 12
运行库，然后校验 26 条 episode、20,344 帧和所有训练图像。不要额外加载集群 CUDA module，
否则它可能通过 `LD_LIBRARY_PATH` 覆盖 JAX 随 pip 安装的 CUDA/cuDNN 库。

## 先跑 20-step gate

最省排队资源的单卡功能检查：

```bash
bash submit_leonardo.sh smoke
squeue -u "$USER"
```

如需保留多卡兼容性，也可以额外检查四卡数据并行和 GPU 间通信（这不是正式训练必需步骤）：

```bash
bash submit_leonardo.sh smoke4
squeue -u "$USER"
```

作业号返回后查看输出：

```bash
tail -f slurm-draftvla-<job_id>.out
```

首次作业会把 22 条成功 episode 转成 `draftvla/fruits_tactile`。模型的 12 维力输入是每帧左右指
`tactile_estimated_wrenches` 的逐维平均 6D 加有符号半差分 6D，即
`concat(0.5*(left+right), 0.5*(left-right))`，不再使用 UR 法兰 `force_torque`。安全分布标签和
模型输出仍是平均 wrench 的 `[mu×6, sigma×6]` 共 12 维。作业还会下载
`pi0_base`，重新计算当前数据的 state/action/force 归一化统计，然后完成模型加载、JIT、20 步训练
和 checkpoint 写入。4 条明显失败 episode 会从所有 loss 中排除。

## 启动单卡 10k-step 正式训练

确认 smoke 作业结束且 loss 有限后执行：

```bash
bash submit_leonardo.sh full
```

正式任务会在一个节点上申请 1 张 A100、16 个 CPU 核和 120 GB 内存。训练配置在
`train_config.conf`，单卡处理完整的 global batch 16；`FSDP_DEVICES=1` 对应
`(data=1, fsdp=1)`，不启用模型切分。保持 10,000 steps。日志、CSV、TensorBoard 和 checkpoint
分别写入 `examples/force/` 与 `checkpoints/pi0_draftvla/draftvla_full/`。

如果只改训练超参数并复用同一转换数据，可直接提交；如果原始 episode 或转换逻辑变了，使用：

```bash
FORCE_RECONVERT=1 bash submit_leonardo.sh smoke
```

只有确认 norm stats 已对应当前 `draftvla/fruits_tactile` 时，才可在提交 full 前设置
`SKIP_NORM_STATS=1`；原法兰 wrench 数据集的统计不能复用。
