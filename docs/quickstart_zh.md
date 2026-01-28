## VeOmni 快速上手（多模态训练导向）

这份文档的目标是：让你在 **1 小时内理解 VeOmni 的整体框架**，并能用仓库自带的训练入口把你的多模态训练跑起来（SFT / 继续预训练都适用）。

> 说明：VeOmni 的训练脚本强调 **trainer-free**，即训练逻辑写在脚本里（线性、可读、可改），而不是封装成一个“黑盒 Trainer”。

---

## 项目结构（你最需要看的目录）

- **`configs/`**：训练配置（YAML），按任务/模型组织
  - `configs/pretrain/`：纯文本预训练
  - `configs/sft/`：纯文本 SFT
  - `configs/multimodal/`：多模态/omni（VL/Omni/decoder/data 等）
  - `configs/dit/`：diffusion/DIT 相关任务
- **`tasks/`**：训练入口脚本（你真正运行的 python 文件）
  - `tasks/train_torch.py`：纯文本（plaintext / conversation）训练通用入口
  - `tasks/omni/train_omni_model.py`：通用 omni 多模态训练入口（encoder/decoder/foundation 组合）
  - `tasks/omni/train_qwen_vl.py` 等：特定模型的任务脚本（通常是在通用入口上做了少量定制）
- **`veomni/`**：核心库
  - `veomni/arguments/`：统一参数系统（YAML + CLI 覆盖 + dataclass 校验）
  - `veomni/data/`：dataset / dataloader / collator / transform（含多模态）
  - `veomni/models/`：HF 模型加载、omni 模型组装、权重保存
  - `veomni/distributed/`：FSDP1/FSDP2、sequence parallel(Ulysses)、expert parallel、梯度裁剪等
  - `veomni/checkpoint/`：Torch Distributed Checkpoint(DCP) 存取与转换
- **`train.sh`**：torchrun 启动脚本（自动判断 GPU/NPU、自动设置 `--nproc-per-node`）

---

## 核心理念：Trainer-free + 配置驱动 + 组合式并行

### 训练逻辑为什么“好改”

VeOmni 的训练主流程基本都在 `tasks/*.py` 里按顺序写出来：

- 初始化分布式（`torch.distributed.init_process_group`）
- 解析配置（YAML/CLI → dataclass）
- 初始化并行状态（device mesh）
- 构建数据集与 dataloader
- 构建模型 → 并行包装（FSDP/EP/SP）→ optimizer → scheduler
- 进入训练循环（forward/backward/clip/step/log/save）

你想改 loss、改 batch 组成方式、加 metric、改 checkpoint 频率，直接改脚本即可。

### 配置系统如何工作（非常关键）

VeOmni 的 `parse_args(...)` 规则是：

- **第一个位置参数**是 YAML 配置文件路径：`configs/**.yaml`
- 其余参数用 `--train.xxx / --data.xxx / --model.xxx` 这种**点号路径**覆盖 YAML
- 最终会递归实例化 dataclass，并触发各层 `__post_init__` 做校验与默认值补齐

示例（推荐背下来）：

```bash
bash train.sh tasks/omni/train_omni_model.py configs/multimodal/omni/seed_omni.yaml \
  --model.model_path /path/to/foundation_model \
  --data.train_path /path/to/train.json \
  --train.output_dir /path/to/output
```

---

## 两条训练入口：纯文本 vs 多模态（你该选哪个）

### A) 纯文本（预训练 / SFT）

用 `tasks/train_torch.py`：

- `data.data_type: plaintext` → `process_pretrain_example(...)`
- `data.data_type: conversation` → `process_sft_example(...)` + chat template

启动方式：

```bash
bash train.sh tasks/train_torch.py configs/sft/qwen3_sft.yaml \
  --model.model_path /path/to/hf_model \
  --data.train_path /path/to/train.parquet \
  --train.data_parallel_mode fsdp2 \
  --train.init_device meta
```

### B) 多模态 / Omni（图文/多模态输入输出）

用 `tasks/omni/train_omni_model.py` 或某个 `tasks/omni/train_*.py`：

- `build_omni_model(...)` 组装 foundation + encoder/decoder（取决于配置）
- `build_omni_processor(...)` 负责 tokenizer + 图像等模态的预处理
- transform 通常走 `encode_multimodal_sample(...)`
- collator 根据 `rmpad_with_pos_ids` / sequence parallel 等追加额外处理

最小可运行启动方式（以仓库自带 `configs/multimodal/omni/seed_omni.yaml` 为例）：

```bash
bash train.sh tasks/omni/train_omni_model.py configs/multimodal/omni/seed_omni.yaml \
  --model.model_path /path/to/foundation_model \
  --data.train_path /path/to/train.json \
  --train.output_dir /path/to/output
```

如果你在做 Qwen3-VL 这类模型，更推荐直接用对应的示例脚本与配置（见 `docs/examples/qwen3_vl.md`）。

---

## 训练流程拆解（端到端）

下面这段可以当作“读代码索引”，基本对应 `tasks/train_torch.py` 和 `tasks/omni/train_omni_model.py` 的逻辑。

### 1) 分布式与设备

- 通过 `train.sh` 调 `torchrun`，自动设置进程数与一些环境变量
- 训练脚本里会 `dist.init_process_group(...)`
- 根据 `LOCAL_RANK` 设置当前 device

### 2) 初始化并行状态（DeviceMesh）

通过 `init_parallel_state(...)` 创建全局并行状态（DP/FSDP/EP/SP 等），后续数据切分、loss all-reduce、EP checkpoint 恢复等都依赖它。

常用参数：

- `train.data_parallel_mode`: `ddp | fsdp1 | fsdp2`
- `train.expert_parallel_size`: 开启 MoE expert parallel
- `train.ulysses_parallel_size`: 开启 Ulysses sequence parallel（长序列）

### 3) 数据：dataset → transform → dataloader

数据集构建入口：`veomni/data/dataset.py`

- `datasets_type: mapping`：适合小数据（有 `__len__`）
- `datasets_type: iterable`：适合大数据/流式（HF streaming），支持按 node 切分
- `train_path` 以 `.yaml` 结尾时会自动启用 multisource（interleave）

dataloader 构建入口：`veomni/data/data_loader.py`

关键点：

- VeOmni 的 dataloader 产出通常是 `micro_batches: List[Dict[str, Tensor]]`
- 每一步会对这组 micro-batch 进行累积反传（对应 gradient accumulation）
- `rmpad / rmpad_with_pos_ids + dyn_bsz` 会启用 packing + 动态 token batching（更省显存、更高吞吐）

### 4) 模型：加载与并行包装

模型构建入口：

- 纯文本/基础模型：`veomni/models/auto.py::build_foundation_model`
- 多模态/omni：`veomni/models/seed_omni/...`（由 `tasks/omni/*` 调用）

并行包装入口：

- `veomni/distributed/torch_parallelize.py::build_parallelize_model`
  - 支持 FSDP1 / FSDP2
  - 支持 EP（MoE）与 FSDP 组合
  - 支持（受限的）TP 开关（但当前参数里默认限制较多）

### 5) 训练循环（loss / backward / step）

典型一轮 step 的结构：

- 取 `micro_batches = next(iter(dataloader))`
- 对每个 `micro_batch`：
  - 把 tensor 移到 device
  - forward 得到 `loss`（多模态任务可能有 `losses` 字典）
  - backward（可能带 activation offload / gradient checkpointing）
- `clip_grad_norm`
- `optimizer.step()` / `scheduler.step()` / `optimizer.zero_grad()`
- 跨 DP group all-reduce loss/grad_norm（用于日志展示一致）

### 6) checkpoint 保存/恢复

默认 checkpoint manager：`train.ckpt_manager: dcp`

行为特点：

- 训练中按 `train.save_steps` 或 `train.save_epochs` 保存
- 默认存到：`train.output_dir/checkpoints/global_step_XXXX`
- 通过 `train.load_checkpoint_path` 恢复（同时恢复 dataloader/environ_meter/rng/lr_scheduler 等 extra state）
- 训练末尾可选把“最后一次 ckpt”导出成 HuggingFace 权重目录（`hf_ckpt/`）

---

## 最常用的上手命令模板（建议直接复制）

### 1) 单机多卡（GPU）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash train.sh tasks/omni/train_omni_model.py configs/multimodal/omni/seed_omni.yaml \
  --model.model_path /path/to/model \
  --data.train_path /path/to/train.json \
  --train.output_dir /path/to/output \
  --train.data_parallel_mode fsdp1
```

### 2) FSDP2 + meta init（更省初始化显存）

```bash
bash train.sh tasks/train_torch.py configs/sft/qwen3_sft.yaml \
  --model.model_path /path/to/model \
  --data.train_path /path/to/train.parquet \
  --train.data_parallel_mode fsdp2 \
  --train.init_device meta
```

### 3) 断点续训

```bash
bash train.sh tasks/omni/train_omni_model.py configs/multimodal/omni/seed_omni.yaml \
  --train.load_checkpoint_path /path/to/output/checkpoints/global_step_1000
```

---

## 快速排障清单（高频问题）

- **config_file 怎么传？**：YAML 是“位置参数”，放在脚本后面第一个；覆盖参数才用 `--train.xxx` 形式。
- **iterable 数据集不收敛/步数不对？**：给 `data.train_size` 或 `train.max_steps`，并确认 `args.train.compute_train_steps(...)` 被调用。
- **rmpad 报错/不兼容**：部分多模态模型禁用 `rmpad`，改用 `rmpad_with_pos_ids`（具体看对应任务脚本里的限制）。
- **磁盘 I/O 慢**：考虑 `train.broadcast_model_weights_from_rank0=true`（默认就是 true）减少多 rank 重复读权重。
- **长序列**：优先看 `docs/key_features/ulysses.md`，并设置 `--train.ulysses_parallel_size N`。

---

## 进一步阅读（仓库内最推荐的文档）

- `docs/usage/basic_modules.md`：框架核心模块总览（dataset/dataloader/model/parallel/checkpoint）
- `docs/examples/qwen3.md`：纯文本 SFT 的可运行例子
- `docs/examples/qwen3_vl.md`：多模态 VL 的可运行例子
- `docs/key_features/ulysses.md`：长序列 Ulysses sequence parallel

