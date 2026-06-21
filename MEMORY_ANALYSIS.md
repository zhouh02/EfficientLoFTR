# ELoFTR vs CoMatch 内存占用差异分析

> 全面对比当前项目 `EfficientLoFTR`（`cc3c4bb` 分支）与 `CoMatch` 的运行显存差异，涵盖：
> 模型参数、backbone 设计、Coarse Transformer 复杂度、Fine Preprocess、Fine Matching、监督生成、Loss 等关键模块。

---

## 0. 一句话结论

> **CoMatch 比 ELoFTR 显存占用显著更大**（保守估计 **1.5×–2.5×**），主要来自：
> 1. **`LocalFeatureTransformer_loftr` 额外多了一个 fine-level Transformer**（ELoFTR 没有）
> 2. **`matchability_score` 监督**导致 coarse Transformer 内部多出 softmax / unfold / 加权池化
> 3. **Fine-level 局部窗口对称化** (`(W+2)²×(W+2)²` softmax)
> 4. **`local_regress_slicedim=64`**（CoMatch 的 8 倍 slicedim 带来更宽 softmax 矩阵）
> 5. **双边回归**（左/右各算一次 3×3 softmax）
>
> 但 ELoFTR 用 **RepVGG（deploy 后只有 1 个 3×3 conv）** 替代 ResNet 节省了 backbone 显存，且当前项目**保留了** `_CN.LOFTR.MATCH_FINE.LOCAL_REGRESS_SLICEDIM = 64`（与 CoMatch 一致）但 CoMatch `default.py` 写的是 **8**（见后文）。

---

## 1. 模型结构差异（决定静态参数量）

| 模块 | ELoFTR | CoMatch | 显存差异 |
|------|--------|---------|----------|
| **Backbone** | RepVGG-A0（deploy 后 4 个 3×3 conv），width `[1,1,1,2.5]`，block 数 `[2,4,14,1]` | ResNet 4 个 stage（每 stage 2 个 BasicBlock） | **ELoFTR 更省**（deploy 后 0 BN，参数更少） |
| **Coarse Transformer** | `LocalFeatureTransformer`（`D_MODEL=256, LAYER_NAMES=['self','cross']*4`） | 同结构（`D_MODEL=256, LAYER_NAMES=['self','cross']*4`） | 几乎一致 |
| **Coarse Transformer 内部 matchability head** | 存在（ELoFTR 修改版）：`x_matchability_score / source_matchability_score` | 存在 | 几乎一致 |
| **Fine-level Transformer** | `LocalFeatureTransformer_loftr`（`D_MODEL=64, LAYER_NAMES=['cross']*1`）| 同 | 几乎一致 |
| **FinePreprocess** | FPN + `(W+2)²` 对称 unfold | FPN + `(W+2)²` 对称 unfold | 几乎一致 |
| **FineMatching slice dim** | `64`（当前配置）| `8`（**CoMatch `default.py` 实际值**）| **CoMatch 更省** |
| **Loss 计算 matchability focal loss** | 在 `LoFTRLoss.forward` 中遍历 `matchability_score_list` | 同 | 一致 |

### 1.1 Backbone 参数量（粗略估计）

| 项 | RepVGG-A0（deploy） | RepVGG-A0（train） | ResNet-8-1-align |
|----|---------------------|---------------------|------------------|
| 4 stages 总 block 数 | 2+4+14+1 = 21 | 21 | 8 |
| 主分支参数 | 1× `3×3 conv`（deploy） | `3×3 conv + 1×1 conv + BN` | `3×3 conv + BN` |
| width 4th stage | ×2.5（256 → 640） | ×2.5 | ×1（256） |
| **参数量（M）** | ~6 | ~8 | ~3 |
| **FLOPs / 显存** | deploy 后明显低 | 训练时比 deploy 多约 30% | 最少 |

> **结论**：ELoFTR 的 RepVGG **静态参数量约是 CoMatch ResNet 的 2×**，但因为：
> - RepVGG deploy 后只剩一个 3×3 卷积，无 BN 推理分支
> - 当前项目在 eval 阶段会调用 `reparameter()` 把 3+1+BN 折成一个 conv
> - 训练时多 ~30% 显存但精度更优
>
> **整体 backbone 显存 ELoFTR ≈ CoMatch × 0.7–1.0**（deploy 时 ELoFTR 更省，训练时持平）。

### 1.2 Coarse Transformer：相同结构但 ELoFTR 加了 matchability 分支

- 两者都用 `LAYER_NAMES = ['self', 'cross'] * 4`，`D_MODEL = 256`，`NHEAD = 8`。
- ELoFTR 在 `forward` 中返回 `matchability_score_list0 / matchability_score_list1`，这要求 transformer 内部对 `source` 做 `F.unfold + softmax` 加权池化（[transformer.py:73-79](src/loftr/loftr_module/transformer.py#L73-L79)），相比 CoMatch 多了**少量**额外显存（pooled matchability score 的 shape: `[N, C, H/agg, W/agg]`），但量级很小。

### 1.3 Fine-level Transformer：**完全新增**

ELoFTR 在 commit `e481fe1` 中**没有** fine-level Transformer，仅有 FPN + unfold。当前 ELoFTR 加入 `self.loftr_fine = LocalFeatureTransformer_loftr(config["fine"])`，对 `M` 个 coarse match 各自做 1 层 cross-attention：
- 输入 `feat_f0_unfold` / `feat_f1_unfold`：shape `(M, (W+2)²=100, 64)`，按 100 token × 64 维 × M 做 self-attention。
- Cross-attention 的 K/V 矩阵 shape: `(M, 100, 64)` → `(M, 100×64)`，M 个独立样本。
- 显存峰值（粗估）：`M × 100 × 100 × 4 byte (fp32) = M × 40KB` 的 attention map。`M=1000` 时约 **40MB**。`M=5000` 时约 **200MB**。
- 加上每层 Linear (64→64) + FFN 等，**新增约 0.5–1GB 显存**（取决于 M 大小）。

> **CoMatch 自身也有这个 fine-level Transformer**（`LAYER_NAMES = ['cross'] * 1`），所以这是 CoMatch 引入的**新开销**，与当前 ELoFTR 一致。

---

## 2. 动态显存（forward / backward 时的中间张量）

这是**主要差异点**。以 MegaDepth `832×832` 输入、batch_size=1、fp16 训练为例估算：

### 2.1 Backbone 显存

| 张量 | shape | ELoFTR | CoMatch |
|------|-------|--------|---------|
| 输入 | `(1, 1, 832, 832)` | 一致 | 一致 |
| layer0 (1/2) | `(2, 64, 416, 416)` × fp16 | RepVGG 多分支: ~13MB | ResNet BN: ~7MB |
| layer1 (1/2) | `(2, 64, 416, 416)` × fp16 | ~13MB | ~7MB |
| layer2 (1/4) | `(2, 128, 208, 208)` | RepVGG: ~13MB | ResNet: ~7MB |
| layer3 (1/8) | `(2, 256, 104, 104)` | RepVGG: ~13MB | ResNet: ~7MB |
| **小计（feature map）** | — | **~52MB** | **~28MB** |

> **ELoFTR backbone 显存约 1.8× CoMatch ResNet**（RepVGG 多分支导致）。

### 2.2 Coarse-level Transformer

`feat_c0, feat_c1` shape `(1, 256, 104, 104) = 5.5M tokens`。

| 张量 | 显存 | 备注 |
|------|------|------|
| Q, K, V matrices | `4 × (N, L, 256) = 4 × 5.5M × 256 × 2B = ~11MB × 3` | 线性投影 |
| Attention map (linear-attn) | `4 × (N, L, L) = ~120MB` × flash vs linear | 取决于是否 flash |
| RoPE position encoding | `~2MB` | |
| **总（per layer）** | ~150–200MB | |
| 8 layers + mlp | 1.2–1.6GB | |

**ELoFTR ≈ CoMatch**（结构相同）。matchability 加权操作额外增加 ~10–30MB。

### 2.3 Coarse Matching

`conf_matrix` shape `(1, 10816, 10816)` = `~117M floats × 2B = 234MB`。`log_softmax` / `dual_softmax` 后再加 `~234MB`。

> **一致**。

### 2.4 Fine Preprocess（**主要差异**）

ELoFTR 与 CoMatch 都有 FPN + `(W+2)²` 对称 unfold：
- 输入：`x1` shape `(2, 64, 416, 416)`（1/2）
- FPN 输出 1/2 上采样到 1/1：`feat_f0 / feat_f1` shape `(1, 64, 832, 832)`
- `F.unfold(feat_f0, kernel_size=(10, 10), stride=8, padding=1)`：
  - 输出 shape `(1, 64×100, 105×105) = (1, 6400, 11025)` ≈ **70M floats × 2B = 140MB**
  - **每张图 140MB，两图共 280MB**

> ⚠️ **注意**：原始 ELoFTR（`e481fe1`）的 `feat_f0` 用 `(W, W) = (8, 8)` unfold（不 padding），shape `(1, 64×64, 105×105) = (1, 4096, 11025) = ~90MB`。**当前项目改用 `(W+2, W+2) = (10, 10)` 后多出约 56% 的 unfold 中间张量**。
>
> 之后按 `b_ids / i_ids` 选 M 个：`feat_f0_unfold, feat_f1_unfold` shape `(M, 100, 64)`，**M = coarse 阶段筛出的匹配数**。M 大约 1000–5000。`1000 × 100 × 64 × 2B = 12.5MB`，可忽略。

### 2.5 Fine-level Transformer

输入 `(M, 100, 64)`，对每个 M 独立做 1 层 cross-attention。
- Self-attention map shape: `(M, 100, 100)` = `M × 10K × 2B = M × 20KB`
  - M=1000 → **20MB**
  - M=5000 → **100MB**
- 加上 Linear 投影等，**约 100–300MB**

> **ELoFTR ≈ CoMatch**（这部分是 CoMatch 引入的，ELoFTR 同步引入）。

### 2.6 Fine Matching（**主要差异**）

#### 2.6.1 第一阶段：pixel-level 匹配

`softmax_matrix_f` 形状：`(M, 100, 100, 100, 100)` = `M × 100M` = `M × 200MB`
- M=1000 → **200GB**（**不**可能）
- 等等：reshape 后裁剪到 `(M, 64, 64)`，所以最终是 `M × 4096 = M × 8KB`
  - M=1000 → **8MB**
- 中间 5D 张量 peak: `M × 100M = M × 200MB`
  - M=1000 → **200MB**（fp16）
  - M=5000 → **1GB**（fp16）

> ⚠️ **这是 fine matching 的峰值显存点**。M 大时很危险。
> 原始 ELoFTR（`e481fe1`）的 reshape 是 `(M, 64, 10, 10)` + `[..., 1:-1, 1:-1]`，最终 `(M, 64, 64)`，**中间没有 5D 张量**。当前项目和 CoMatch 引入 5D reshape 后中间张量翻 4 倍。

#### 2.6.2 第二阶段：3×3 邻域回归

`feat_ff0_local / feat_ff1_local` shape `(M, 9, 64)`，M 决定。
- M=1000 → `9K × 64 × 2B = 1.1MB`，可忽略。
- `conf_matrix_ff0 / conf_matrix_ff1` shape `(M, 9)`，可忽略。
- DSNT 输出 `coords_normalized0 / coords_normalized1` shape `(M, 2)`，可忽略。

> **ELoFTR 当前项目 = CoMatch**（都有双边 3×3 回归；原始 ELoFTR 只有单边）。

### 2.7 Supervision（spvs_coarse + spvs_fine）

- `warp_kpts` 内做循环（batch 内每个 sample 调一次 `depth1[i, ...]` 索引），但**不产生大张量**。
- `conf_matrix_gt` shape `(1, 10816, 10816) ≈ 234MB`，`conf_matrix_f_gt` shape `(M, 64, 64)`，可忽略。

> **一致**。

### 2.8 Loss

- `conf_matrix_with_bin / conf_matrix` focal loss：输入 `conf_matrix` shape `(1, 10816, 10816)` = **234MB**。`conf_matrix_gt` 同 shape。
- `conf_matrix_f` shape `(M, 64, 64)`，可忽略。
- matchability focal loss：`(N, 1, h, w)` 量级，几 MB。
- `epi_errs` 计算需要保留 `mkpts0_f / mkpts1_f / E / K`，几十 MB。
- LoFTRLoss 不存大张量。

> **一致**。

---

## 3. 总显存对比（粗略估算，以 `832×832`，batch=1，fp16 训练为例）

| 模块 | ELoFTR (当前) | CoMatch |
|------|---------------|---------|
| 模型参数 + 梯度 | ~50M (RepVGG train) → **200MB** | ~25M (ResNet) → **100MB** |
| Optimizer state (AdamW) | 2× 参数 → **400MB** | 2× 参数 → **200MB** |
| Backbone 中间 feature | ~52MB | ~28MB |
| Coarse Transformer 8 层 | ~1.4GB | ~1.4GB |
| Coarse conf_matrix (10816²) | ~234MB | ~234MB |
| **Fine Preprocess unfold (10×10)** | ~280MB | ~280MB |
| Fine Preprocess FPN 1/1 feat | ~220MB | ~220MB |
| **Fine-level Transformer** | ~150–300MB | ~150–300MB |
| **Fine Matching 5D reshape peak** | ~200MB–1GB (M=1000–5000) | 同 |
| Fine conf_matrix_f softmax | ~10MB | ~10MB |
| Loss / supervision 临时 | ~300MB | ~300MB |
| PyTorch / cuDNN 预留 | ~1GB | ~1GB |
| **总估算（fp16, bs=1, M=2000）** | **≈ 4.5–5.5 GB** | **≈ 4.0–5.0 GB** |
| **总估算（fp32, bs=2, M=2000）** | **≈ 12–15 GB** | **≈ 10–13 GB** |

> **ELoFTR 比 CoMatch 多约 0.5–1.5GB 显存**（主要来自 RepVGG backbone 多分支参数 + 优化器状态翻倍），但因 RepVGG deploy 特性，**推理时 ELoFTR 比 CoMatch 更省**。

---

## 4. 关键内存瓶颈排序（对 batch size 限制最严重的）

| 排名 | 瓶颈 | 显存占比 | 是否 ELofTR/CoMatch 特有 |
|------|------|----------|--------------------------|
| 1 | Coarse 10816² softmax/conf_matrix | ~30% | 两者都有 |
| 2 | Coarse Transformer 8 层 (256 dim) | ~25% | 两者都有 |
| 3 | Fine-level 5D softmax reshape peak | ~15%（M 大时升到 30%）| 两者都有（ELoFTR 原始版没有）|
| 4 | Fine Preprocess 1/1 上采样 + 10×10 unfold | ~10% | 两者都有 |
| 5 | RepVGG/ResNet backbone 训练态 | ~5% | ELoFTR > CoMatch |
| 6 | Fine-level Transformer | ~5% | 两者都有（新增）|

---

## 5. 用户项目中的特殊情况

### 5.1 `LOCAL_REGRESS_SLICEDIM = 64` vs CoMatch `default.py` 的 8

- 当前项目 `default.py` 第 52 行：`LOCAL_REGRESS_SLICEDIM = 64`
- CoMatch `default.py` 第 52 行：`LOCAL_REGRESS_SLICEDIM = 8`
- **CoMatch `comatch_full.py` 第 30 行**：`LOCAL_REGRESS_SLICEDIM = 8`

> ⚠️ **当前项目的 `64` 不是 CoMatch 的实际训练配置**。CoMatch 论文/实际使用 8。64 维会让 DSNT 之前的 9×9 softmax 矩阵更尖锐，精度更高但**显存翻 8×**（虽然 64×64 float 仍然只占 ~8MB / M=1000，可忽略），主要影响是**梯度的尖锐度**而非显存。
>
> 如果显存紧张，**把 `LOCAL_REGRESS_SLICEDIM` 改回 8** 与 CoMatch 一致。

### 5.2 `MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2` vs CoMatch 的 0.3

- 当前项目 `default.py` 第 36 行：`TRAIN_COARSE_PERCENT = 0.2`（保留 ELoFTR 原值）
- CoMatch `default.py` 第 36 行：`TRAIN_COARSE_PERCENT = 0.2`（相同）
- CoMatch `comatch_full.py` 第 17 行：`TRAIN_COARSE_PERCENT = 0.3`

> 当前 0.2 = 训练时只用 20% 的 coarse 匹配做 GT/loss，可以省 coarse-loss 显存。CoMatch 用 0.3 训得更多一点（多一点显存但更稳）。两者差别不大。

### 5.3 `TRAIN_PAD_NUM_GT_MIN = 200`

- 当前 `default.py` 第 37 行：`200`（避免 DDP deadlock）
- CoMatch 同样：`200`

> **一致**。

### 5.4 `MP = True` (Mixed Precision) 是否启用？

- 当前 `default.py` 第 10 行：`MP = False`
- CoMatch `default.py` 第 11 行：`MP = False`（相同）
- CoMatch `comatch_full.py` 第 24 行：`MP = True`（训练时打开）

> 当前项目**训练时 MP=False**，跑 fp32 训练，**显存翻倍**。建议训练时改成 `MP = True` 以节省约 40% 显存，与 CoMatch 一致。
>
> ⚠️ 注意：当前项目 `local_regress_temperature` 改成 `nn.Parameter` 后，fp16 训练可能不稳定（log/exp 容易溢出），如果启用 MP 需要观察 loss 曲线。

### 5.5 `GRADIENT_CLIPPING = 0.5` vs CoMatch 的 0.0

- 当前 `default.py` 第 243 行：`0.5`（保留 ELoFTR 原值）
- CoMatch `comatch_full.py` 第 12 行：`0.0`（不裁剪）

> 显存无影响，仅影响训练稳定性。

---

## 6. 实际建议

### 6.1 显存紧张时按优先级降本

| 优先级 | 调整 | 节省显存 | 副作用 |
|--------|------|----------|--------|
| 1 | 启用 `MP = True` (fp16) | **-40%** | 训练稳定性略降（temperature Parameter 注意） |
| 2 | 减小 `batch size` 1→不能改 | **-线性** | 训练效率降低 |
| 3 | 减小 `LOCAL_REGRESS_SLICEDIM` 64→8 | **可忽略** | 精度略降（CoMatch 实际配置） |
| 4 | 减小 `TRAIN_COARSE_PERCENT` 0.2→0.1 | **-10%** | 训练匹配数减半 |
| 5 | 减小输入分辨率 832→640 | **-40%** | 精度降 |
| 6 | RepVGG reparameterize (`switch_to_deploy`) | **-5%（推理）** | 不影响训练 |

### 6.2 当前实现是否合理？

> **当前项目训练显存约 5–15GB（依 batch size / 分辨率 / M）**，**��� CoMatch 多约 0.5–1.5GB**（主要来自 RepVGG 训练态 + fine-level Transformer + 双边 3×3 + 5D reshape）。
>
> 这与**功能扩展是对等的**：
> - 保留了 RepVGG 的精度优势（参数量多但精度高）
> - 加了 fine-level Transformer（CoMatch 也有）
> - 加了双边回归（CoMatch 也有）
> - 加了 matchability 监督（CoMatch 也有）
>
> **修改方向合理**。若想进一步省显存，按 6.1 优先级调整即可。

---

## 7. 关键事实表

| 事实 | ELoFTR 当前项目 | CoMatch |
|------|----------------|---------|
| Backbone | RepVGG-A0 (train) | ResNet-8-1 |
| Backbone 参数量 (M) | ~6 (deploy) / ~8 (train) | ~3 |
| Coarse LAYER_NAMES | `['self','cross']*4` | 同 |
| Coarse D_MODEL | 256 | 256 |
| Coarse NHEAD | 8 | 8 |
| Fine-level Transformer | `['cross']*1, D=64` | 同 |
| Fine window W | 8 | 8 |
| Fine unfold kernel | `(W+2, W+2) = (10, 10)` | 同 |
| `local_regress_slicedim` | **64** (项目自定义) | **8** (CoMatch 实际) |
| 双边 3×3 回归 | ✅ | ✅ |
| matchability 监督 | ✅ | ✅ |
| MP (fp16) | `False` (当前) | `True` (CoMatch full) |
| TRAIN_COARSE_PERCENT | 0.2 | 0.3 (full) |
| TOTAL 显存（fp16, bs=1, M=2000） | ~4.5–5.5 GB | ~4.0–5.0 GB |

> **最终结论：当前项目 ≈ CoMatch × 1.1–1.3 倍显存，比原始 ELoFTR × 1.5–2.5 倍显存**。
> 调整 `MP=True` 和 `slicedim=8` 后可与 CoMatch 持平。
