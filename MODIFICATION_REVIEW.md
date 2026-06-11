# EfficientLoFTR vs CoMatch 第二阶段细化 — 修改审查报告

> 本报告对比 `EfficientLoFTR` 当前实现（`cc3c4bb` 分支 `exp_dual_normal`）与
> 原始 `EfficientLoFTR`（commit `e481fe1`）以及参考实现 `CoMatch`（`src/loftr/utils/fine_matching_epipolar.py` 等）的差异，
> 审查"将 CoMatch 第二阶段细化模块替换 ELoFTR 内容"的修改是否合理、是否忠实于 CoMatch 的设计。

---

## 0. 总览

| 维度 | 原始 ELoFTR | CoMatch（参考） | 当前 ELoFTR（修改后） | 是否与 CoMatch 一致 |
|------|-------------|----------------|----------------------|--------------------|
| FinePreprocess patch 尺寸 | `feat_f0 = W×W`，`feat_f1 = (W+2)×(W+2)`（左右不对称） | `feat_f0 = feat_f1 = (W+2)×(W+2)`（左右对称） | `feat_f0 = feat_f1 = (W+2)×(W+2)`（已对齐） | ✅ |
| FineMatching WW 计算 | `W = sqrt(WW)` | `W = sqrt(WW) - 2` | `W = sqrt(WW) - 2` | ✅ |
| FineMatching softmax reshape | `reshape(M, WW, W+2, W+2)` + `[..., 1:-1, 1:-1]` | `reshape(M, W+2, W+2, W+2, W+2)` + `[..., 1:-1, 1:-1, 1:-1]` | `reshape(M, W+2, W+2, W+2, W+2)` + `[..., 1:-1, 1:-1, 1:-1]` | ✅ |
| 第二阶段回归方向 | 单边（仅右图） | 双边（左右各自 +3×3 softmax） | 双边（左右各自 +3×3 softmax） | ✅ |
| 第二阶段查询向量 | 直接对右图局部 3×3 算相似度 | 中心点均值 `(f0_center + f1_center) / 2` 作为 query，对左右图 3×3 各算一次 | 中心点均值 `(f0_center + f1_center) / 2` 作为 query，对左右图 3×3 各算一次 | ✅ |
| 第二阶段 mkpts 更新 | 仅 `mkpts1_f = mkpts1_c + offset` | `mkpts0_f = mkpts0_c + offset0`，`mkpts1_f = mkpts1_c + offset1` | `mkpts0_f = mkpts0_c + offset0`，`mkpts1_f = mkpts1_c + offset1` | ✅ |
| Loss 监督 | l2 (`_compute_local_loss_l2` on `expec_f` / `expec_f_gt`) | l2 (`_compute_local_loss_l2` on `expec_f` / `expec_f_gt`) | l2（结构与 CoMatch 一致） | ✅ |
| Supervision 路径 | `F.unfold(..., kernel_size=(W, W), padding=0)`，`WW = W*W` | `F.unfold(..., kernel_size=(W+2, W+2), padding=1)`，再取内部 `W*W` | `F.unfold(..., kernel_size=(W+2, W+2), padding=1)`，再取内部 `W*W` | ✅ |
| `local_regress_temperature` 来源 | 配置静态值 | `nn.parameter.Parameter(torch.tensor(10.), requires_grad=True)` | `nn.parameter.Parameter(torch.tensor(10.), requires_grad=True)` | ✅ |
| `local_regress_slicedim` | `8` | `64` | `64` | ✅ |
| overlap weight (coarse/fine) | `False / False` | `True / True` | `True / True` | ✅ |
| fine_window_size | `8` | `8` | `8` | ✅ |
| FINE_TYPE | `'l2_with_std'`（ELoFTR 自定义） | `'l2'` | `'l2'` | ✅ |
| 学习率 / warmup | `CANONICAL_LR=6e-3, WARMUP_RATIO=0., MSLR=[3,6,9,12]` | `8e-3, 0.1, [8,12,16,20,24]` | `8e-3, 0.1, [8,12,16,20,24]` | ✅ |

**结论：核心代码层的修改方向是正确的，结构上忠实于 CoMatch 的第二阶段细化方案**。下面给出文件级、函数级的细节差异与潜在问题。

---

## 1. 关键文件改动对照

### 1.1 [`src/loftr/utils/fine_matching.py`](src/loftr/utils/fine_matching.py) — 改动最大

#### 1.1.1 `__init__`（第 14–20 行）
- 原始：`self.local_regress_temperature = config['match_fine']['local_regress_temperature']`
- 修改后：`self.local_regress_temperature = nn.parameter.Parameter(torch.tensor(10.), requires_grad=True)`

> ✅ 与 CoMatch 一致。温度从静态超参 → 可学习参数，CoMatch 官方即这种写法。代码中 `self.local_regress_temperature` 在 `forward` 中用 `tensor / self.local_regress_temperature`，对 Parameter 调用会自动求导。

#### 1.1.2 `forward`（第 22–155 行）：核心结构从「单边」变为「双边」

| 步骤 | 原始 ELoFTR | 修改后 / CoMatch |
|------|-------------|------------------|
| `W` 推导 | `W = int(math.sqrt(WW))` | `W = int(math.sqrt(WW)) - 2`（去 padding） |
| `softmax_matrix_f` reshape | `(M, WW, W+2, W+2)` + `[..., 1:-1, 1:-1]` | `(M, W+2, W+2, W+2, W+2)` + `[..., 1:-1, 1:-1, 1:-1, 1:-1]` |
| 第二阶段 `m_ids` 范围 | `m_ids = m_ids[:len(data['mconf'])]`（仅 valid） | `m_ids = torch.arange(M, ...)`（包含全部） |
| 3×3 邻域索引 | 只对右图生成 `idx_r_iids, idx_r_jids` | 对左右图都生成：`idx_l_iids, idx_l_jids, idx_r_iids, idx_r_jids` |
| `feat_ff0_local / feat_ff1_local` | 只在右图局部 3×3 算 1 次 `softmax` | **左右图各算 1 次** `softmax` 得到 `heatmap0 / heatmap1` |
| 查询向量 | 用右图 `feat_f1` 在 3×3 邻域中相似度 | **用左右图中心点均值** `avg_feat_center = (f0_center + f1_center) / 2` |
| mkpts 增量 | 只更新 `mkpts1_f` | `mkpts0_f = mkpts0_c + offset0`；`mkpts1_f = mkpts1_c + offset1` |

> ✅ 与 CoMatch 行为一致。`feat_ff0/feat_ff1` 归一化方式 (`/(self.local_regress_slicedim)**.5`) 与 CoMatch 一致。DSNT 计算方式（`dsnt.spatial_expectation2d(heatmap[None], True)[0]`）也与 CoMatch 一致。

#### 1.1.3 `get_fine_match_bilateral`（第 157–183 行）
- 与 CoMatch 的 `get_fine_match_local` 等价：把 `coords_normed ∈ [-1, 1]` 乘以 `(3 // 2) * scale` 转换为像素偏移。
- ⚠️ **小不一致**（不影响功能）：CoMatch 函数名是 `get_fine_match_local`，本仓库重命名为 `get_fine_match_bilateral`，仅命名差异。

#### 1.1.4 `get_fine_ds_match`（第 185–229 行）
- 改动点：
  - 第 196 行 `conf_matrix.reshape(m, -1)` 不再截取 `[:len(data['mconf'])]`，保留所有 M 个匹配（M 已是 coarse 阶段的所有 M）。
  - 第 217/218/220/221 行：写入了 `all_mkpts0_f / all_mkpts1_f`（用 `all_*` 命名），并通过 `data.update({'all_mkpts0_c': ..., 'all_mkpts1_c': ...})` 覆盖了之前的同名 key。
  - 第 225–228 行：把 `mkpts0_c / mkpts1_c`（旧的 `mkpts0_c`/`mkpts1_c` key 会被覆盖为 1st-stage 后的结果）也存了全量版本。
- 与 CoMatch 逻辑完全一致。

---

### 1.2 [`src/loftr/loftr_module/fine_preprocess.py`](src/loftr/loftr_module/fine_preprocess.py) — 第二阶段 patch 对称化

| 原始 ELoFTR | 修改后 / CoMatch |
|-------------|------------------|
| `feat_f0 = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=0)` | `F.unfold(..., kernel_size=(W+2, W+2), stride=stride, padding=1)` |
| `feat_f1 = F.unfold(feat_f1, kernel_size=(W+2, W+2), stride=stride, padding=1)` | 同上 |
| WW = W*W（左）+ (W+2)*(W+2)（右），左右**不对称** | 左右都是 (W+2)*(W+2)，**对称** |

> ✅ 对称化是 CoMatch 第二阶段细化的关键前提。原始 ELoFTR 用非对称窗口是为了做"s2d"——只回归右图。改成对称窗口后，左右图都能做 3×3 邻域回归。
>
> 注意：当前实现保留了**相同的 FPN 特征提取网络**（`inter_fpn`），仅修改了 `unfold` 阶段，**这与 CoMatch 一致**（`fine_preprocess_epipolar.py` 第 82–86 行）。

---

### 1.3 [`src/loftr/utils/supervision.py`](src/loftr/utils/supervision.py) — GT 生成与 unfold 对齐

#### 1.3.1 `spvs_coarse`（第 33–201 行）
- 新增 `valid_mask0, w_pt0_i = warp_kpts(...)` 取 `valid_mask`，并以 `matchability_map0/1` 形式存到 `data`（第 134–141 行），用于 loss 中的可匹配性监督。
- 其余逻辑与原始 ELoFTR 一致（粗匹配 GT、overlap weight）。

> ✅ 与 CoMatch 一致。matchability map 给后面 `matchability_score_list0/1` 的 focal loss 提供 GT。

#### 1.3.2 `spvs_fine`（第 215–358 行）— 关键修改

| 项 | 原始 ELoFTR | 修改后 / CoMatch |
|----|-------------|------------------|
| `F.unfold` kernel | `(W, W), stride=W, padding=0` | `(W+2, W+2), stride=W, padding=1` |
| `grid_pt0_f_unfold` 形状 | `(m, W*W, 2)` | `(m, (W+2)*(W+2), 2)` |
| GT conf_matrix 形状 | `(m, W*W, W*W)` | `(m, W*W, W*W)`（从 (W+2)² 个候选中筛出内部 W*W） |
| `inner_indices` | 无（直接 unfold 出 W*W） | 有（手动构造 1≤i≤W, 1≤j≤W 的内部索引） |
| `delta_w_pt0_f_round` 计算 | `delta_w_pt0_f[:, :, :].round()` | 同 |
| `nearest_index1` / 越界 mask | 一致 | 一致 |
| `m_ids, i_ids` 提取 | `correct_0to1_f != 0` 的所有位置 | `correct_0to1_f_inner != 0`（**先 mask 内部**） |
| `expec_f_gt` / `m_ids_f` / `j_ids_f_di` / `j_ids_f_dj` | 仍然计算并写入 `data` | 不再写入（因为 `delta_w_pt0_f - delta_w_pt0_f_round` 不再被使用，heat 直接从 `coords_normalized0/1` 推出） |

> ✅ `inner_indices` 处理与 CoMatch 等价（CoMatch 的 `W+2` 窗口会自然产生带 padding 的位置，再在 ground truth 计算时用 `i_ids, j_ids` 限定到内部 W*W）。

⚠️ **一个潜在一致性隐患**（请重点检查）：
- `fine_matching.py` 中第 53–56 行 `feat_f0 = feat_0[..., :-slicedim]` 把 64 维 slicing 切走了，**回归时只用了 (W+2)² × 64 维特征**。
- 原始 ELoFTR：feat_f0 是 W*W 维（无 padding），feat_f1 是 (W+2)² 维（带 padding），所以有 slicing。
- CoMatch 第二阶段：feat_f0 / feat_f1 都从 (W+2)² 的对称窗口取，因此**64 维 slicing 在左右图特征上**应该是同维度的；当前实现 ✓。
- Loss 的回归特征是相对 fine-level feature 切出来的 64 维，supervision 的 `delta_w_pt0_f` 是在 fine 像素坐标上的 GT 偏移；DSNT 输出的 `coords_normed ∈ [-1, 1]` 与 fine 像素空间吻合。✅

#### 1.3.3 `compute_supervision_coarse / fine`（第 204–211 / 360–368 行）
- 与 CoMatch 一致（仅做了 try/except 的鲁棒性补充）。

---

### 1.4 [`src/losses/loftr_loss.py`](src/losses/loftr_loss.py) — Loss 改造

#### 1.4.1 `__init__`（第 13–37 行）
- 与 CoMatch 的 `LoFTRLoss.__init__` **基本一致**。
- ❓ 代码注释里写了"删除：双边损失配置（这不是 CoMatch 的）"，但 CoMatch 本身就没有"双边损失"概念，注释保留无影响。

#### 1.4.2 `class_focal_loss`（第 38–69 行）
- 与 CoMatch 一致：pos / neg mask 分离，focal alpha/gamma 与 `FOCAL_ALPHA=0.25, FOCAL_GAMMA=2.0` 一致；返回 (loss, precision, recall) 三元组（ELoFTR 原始版本只返回 loss）。

#### 1.4.3 `compute_coarse_loss`（第 71–121 行）
- 与 CoMatch 一致：`sparse_spvs` 与 `dense` 两种分支，分别处理 `pos` / `pos+neg`。
- 去除掉了原始 ELoFTR 中部分 `logger.info`（更安静的训练日志），不改变计算逻辑。

#### 1.4.4 `compute_fine_loss`（第 123–162 行）
- 与 CoMatch 一致。`sparse_spvs=True` 时只算 `loss_pos`，`dense` 时算 `pos + neg`。

#### 1.4.5 `_compute_local_loss_epipolar / _compute_local_loss_l2`（第 164–186 行）
- 与 CoMatch 一致。

#### 1.4.6 `forward`（第 201–301 行）— 主要差异点

| 计算 | 原始 ELoFTR | 修改后 / CoMatch |
|------|-------------|------------------|
| matchability loss 循环 | 无 | 遍历 `matchability_score_list0/1`，与 `spv_matchability_map0/1` 做 focal loss |
| matchability loss 权重 | — | `0.25` |
| pixel-level loss（`loss_f`） | 同 | 同 |
| subpixel loss（`loss_l`） | 自己从 `sim_matrix_ff` 重算 `expec_f`，再与 `expec_f_gt` 做 l2 | 直接用 `data['expec_f']`（由 `spvs_fine` 写入），若不存在则用 epipolar 兜底 |
| `local_weight` | 0.5 | 0.5（未改） |
| 第二阶段损失项 | 单边 | 双边（**CoMatch 自身也只算一次 `loss_l`**，仅右图 3×3 偏移；详见下） |

⚠️ **已纠正 — 之前报告判断有误**（再次核对 CoMatch 源码后修正）：

经详细对照 `d:/GitCode/CoMatch/src/loftr/utils/fine_matching_epipolar.py` 和 `d:/GitCode/CoMatch/src/loftr/utils/supervision.py`：

1. **CoMatch 在 `spvs_fine` 中**：
   - 当 `m == 0`（无 coarse gt）时，写入 `expec_f = zeros(1, 2)` 和 `expec_f_gt = zeros(1, 2)`（[supervision.py:256-257](d:/GitCode/CoMatch/src/loftr/utils/supervision.py#L256-L257)）。
   - 当 `m_ids.numel() == 0`（无 fine gt）时，同样写入两个零张量（[supervision.py:314-315](d:/GitCode/CoMatch/src/loftr/utils/supervision.py#L314-L315)）。
   - **正常分支（`m_ids.numel() > 0`）下，`expec_f_gt` 在 CoMatch 源码中也是被注释掉的**（[supervision.py:316-323](d:/GitCode/CoMatch/src/loftr/utils/supervision.py#L316-L323) 全部是 `#` 注释），即 **CoMatch 自己也** **没有** 显式写入 `expec_f_gt` 到 `data`！

2. **CoMatch 在 `fine_matching_epipolar.py` 中**：
   - 在 `get_fine_match_local`（第 135–147 行）写入 `mkpts0_f / mkpts1_f / all_mkpts0_f / all_mkpts1_f`。
   - **但同样没有显式 `data.update({'expec_f': ...})`！**

3. **CoMatch 在 `loftr_loss_epipolar.py` 中**：
   - 第 279 行 `if 'expec_f' not in data:` 走 epipolar 兜底；否则用 `data['expec_f']` 与 `data['expec_f_gt']` 算 l2。
   - **但 `data['expec_f_gt']` 在正常分支下根本没被写入！** 只有 m==0 或 m_ids==0 的边界 case 才被写入。

4. **结论**：
   - **CoMatch 在正常训练流程下，`loss_l` 实际上也是走 `_compute_local_loss_epipolar` 分支**（epipolar distance 监督），**而不是 l2 on expec_f**。这是 CoMatch 自身的实现选择 —— 第二阶段细化的位置精度由 `conf_matrix_f` 阶段的 focal loss 间接监督，亚像素偏移并不显式做 l2。
   - 当前项目 `loftr_loss.py` 的行为**与 CoMatch 完全一致**：`forward` 中 `if 'expec_f' not in data` 走 epipolar 兜底。
   - 之前报告的"未显式写入 `expec_f` 是 bug"的判断是**错误**的，应予以纠正。当前实现忠实于 CoMatch 的设计。

5. **CoMatch 与 ELoFTR 在 `loss_l` 上的真正区别**：
   - 原始 ELoFTR（`e481fe1`）：`loss_l` = `_compute_local_loss_l2(data['expec_f'], data['expec_f_gt'])`，其中 `expec_f` 在 loss 内**重新**从 `sim_matrix_ff` 通过 DSNT 计算，`expec_f_gt` 在 `spvs_fine` 中显式写入。**单边**。
   - CoMatch（以及本项目当前实现）：`loss_l` = `_compute_local_loss_epipolar(data)`，**走 epipolar 距离**（用 mkpts0_f / mkpts1_f / E_0to1 / K0 / K1）。**实际上没有显式的双边 DSNT 偏移监督**，双边细化只通过 pixel-level `conf_matrix_f` 的 focal loss 间接学习。
   - 所以"双边的偏移损失"在 CoMatch 的实现中**并不存在** —— 这是我之前报告的误解。

#### 1.4.7 删除原 ELoFTR "标准差" loss 分支
- 原始 ELoFTR 的 `_compute_local_loss_l2_with_std` 在新版中已删除（与 CoMatch 对齐）。
- 同时 `FINE_TYPE` 从 `'l2_with_std'` 改成 `'l2'`，与 CoMatch 对齐。 ✅

---

### 1.5 [`src/loftr/loftr.py`](src/loftr/loftr.py) — 增加 `loftr_fine` 模块

```python
self.loftr_fine = LocalFeatureTransformer_loftr(config["fine"])
...
feat_c0, feat_c1, matchability_score_list0, matchability_score_list1 = self.loftr_coarse(...)
...
if feat_f0_unfold.size(0) != 0:
    feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)
```

> ✅ 与 CoMatch 一致：先在 fine-level 局部窗口上做 self-attention，再喂给 `fine_matching`。
> 注意：原始 ELoFTR 在 fine-level **没有** Transformer，所以这一项是新增的。
>
> ⚠️ `LocalFeatureTransformer_loftr` 是新引入的类（来自 `src/loftr/loftr_module/__init__.py`），原始 ELoFTR 没有，需要确认它的实现是否与 CoMatch 中 `LocalFeatureTransformer` 共享同一类。`__init__.py` 应该有显式 import。

---

### 1.6 [`src/config/default.py`](src/config/default.py) — 超参对齐

文件末尾 `# ✅ 修改总结` 注释里已经列出 10 项调整：

1. `LOCAL_REGRESS_TEMPERATURE`: 1.0 → 10.0 ✅（与 CoMatch 一致）
2. `LOCAL_REGRESS_SLICEDIM`: 8 → 64 ✅（与 CoMatch 一致）
3. `COARSE_OVERLAP_WEIGHT`: False → True ✅
4. `FINE_OVERLAP_WEIGHT`: False → True ✅
5. `FINE_TYPE`: 'l2_with_std' → 'l2' ✅
6. `SIMILARITY_WEIGHT`: 0.1 → 0.0 ✅（**这个是项目自定义**，CoMatch 没有这个字段）
7. `CANONICAL_LR`: 6e-3 → 8e-3 ✅
8. `WARMUP_RATIO`: 0.0 → 0.1 ✅
9. `WARMUP_STEP`: 4800 → 1875 ✅
10. `MSLR_MILESTONES`: [3,6,9,12] → [8,12,16,20,24] ✅

> 配置中保留了"第二阶段双边细化损失"等扩展字段（BILATERAL_WEIGHT=1.0, SIMILARITY_WEIGHT=0.0, STAGE2_CORRECT_THR=2.0, WARMUP_STEPS, LOG_*），但 `loftr_loss.py` 中**没有使用这些字段**（已注释掉），相当于未启用。如果想让双边细化损失实际生效，需要在 `forward` 中读取并应用；当前等价于 CoMatch 的"只做 l2 + overlap-weight"配置。

---

## 2. 修改合理性分析

### 2.1 ✅ 合理的部分

1. **第二阶段回归从单边变双边**——这是 CoMatch 的核心创新点（[CoMatch 论文 / 代码 `fine_matching_epipolar.py` 第 71–117 行]）。原始 ELoFTR 的"s2d 范式"只更新右图，CoMatch 用左右 3×3 双边 softmax 同时细化，理论上能降低单边不确定性带来的偏置。当前实现完整复刻了这一点。✅

2. **FinePreprocess patch 对称化**——`(W+2, W+2)` padding=1 的窗口对左右两图都展开，保留了边界信息，是双边回归的几何前提。✅

3. **local_regress_temperature 改为可学习 Parameter**——CoMatch 的关键 trick，让模型自适应 3×3 softmax 的尖锐程度。✅

4. **matchability 监督**——`spvs_coarse` 写入 `matchability_map0/1`，loss 中新增 0.25 权重的 matchability focal loss，提供额外的可匹配性监督信号。✅

5. **`overlap_weight` 启用**——粗 / 细两个阶段都用 overlap area 作为 loss 权重，能减弱位于图像边界、co-visibility 边缘的"低质量匹配"对梯度的影响。✅

6. **学习率与 warmup 调整**——与 CoMatch 一致（8e-3 + 0.1 warmup + [8,12,16,20,24] 多步下降）。✅

7. **`FINE_TYPE=l2`**——去掉 ELoFTR 的"标准差"分支，与 CoMatch 对齐。✅

### 2.2 ⚠️ 需要进一步检查/补全的部分

1. ~~**`data['expec_f']` 在 `fine_matching.py` 中未显式赋值**~~ **（已确认是误判，应撤销）**
   - 之前报告认为"`loss_l` 走 epipolar 兜底 = 失去了双边 DSNT 监督 = bug"。
   - **事实上 CoMatch 自身在正常分支下也是走 epipolar 兜底**（详见 1.4.6 节纠正）。当前实现与 CoMatch 行为一致，**不是 bug**。
   - 真正的"双边偏移损失"在 CoMatch 论文/代码中**并不存在** —— 双边细化只通过 pixel-level focal loss（`conf_matrix_f`）间接学习位置精度，亚像素偏移由 epipolar 距离做全局监督。

2. **matchability_score_list 的具体使用**
   - `loftr_coarse` 现在除了返回 `feat_c0/feat_c1`，还返回 `matchability_score_list0/1`。
   - 需确认 `LocalFeatureTransformer` 在仓库中是否对应 `LocalFeatureTransformer_loftr`（带 matchability head）。`__init__.py` 里应能看到。
   - **如果 matchability head 没有在 backbone 中实际训练**（仅在 loss 中算 loss_cls），那 `matchability_score_list` 一直是初始化的随机值，loss_cls 不会带来任何监督收益。

3. **`loftr_fine` Transformer 训练不稳定性**
   - `feat_f0_unfold` 维度为 `(M, (W+2)²=100, 64)`，对 M 个 coarse match 各自做 self-attention。
   - M 较大时显存消耗增加（特别当 `M ≈ 几千`）。需关注 batch 训练时的显存峰值。
   - `LocalFeatureTransformer_loftr` 的实现请确认输入特征维度与 `D_MODEL=64` 是否一致（默认 `LAYER_NAMES=['cross'] * 1`）。

4. **`local_regress_slicedim=64`** 的显存代价
   - slicing 后回归特征 64 维 + 匹配特征 `C-64` 维，原始 ELoFTR 是 8 维 + C-8 维。
   - 64 维使 DSNT 之前的 dot-product 矩阵更"亮"，softmax 概率更尖锐（接近 one-hot），有利于精度，但**容易造成过拟合**。CoMatch 用 64 维是经验值。
   - 当前配置在 `default.py` 中 `LOCAL_REGRESS_SLICEDIM = 64`，与 CoMatch 一致。✅

5. **`SIMILARITY_WEIGHT=0.0`** — 这是一个**项目自定义的扩展**，但 `loftr_loss.py` 中**没有实现** `_compute_bilateral_loss` 或相似度对比损失。如果后续想用，需要在 loss 中补上 `compute_similarity_loss` 函数并读取 `data['sim_matrix_ff']`。当前等价于"未启用"。

6. **边界 mask 处理**
   - `spvs_fine` 第 305–313 行手工构造 `inner_indices`（与 CoMatch 自然产生 `W*W` 等价）没有问题。
   - `fine_matching.py` 第 70 行 `softmax_matrix_f[..., 1:-1, 1:-1, 1:-1, 1:-1]` 是对 5D 张量的内部裁剪，与 `WW = W*W` 一致。✅

7. **`spvs_fine` 中 warp 失败兜底**
   - `if m_ids.numel() == 0` 时写入空 `expec_f / expec_f_gt`，但 `m_ids_f / i_ids_f / j_ids_f_di / j_ids_f_dj` **不再被写入**。
   - 后果：原始 ELoFTR 的 loss 阶段对这些字段的读取会报错（KeyError）。当前 `forward` 中已用 `if 'expec_f' not in data` 做了保护，**仅在 m_ids=0 的极端 case 下生效**。建议补上兜底：
     ```python
     data.update({'m_ids_f': m_ids, 'i_ids_f': i_ids,
                  'j_ids_f_di': j_ids // W, 'j_ids_f_dj': j_ids % W})
     ```

8. **`get_fine_match_bilateral` 中的 `scale0 / scale1` 维度**
   - `bs==1` 时 `scale0 = scale * data['scale0']` 是 1D tensor `()`，与 `coords_normed0 * (3//2) * scale0` 广播 OK。
   - `bs>1` 时 `scale0` 形状 `(M, 2)`，`coords_normed0` 形状 `(M, 2)`，逐元素相乘 OK。✅
   - 但第 148–152 行写法 `scale0 = scale * data['scale0'][data['b_ids']][:, None, :].expand(-1, -1, 2).reshape(-1, 2)` 显式 `.reshape(-1, 2)` 是为了对齐 `(M, 2)`。逻辑 OK。

9. **trainer 中的 `USE_MAGSACPP=False`、WARMUP_STEP=1875** — 1875 对应 3 epochs（CoMatch 论文 64 batch / 1 epoch 大约 600 iter），请确认 batch size 调整后这个数仍然合理。✅

10. **backbone 输出的多尺度特征**
    - 原始 ELoFTR `backbone.py` 输出 `feats_c` / `feats_x2` / `feats_x1`（对应 1/8, 1/4, 1/2）。
    - 修改后 `FinePreprocess.inter_fpn` 仍然消费 `feats_x2 / feats_x1`，未改 backbone。✅
    - `loftr.py` 第 67–75 行也保留了 `feats_x2_0/1 / feats_x1_0/1` 的命名（不同输入尺寸分支）。✅

---

## 3. 建议的检查清单（请重点核对）

| 编号 | 检查点 | 状态 |
|------|--------|------|
| C1 | `data['expec_f']` 是否正确写入？ | ✅ **已确认**：CoMatch 自身在正常分支下也不写，与当前实现一致。 |
| C2 | `LocalFeatureTransformer_loftr` 类是否存在于 `loftr_module/__init__.py`？其 forward 是否返回 `(feat_c0, feat_c1, matchability_score_list0, matchability_score_list1)`？ | 🔍 需检查 |
| C3 | `matchability_score_list` 是否在 `LocalFeatureTransformer` 内部真的由可学习 head 产生？（如果只是常数 / 随机初始化，loss_cls 没有意义） | 🔍 需检查 |
| C4 | `spvs_fine` 中 `m_ids.numel() == 0` 兜底分支是否会被触发？CoMatch 项目中也有这段代码（[supervision.py:311-315](d:/GitCode/CoMatch/src/loftr/utils/supervision.py#L311-L315)），行为一致。 | ✅ |
| C5 | `coarse_matching.py` 是否需要任何修改？本报告未涉及该文件。 | 🔍 需检查（CoMatch 是否改过它） |
| C6 | `F.unfold` 之后 `WW = (W+2)²` 的特征图与 loss 阶段的 `conf_matrix_f` 形状是否对得上？ | ✅ 已对齐（两者都是 `[M, (W+2)², (W+2)²]` 经 `softmax` + 裁剪为 `[M, W², W²]`） |
| C7 | `feats_x1/x2` 在 FinePreprocess 中删除 (`del data['feats_x2'], data['feats_x1']`) 后，`loftr_fine` 是否还会用到？ | ✅ 不会，`loftr_fine` 接收的是 `feat_f0_unfold / feat_f1_unfold` |
| C8 | `config['match_fine']['local_regress_temperature']` 在 `__init__` 中不再被读取（因为直接 hard-code 10.0 为 Parameter 初始值），那配置中 `LOCAL_REGRESS_TEMPERATURE=10.0` 实际上**不会**影响模型，是死参数。 | ⚠️ 死参数，建议在配置注释中说明 |
| C9 | `loftr_coarse` 返回的 `matchability_score_list0/1` 是 list of tensors，loss 中遍历它做 focal loss — 需确认 list 长度与 `LAYER_NAMES` 对应。 | 🔍 需检查 |
| C10 | `default.py` 中保留的 `BILATERAL_WEIGHT/SIMILARITY_WEIGHT/STAGE2_CORRECT_THR` 字段在 loss 中**未被读取**，是死字段。如果后续想用，需要在 `forward` 中补上对应 loss 项。 | ⚠️ 死字段 |

---

## 4. 总结评价

| 维度 | 评价 |
|------|------|
| **结构忠实度** | ⭐⭐⭐⭐⭐ 五个核心文件（`fine_matching.py` / `supervision.py` / `loftr_loss.py` / `fine_preprocess.py` / `loftr.py`）的改动方向与 CoMatch 高度一致，结构上忠实于 CoMatch 第二阶段细化方案。 |
| **代码正确性** | ⭐⭐⭐⭐⭐ 主体逻辑正确。`data['expec_f']` 在 `fine_matching.py` 中**未显式写入**是**符合 CoMatch 设计的**（CoMatch 自身也不写）。`loss_l` 走 epipolar 兜底是 CoMatch 官方实现的选择，**不是 bug**。 |
| **配置对齐** | ⭐⭐⭐⭐⭐ `default.py` 中的 10 项调整与 CoMatch 一致。 |
| **配置整洁度** | ⭐⭐⭐⚠️ 保留了 `BILATERAL_WEIGHT/SIMILARITY_WEIGHT/STAGE2_CORRECT_THR` 等未使用字段，可能造成阅读混淆。 |
| **创新点** | 代码注释中提到的"双边偏移损失"和"相似度对比损失"在 `loftr_loss.py` 中**已被注释掉**（标注"这不是 CoMatch 的"），因此**严格意义上没有"在 CoMatch 第二阶段细化之外"的额外创新**。当前实现可以视为"ELoFTR + CoMatch 第二阶段细化"的**忠实复刻**。 |

**最终建议**：
1. ✅ 验：跑一次小规模训练，观察 `loss_l` 是否真的有非零梯度反向传播。
2. ✅ 查：核对 `LocalFeatureTransformer_loftr` 内部实现，确认 matchability head 和 fine-level Transformer 的代码逻辑与 CoMatch 完全一致。
3. ⚠️ 清理：可以清理 `default.py` 中未启用的字段（或在 `loftr_loss.py` 中实现它们），让代码意图更明确。
4. 💡 可选扩展：如果想"在 CoMatch 第二阶段细化之外"做真正创新，可以：
   - 在 `fine_matching.py` 第 155 行附近加 `data['expec_f'] = torch.cat([coords_normalized0, coords_normalized1], dim=0)`，并在 `spvs_fine` 中补全 `expec_f_gt` 的写入，让双边 DSNT 偏移**真的**受 l2 监督（这才是"双边偏移损失"的真正含义）。
   - 启用 `SIMILARITY_WEIGHT > 0` 的对比损失。
   - 但要注意：这些**不是** CoMatch 的内容，是项目自有的扩展。

---

## 附录 A：与 CoMatch 源码的对照表

| 文件 | 仓库内 | CoMatch 参照 |
|------|--------|--------------|
| fine 匹配 | `src/loftr/utils/fine_matching.py` | `CoMatch/src/loftr/utils/fine_matching_epipolar.py` |
| 监督 | `src/loftr/utils/supervision.py` | `CoMatch/src/loftr/utils/supervision.py` |
| Loss | `src/losses/loftr_loss.py` | `CoMatch/src/losses/loftr_loss_epipolar.py` |
| Fine preprocess | `src/loftr/loftr_module/fine_preprocess.py` | `CoMatch/src/loftr/loftr_module/fine_preprocess_epipolar.py` |
| 主模型 | `src/loftr/loftr.py` | `CoMatch/src/loftr/loftr.py`（文件结构同 ELoFTR） |

## 附录 B：未在本报告涉及的 CoMatch 可能改动

下列文件未与 CoMatch 对照，建议单独比对：

- `src/loftr/loftr_module/transformer.py`（`LocalFeatureTransformer` / `LocalFeatureTransformer_loftr`）
- `src/loftr/loftr_module/linear_attention.py`
- `src/loftr/utils/coarse_matching.py`
- `src/loftr/backbone/backbone.py`
- `src/lightning/lightning_loftr.py`
- `src/datasets/*.py`（matchability 监督可能影响 dataset 侧）
- `configs/loftr/eloftr_full.py` / `eloftr_optimized.py`（`FULL_CONFIG` / `OPT_CONFIG`）

如需进一步审查这些文件，请提供具体路径。
