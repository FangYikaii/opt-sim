# backend/artifacts 训练产物分析与优化建议（2026-04-26）

## 1. 分析范围与结论摘要

本文基于 `backend/artifacts` 下现有 cGAN 相关实验目录、训练日志、评估图片和当前算法代码进行交叉分析，重点回答两件事：

1. 目前算法架构和模型训练存在哪些问题。
2. 下一步应该如何更稳、更快地优化训练。

核心结论先给出来：

- 当前实现本质上是一个 `Lab -> 三层膜厚(d1,d2,d3)` 的条件生成模型，不是论文原始方案的完整复现，而是一个工程化的“轻量替代版”。
- 这套实现已经能学到一定的颜色到结构映射，但训练目标之间存在明显冲突：当模型更强调分布多样性时，`d2` 恢复精度和 `DeltaE` 往往会明显变差。
- 从现有产物看，`cgan_reproduction_confirm_tune4_alpha_20260425` 是当前综合最可用的结果；`cgan_reproduction_confirm_tune4_alpha_score_d3guard_20260426` 更分散，但最终质量更差；`tune5/6/7` 这组三个 SN 结构实验在 500 epoch 时已经明显退化。
- 当前最大的瓶颈不在“网络宽度不够”，而在“目标函数设计、checkpoint 评估一致性、条件约束强度、边界偏置和实验可解释性”。

## 2. 当前算法架构梳理

### 2.1 数据与任务

- 训练数据来自 `backend/data/paper-reproduction/training set.csv`，训练集 45000 个样本，测试集 5000 个样本。
- 输入条件是颜色空间 `Lab`。
- 生成目标是三层厚度参数：`Ag bottom / SiO2 / Ag top`，单位为 nm。
- 设计变量在代码里做 min-max 归一化，`Lab` 做标准化。

相关代码：

- 归一化与标准化定义：`backend/app/algorithms/cgan.py:159-172`
- 数据载入：`backend/scripts/train_cgan_reproduction.py:133-157`
- paper 数据集训练入口：`backend/scripts/train_cgan_reproduction.py:1393-1403`

### 2.2 模型结构

当前模型由三部分组成：

1. `LabRegressor`
2. `Generator`
3. `Evaluator/Discriminator`

#### `LabRegressor`

- 输入：归一化后的三层厚度。
- 输出：标准化后的 `Lab`。
- 作用：给生成器提供“颜色一致性”约束。

实现位置：`backend/app/algorithms/cgan.py:46-55`

#### `Generator`

- 输入：`target Lab + noise z`
- 输出：归一化后的 `d1,d2,d3`
- 输出层是 `Sigmoid`，随后再反归一化并 clip 到设计边界。

实现位置：

- 结构：`backend/app/algorithms/cgan.py:58-87`
- 采样与反归一化：`backend/app/algorithms/cgan.py:802-901`

#### `Evaluator / Discriminator`

- 输入只有生成的设计参数 `d1,d2,d3`，不看 `Lab` 条件。
- 这是一个“无条件判别器”，只负责判断厚度样本是否像训练分布。

实现位置：`backend/app/algorithms/cgan.py:90-109`

### 2.3 训练目标

训练循环在 `fit_lightweight_cgan` 中实现：`backend/app/algorithms/cgan.py:316-674`

判别器采用 hinge 风格目标：

- `max(0, 1 - D(real))`
- `max(0, 1 + D(fake))`

见：`backend/app/algorithms/cgan.py:461-468`

生成器目标为：

`generator_loss = -D(fake) + alpha * lab_mse`

见：`backend/app/algorithms/cgan.py:470-478`

也就是说，这套训练实际上在平衡两个目标：

- 让输出落在“像真实厚度分布”的区域
- 让输出经 `LabRegressor` 预测后尽量接近目标颜色

`alpha` 会线性爬升：

- `alpha = max_alpha * epoch / alpha_ramp_epochs`

见：`backend/app/algorithms/cgan.py:438-440`

### 2.4 Checkpoint 选择与最终评估

训练时每隔若干 epoch 做一次 checkpoint 评估，评估指标来自 paper 风格测试集统计：

- `mean_best_delta_e`
- `median_best_delta_e`
- `d2_ground_truth_within_5nm_ratio`
- `JSD(d1,d2,d3)`
- `solution_group_counts`

核心评估逻辑：

- 生成测试集每个目标的若干候选解：`backend/scripts/train_cgan_reproduction.py:374-503`
- checkpoint 综合打分函数：`backend/scripts/train_cgan_reproduction.py:506-534`
- 训练过程中 checkpoint 评估：`backend/scripts/train_cgan_reproduction.py:1477-1516`
- 最终整套 artifacts 导出：`backend/scripts/train_cgan_reproduction.py:1599-1669`

这里有一个很重要的实现细节：

- 训练中选 best checkpoint 时，使用的是 `min(paper_samples_per_lab, checkpoint_samples_per_lab)`。
- 最终 `metrics.json` 导出时，用的是完整的 `paper_samples_per_lab`。

代码证据：

- checkpoint 评估采样数：`backend/scripts/train_cgan_reproduction.py:1480-1485`
- 最终评估采样数：`backend/scripts/train_cgan_reproduction.py:1600-1613`

这意味着：训练中的 best checkpoint 分数，和最终导出的 paper 指标，并不是严格同一评估预算下得到的。

## 3. artifacts 里各实验的主要现象

## 3.1 完整评估 run

### A. `cgan_reproduction_paper_stable_20260425_old`

特点：

- 旧基线，训练 `100000` epoch，但实际 best checkpoint 在 `1000` epoch。
- 最终 `paper_reproduction.samples_per_lab = 1000`

关键指标：

- checkpoint score: `13.1578`
- mean best DeltaE: `11.9934`
- median best DeltaE: `10.1304`
- `d2 within 5nm`: `0.6540`
- mean solution groups: `1.2932`

图像现象：

- `paper_figure4_distribution_comparison.png` 显示生成分布在 `d1/d3` 两端和 `d2` 多峰区域有明显偏置。
- `candidate_diversity.png` 显示某些颜色目标上候选几乎挤到边界，尤其 `Ag top` 接近 0。
- `candidate_samples.csv` 里 `#4f86c6` 的 cGAN 候选明显坍塌到低 `d3` 区域，最优 DeltaE 仍高达 `44+`。

判断：

- 该版本整体分布比后续某些 run 更稳定，但颜色条件约束不够，出现了“像真实分布，但不满足目标颜色”的问题。

### B. `cgan_reproduction_confirm_tune4_alpha_20260425`

特点：

- `g_lr=1e-3`, `d_lr=2e-4`
- `alpha_ramp_epochs=40000`, `max_alpha=0.3`
- final `paper_samples_per_lab = 1000`

关键指标：

- checkpoint score: `16.0419`
- final evaluation mean best DeltaE: `7.7226`
- final evaluation median best DeltaE: `7.0354`
- final evaluation `d2 within 5nm`: `0.8858`
- final evaluation mean solution groups: `1.1782`

图像现象：

- `loss_curve.png` 中判别器很快收敛到约 `2.0`，生成器 loss 稳定在小正值。
- `paper_figure_s6_d2_accuracy.png` 的 `|Delta d2|` 分布相对更集中，长尾比旧基线短。
- `candidate_samples.csv` 里三组颜色目标中，只有 `#4f86c6` 明显优于 retrieval；其它目标仍存在质量不稳。
- `paper_figure4_distribution_comparison.png` 仍然存在强边界堆积，尤其 `d1/d3` 靠近 50 nm。

判断：

- 这是目前“颜色质量最好、d2 恢复也最好”的一版。
- 但它的多解性很弱，`mean solution groups` 只有 `1.1782`，离 paper 目标 `3.58` 还差很远。

### C. `cgan_reproduction_confirm_tune4_alpha_score_d3guard_20260426`

特点：

- `alpha_ramp_epochs=80000`, `max_alpha=0.2`
- checkpoint 权重显式加入 `d3_jsd`
- final `paper_samples_per_lab = 64`

关键指标：

- checkpoint score: `16.7162`
- final evaluation mean best DeltaE: `12.0572`
- final evaluation median best DeltaE: `11.7634`
- final evaluation `d2 within 5nm`: `0.4466`
- final evaluation mean solution groups: `6.3194`

图像现象：

- `paper_figure5_solution_metrics.png` 显示 solution groups 的确显著增多。
- `candidate_diversity.png` 里候选散布更广，说明模型确实学会了更强的“发散”。
- 但 `paper_figure_s6_d2_accuracy.png` 明显更宽，`d2` 恢复出现长尾。
- `analysis_checkpoint_trajectory.png` 显示 1500 epoch 最佳，之后 checkpoint score 连续恶化。

判断：

- 该实验在“多解性/分布分散度”上成功了，但牺牲了颜色命中率和 `d2` 恢复精度。
- 更关键的是，这个 run 的最终指标是基于 `64` 个样本/目标，不可直接与 `1000` 样本评估的 run 横向比较。

## 3.2 部分完成 run（只有日志/checkpoint，未见完整最终指标）

### D. `cgan_reproduction_confirm_tune4_alpha_midpoint_20260426`

可见结果：

- best epoch: `500`
- score: `17.3817`
- mean best DeltaE: `12.3979`
- `d2 within 5nm`: `0.4320`

判断：

- 相比 `confirm_tune4_alpha_20260425`，checkpoint 阶段已经明显更差。
- 说明仅靠放缓 `alpha` 并不能自然提升多解与精度的平衡。

### E. `cgan_reproduction_confirm_tune5_sngan_dual_sn_depth_20260426`

可见结果：

- best epoch: `500`
- score: `31.7637`
- mean best DeltaE: `23.6306`
- `d2 within 5nm`: `0.1092`

训练现象：

- `d_loss` 接近 `1.0`
- `g_loss` 很快变成负值
- 500 epoch 就已经很差

判断：

- 这是明显退化实验。
- 更强的 SN + 更深 generator 并没有稳定训练，反而破坏了条件映射能力。

### F. `cgan_reproduction_confirm_tune6_sngan_balanced_160_160_20260426`

可见结果：

- best epoch: `500`
- score: `26.4273`
- mean best DeltaE: `18.5702`
- `d2 within 5nm`: `0.1406`

判断：

- 比 tune5 好，但仍显著差于 tune4 系列。
- “平衡 G/D hidden dim” 没有解决核心问题。

### G. `cgan_reproduction_confirm_tune7_sngan_conservative_g128_d160_20260426`

可见结果：

- best epoch: `500`
- score: `28.5693`
- mean best DeltaE: `20.3505`
- `d2 within 5nm`: `0.0882`

判断：

- 更保守的 generator 容量也没有救回来。
- 说明问题不主要是“G 太强/太弱”，而是训练目标和条件约束设计本身。

## 4. 当前算法架构存在的问题

## 4.1 判别器没有接收条件 `Lab`，导致“条件正确性”主要靠回归器兜底

当前 `Evaluator` 只看 `design_norm`，不看目标 `Lab`：`backend/app/algorithms/cgan.py:90-109`

后果：

- 判别器只学习“像不像真实厚度分布”，不会惩罚“颜色条件错配”的样本。
- 生成器会优先去找判别器喜欢的厚度区域，再通过 `alpha * lab_mse` 被动拉回目标颜色。
- 一旦 `alpha` 太小，模型就更像无条件生成器；一旦 `alpha` 太大，又会把生成结果硬拉到少数容易满足颜色的区域。

这正好对应当前实验现象：

- `paper_stable_old` 更像“分布合理但颜色不准”。
- `confirm_tune4_alpha` 更像“颜色更准但多样性下降”。
- `confirm_tune4_alpha_score_d3guard` 更像“更分散但 d2 和 DeltaE 变差”。

## 4.2 `LabRegressor` 充当了代理物理模型，但它不是冻结真值物理器

当前 `LabRegressor` 是从数据拟合得到的神经网络近似器：`backend/app/algorithms/cgan.py:243-313`

后果：

- 生成器优化的是“回归器认为颜色对”，不是“真实物理仿真后颜色对”。
- 如果回归器在某些区域有系统偏差，生成器会专门钻这个空子。
- 这会形成典型的 surrogate exploitation。

证据：

- `paper_targets` 中给出的 paper 理想值非常高，但现有模型与其有明显差距。
- 某些候选在回归器驱动下看似合理，真实 `evaluate_design()` 后 DeltaE 仍偏高。

## 4.3 `alpha` 调度过慢，很多实验在有效颜色约束真正生效前就被选了 checkpoint

当前 `alpha` 线性增长：`backend/app/algorithms/cgan.py:438-440`

以几个实验为例：

- `confirm_tune4_alpha_20260425` 在 epoch 1000 时，`alpha` 只有 `0.0075`
- `confirm_tune4_alpha_score_d3guard_20260426` 在 epoch 1500 时，`alpha` 只有 `0.00375`
- `tune5/6/7` 在 epoch 500 时，`alpha` 只有 `0.00125`

这意味着：

- 很多 best checkpoint 实际上是在“对抗项占主导、条件项还很弱”的阶段被选出来的。
- 尤其在 `tune5/6/7` 中，500 epoch 的结果几乎还处于非常早期的条件学习阶段。

## 4.4 训练与评估的采样预算不一致，导致 checkpoint 选择噪声较大

训练时选择 best checkpoint 的采样数是：

- `min(paper_samples_per_lab, checkpoint_samples_per_lab)`

最终导出 `metrics.json` 时用的是：

- `paper_samples_per_lab`

相关代码：

- checkpoint：`backend/scripts/train_cgan_reproduction.py:1480-1485`
- final：`backend/scripts/train_cgan_reproduction.py:1604-1613`

后果：

- 训练中 best checkpoint 可能只是在“小样本评估”下偶然较优。
- 最终全量评估时，排序可能变化。
- `64` 样本和 `1000` 样本的指标可比性明显不足，尤其是 `mean_best_delta_e` 和 `solution_groups`。

## 4.5 生成输出存在明显边界偏置和分布挤压

生成器输出经 `Sigmoid` 后直接映射到设计区间：`backend/app/algorithms/cgan.py:78-87, 844-845`

图像证据：

- `confirm_tune4_alpha_20260425/paper_figure4_distribution_comparison.png`
- `confirm_tune4_alpha_score_d3guard_20260426/paper_figure4_distribution_comparison.png`

可见问题：

- `d1/d3` 在 0 或 50 nm 附近有明显堆积。
- `d2` 分布存在非物理的局部聚峰或边界抬升。

可能原因：

- `Sigmoid + 线性反归一化` 天然更容易在边界附近饱和。
- 判别器只做整体分布拟合，没有“条件局部几何”约束。

## 4.6 “多解性指标”与“目标精度指标”之间缺少显式协调机制

目前用 DBSCAN 统计 `solution_group_counts`，但训练过程中并没有直接优化“同色多解”的结构表达，只是在 checkpoint 打分时被动偏好某些更分散的模型。

相关实现：

- DBSCAN 解簇统计：`backend/scripts/train_cgan_reproduction.py:113-121`
- paper 风格解簇评估：`backend/scripts/train_cgan_reproduction.py:409-435`

后果：

- 模型可能通过“胡乱发散”来增加解簇数，而不是学到真实的多模态条件分布。
- 这正是 `confirm_tune4_alpha_score_d3guard_20260426` 的典型现象：解簇变多，但 `DeltaE` 和 `d2` 精度下降。

## 4.7 实验记录与可解释性还有版本漂移问题

例子：

- `loss_curve.png` 标题仍写着 `Binary cross-entropy`，但训练实现已经是 hinge 风格损失。
- 代码位置：`backend/scripts/train_cgan_reproduction.py:972-981`

后果：

- 容易误导后续复盘，把 `d_loss≈2` 错误理解成 BCE 状态，而它在 hinge 语境下含义不同。

## 4.8 artifacts 导出逻辑有覆盖 best checkpoint 的风险

脚本先在训练中把 best checkpoint 写到 `generator_checkpoint_best.pt`，但训练结束后又执行：

- `save_model_bundle(training_bundle, final_checkpoint_path)`
- `save_model_bundle(training_bundle, best_checkpoint_path)`

见：`backend/scripts/train_cgan_reproduction.py:1549-1552`

虽然 `save_model_bundle()` 对 `generator_checkpoint_best.pt` 做了 `best_generator_state_dict` 特判：`backend/scripts/train_cgan_reproduction.py:909-916`，当前逻辑是能工作的，但这段设计仍然比较脆弱：

- 语义上“best checkpoint”和“final checkpoint”写入路径耦合过深。
- 后续一旦改 bundle 结构或保存逻辑，很容易产生 best/final 混淆。

## 5. 模型训练层面存在的问题

## 5.1 训练过早进入“判别器平衡假象”

多个 run 中：

- `d_loss` 很快贴近 `2.0`（旧/alpha 系列）
- 或很快贴近 `1.0`（tune5/6/7 系列）

这说明：

- loss 数值很快进入稳定区间
- 但不代表条件生成质量已经稳定

尤其 `tune5/6/7` 在 500 epoch 时已经 checkpoint 很差，说明“loss 看起来平”并不等于“任务做好了”。

## 5.2 大 batch + 单步更新，容易削弱条件细节学习

多个实验使用：

- `batch_size = 16384` 或 `40000`
- `steps_per_batch = 1`

相关参数定义：`backend/scripts/train_cgan_reproduction.py:1316-1327`

风险：

- batch 极大时，梯度更偏全局均值，容易学到“平均合理分布”，不利于细粒度条件区分。
- 对于 inverse design 这种一对多映射问题，大 batch 往往会加剧 mode averaging。

## 5.3 噪声维度过小且没有显式多模态约束

当前 `noise_dim = 2` 为默认配置：`backend/scripts/train_cgan_reproduction.py:1318`

风险：

- 对三变量、多解结构空间来说，`z` 太小，承载多模态的能力有限。
- 在没有 mutual information、mode-seeking 或 latent regularization 的情况下，增大 `z` 才更有机会表达解族结构。

## 5.4 只用 surrogate MSE 做条件一致性，目标过于“平均化”

当前颜色约束是：

- `lab_mse = MSE(predicted_lab, target_lab)`

见：`backend/app/algorithms/cgan.py:473-475`

问题：

- MSE 在 `Lab` 空间并不完全等价于感知色差。
- 训练时优化 MSE，评估时却用 `DeltaE2000`，优化目标与评价目标不一致。

## 5.5 实验对比中混入了评估协议变化

例如：

- `confirm_tune4_alpha_20260425` 最终评估 `samples_per_lab = 1000`
- `confirm_tune4_alpha_score_d3guard_20260426` 最终评估 `samples_per_lab = 64`

这会带来两个问题：

- 不能直接把最终 `mean_best_delta_e` 做绝对横向比较
- 也会影响 `solution_groups` 与 `JSD` 的稳定性

## 6. 具体优化建议

以下建议按优先级排序，优先做前 4 条。

## 6.1 第一优先级：把判别器改成真正的 conditional discriminator

建议：

- 让判别器同时输入 `design` 和 `target_lab`
- 可用简单拼接 `concat([design_norm, lab_norm])`
- 或者 projection discriminator 形式

理由：

- 这样判别器不再只约束“像真实分布”，而是约束“这个设计是否与该颜色条件匹配”
- 能从根上减轻 surrogate MSE 单独兜底的压力

预期收益：

- 降低“分布正确但颜色错”的情况
- 缓和 `alpha` 调度对训练成败的敏感性

## 6.2 第一优先级：把训练损失中的颜色项从纯 MSE 升级为更贴近评估目标的色差损失

建议：

- 至少尝试 `Lab MSE + lambda * DeltaE surrogate`
- 如果反向传播实现成本高，可先用带权 MSE：
  - 对 `L/a/b` 维度分别加权
  - 或通过校准让 surrogate 预测更接近 `DeltaE2000`

更好的方案：

- 引入 differentiable 的 `DeltaE76` 近似训练项
- 评估仍保留 `DeltaE2000`

理由：

- 当前“训练目标”和“最终评估目标”不一致，是颜色性能不稳的重要来源。

## 6.3 第一优先级：统一 checkpoint 评估协议与最终评估协议

建议：

- 固定所有实验的：
  - `checkpoint_samples_per_lab`
  - `paper_samples_per_lab`
  - `retrieval_metric`
  - `checkpoint_score_weights`
- 最好让 checkpoint 阶段和 final 阶段都使用相同的 `samples_per_lab`

折中方案：

- 训练中仍用较小样本数，但必须额外记录：
  - small-budget checkpoint score
  - large-budget checkpoint recheck score

理由：

- 不统一评估预算，会让实验排序带有较强随机性。

## 6.4 第一优先级：重新设计 `alpha` 调度，不要让条件项长期过弱

建议：

- 不再使用过长的线性 ramp
- 改成两阶段策略：
  1. 先用较强颜色约束 warm-up，快速建立条件映射
  2. 再逐步增加对抗项，恢复分布多样性

可直接尝试的配置：

- 方案 A：`max_alpha=0.2`, `alpha_ramp_epochs=2000`
- 方案 B：前 `200-500` epoch 固定较高 `alpha`，之后缓慢下降或保持
- 方案 C：根据 `lab_mse` 自适应调度，而不是纯按 epoch 线性增长

理由：

- 目前很多 best checkpoint 在 `alpha` 仍极小阶段就被选出，颜色条件学习明显不足。

## 6.5 第二优先级：降低 batch size，增加更新噪声和条件分辨率

建议：

- 从 `16384/40000` 改到 `1024-4096` 量级做对比
- 保持总训练步数接近，但增加参数更新次数

推荐实验：

- `batch_size=2048`
- `steps_per_batch=1`
- 或 `batch_size=4096, steps_per_batch=2`

理由：

- 逆设计的一对多任务通常不适合过大 batch。
- 更高更新频率往往更有利于学习局部条件结构。

## 6.6 第二优先级：增大 latent 维度，并加入多模态保持项

建议：

- `noise_dim` 从 `2` 增加到 `8` 或 `16`
- 增加 mode-seeking regularization，例如：
  - 不同 `z` 生成结果应保持一定差异
  - `||G(c,z1)-G(c,z2)|| / ||z1-z2||`

理由：

- 当前噪声表达空间太小，多解性更多是靠训练偶然性而非显式建模。

## 6.7 第二优先级：弱化边界饱和效应

建议：

- 尝试替代 `Sigmoid + min-max`
- 可改为：
  - `tanh` 后映射
  - 或输出未约束值，再加软边界惩罚

另一个工程上更稳的办法：

- 对 `d1/d3` 与 `d2` 分别做更适合的尺度变换
- 尤其 `d2` 范围远大于 `d1/d3`，统一 min-max 会放大尺度不均衡

## 6.8 第二优先级：把 surrogate 与真实物理评估做闭环校正

建议：

- 训练中周期性抽样一批生成设计
- 用真实 `evaluate_design()` 做物理回算
- 统计 surrogate 误差
- 将这部分误差反馈到：
  - regressor 再训练
  - 或 checkpoint score 的附加惩罚

理由：

- 这样可以减少生成器利用 surrogate 偏差的空间。

## 6.9 第三优先级：改进实验管理与图表命名

建议：

- 把 `loss_curve.png` 标题改成与 hinge loss 一致的表述
- 在 `metrics.json` 里单独区分：
  - checkpoint evaluation budget
  - final evaluation budget
- 对每个 run 自动生成简短 `run_summary.json`

理由：

- 便于后续快速判定实验是否可直接比较。

## 7. 下一轮最值得做的实验组合

如果只做 4 组新实验，我建议按下面顺序来：

### 实验组 1：先验证“条件判别器”是否解决主矛盾

- conditional discriminator
- `batch_size=2048`
- `noise_dim=8`
- `alpha_ramp_epochs=2000`
- `max_alpha=0.2`
- checkpoint/final 统一 `samples_per_lab=256`

目标：

- 看 `mean_best_delta_e` 和 `d2_within_5nm` 是否能同时优于当前 tune4

### 实验组 2：在实验组 1 基础上加入 mode-seeking 正则

目标：

- 看 `solution_groups` 能否提升到 `2.5-4.0` 区间，同时不显著损害 `DeltaE`

### 实验组 3：比较 `Lab MSE` 与 `DeltaE` 近似损失

目标：

- 检查训练目标与评估目标对齐后，颜色命中率是否更稳定

### 实验组 4：只做评估协议一致性对照

固定同一个 checkpoint，分别用：

- `samples_per_lab=64`
- `256`
- `1000`

目标：

- 量化当前评估预算对指标排序的影响，给后续实验设定统一标准

## 8. 现阶段推荐基线

如果现在就需要选一个“继续迭代的基线”，建议用：

- `backend/artifacts/cgan_reproduction_confirm_tune4_alpha_20260425`

理由：

- 在当前可完整比较的 run 里，它的 `mean best DeltaE`、`d2 within 5nm` 最好。
- 尽管多解性偏弱，但这是一个更稳的起点。
- 后续应在它的基础上引入 conditional discriminator 和更合理的 `alpha` 调度，而不是继续单纯堆 SN 或改 hidden dim。

## 9. 一句话总结

目前这套 cGAN 的主要问题不是“网络不够大”，而是“条件约束没有进入判别器、颜色损失与评估目标不一致、alpha 调度过慢、评估协议不统一”，所以最优先的优化方向应该是训练目标与评估体系重构，而不是继续盲目调宽度/深度。
