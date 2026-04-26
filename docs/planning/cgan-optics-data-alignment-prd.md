# PRD: cGAN、光源数据与色度计算对齐计划

## Status

- Status: Proposed
- Date: 2026-04-25
- Owner: Codex + Fangyikai

## Objective

将当前结构色反设计算法的训练参数、光源数据、色度匹配数据、特征缩放和颜色度量方式，统一对齐到“论文思路 + 实测参考数据 + 工程可复现”的状态，降低当前 pseudo colorimetry 带来的误差，并为后续 cGAN 复现实验和检索效果比较建立稳定基线。

本次 PRD 重点覆盖以下六项需求：

1. cGAN 训练参数按论文思路和当前实验经验重新整理，显式支持独立的 G/D 学习率和更新步数。
2. 用 `refer_data/D65.csv` 替换当前 pseudo D65 数据。
3. 用 `refer_data/tristimulus.csv` 替换当前 pseudo tristimulus 数据。
4. 数据 feature scaling 调整为：thickness 用 normalization，Lab 用 standardization。
5. optics 里的颜色空间换算和 Delta E 计算优先接入 `colour-science` 库。
6. nearest retrieval 增加 `Delta E 2000` 路径并做对比实验。

## Why Now

当前实现已经具备端到端运行能力，但有几处核心偏差会直接影响训练和检索质量：

- `backend/app/algorithms/optics.py` 仍使用手写 Gaussian 近似的 `_X_BAR`、`_Y_BAR`、`_Z_BAR` 和 `_ILLUMINANT_D65`。
- `backend/app/algorithms/cgan.py` 当前对 `Lab` 和 thickness 都使用 min-max normalization。
- `backend/app/algorithms/inverse_design.py` 当前 nearest retrieval 使用 `NearestNeighbors` 对 Lab 空间做欧氏距离检索，不是 `Delta E 2000`。
- `backend/scripts/train_cgan_reproduction.py` 中的 retrieval 基线同样是欧氏距离。

这些偏差会让“训练损失在降”和“颜色真实接近目标”之间出现错位，也会让 paper reproduction 的指标难以解释。

## Current State Summary

### 已有基础

- 当前 cGAN 训练已经是 G/D 交替更新，即每个 batch 内 `Evaluator` 和 `Generator` 各更新一次，等价于 `1:1` 更新节奏。
- 当前学习率实现实际上已经是 `G:D = 5:1`：
  - `Generator lr = learning_rate`
  - `Evaluator lr = learning_rate / 5`
- 当前默认值是：
  - `Generator lr = 2e-3`
  - `Evaluator lr = 4e-4`

### 与本次需求的差距

- 需要把绝对学习率改为更明确的实验基线：
  - `Generator lr = 1e-3`
  - `Discriminator/Evaluator lr = 2e-4`
- 需要把“共享 base lr 推导 D lr”的写法，改成“显式双学习率配置”。
- 需要把 pseudo D65 / pseudo tristimulus 替换为真实 CSV 数据。
- 需要把 Lab 的缩放从 min-max 改为 standardization。
- 需要把颜色换算尽量交给 `colour-science`，减少自实现公式偏差。
- 需要让 retrieval 支持 `Delta E 2000`，并与现有欧氏距离方案做 A/B 比较。

## Assumptions

1. 当前你说的 “discriminator” 对应现有代码中的 `Evaluator`，本 PRD 统一视为同一角色。
2. 当前 optics 的主波长网格先保持 `380-780 nm`、`5 nm` 步长不变，以避免一次改动过大影响 TMM 性能和已有测试。
3. `refer_data/D65.csv` 作为 illuminant 的唯一 source of truth；`refer_data/tristimulus.csv` 中最后一列 D65 仅用于交叉校验，不作为主数据源。
4. `refer_data/tristimulus.csv` 的列语义为：`wavelength_nm, x_bar, y_bar, z_bar, d65_value`。
5. 论文正文已确认“Generator/Evaluator 交替更新”和“G/D 平衡很关键”，但 Supplementary Table S2 中更细的超参数表还未在仓库内单独结构化提取；因此本期先以你的目标值 `G lr=1e-3, D lr=2e-4, update=1:1` 作为实现默认值。
6. 本期默认不改前端接口形状，除非 artifact 或调试面板需要补充少量元数据字段。

## Non-Goals

- 本期不改薄膜物理模型本身，不改 `Ag / SiO2 / Ag` 堆栈定义。
- 本期不引入新的材料数据库。
- 本期不重做 cGAN 网络结构本身，只调整训练参数接口、数据预处理和评估路径。
- 本期不强行把 retrieval 默认指标切到 `Delta E 2000`；会先支持双路径与实验比较，再决定默认值。

## User-Facing Outcome

完成后，系统应具备以下用户可感知结果：

- 颜色换算基于真实 D65 和真实 tristimulus 数据，不再依赖 pseudo 近似。
- cGAN 训练配置可明确声明：
  - 生成器学习率
  - 判别器学习率
  - G/D 每 batch 更新步数
- 训练和推理中使用更合理的特征缩放方式：
  - thickness 保持 normalization
  - Lab 改为 standardization
- 逆向设计检索支持 `Delta E 2000` 作为候选排序指标。
- 训练 artifacts 和 metrics 能清楚记录当前使用的数据源、缩放策略、颜色度量和检索策略。

## Requirements

### R1. cGAN 训练参数对齐

目标：把 cGAN 训练配置改成“显式、可复现、可比对”的形式。

实现要求：

- 在 `backend/app/algorithms/cgan.py` 中支持独立参数：
  - `generator_learning_rate`
  - `discriminator_learning_rate`
  - `steps_per_batch`
- 默认值设为：
  - `generator_learning_rate = 1e-3`
  - `discriminator_learning_rate = 2e-4`
  - `steps_per_batch = 1`
- `backend/scripts/train_cgan_reproduction.py` 暴露相应 CLI 参数，并写入 `metrics.json`。
- 日志中明确打印 G/D 的学习率与共享更新步数。

验收标准：

- [ ] 训练脚本可以单独配置 G/D 学习率。
- [ ] 训练脚本可以配置共享的 `steps_per_batch`。
- [ ] 默认值符合 `G 1e-3 / D 2e-4 / 1:1`。
- [ ] 训练 artifact 中能追踪这三个参数。

### R2. 用真实 D65 数据替换 pseudo illuminant

目标：移除 `optics.py` 中手写 `_ILLUMINANT_D65` 近似曲线。

实现要求：

- 从 `refer_data/D65.csv` 读取真实 D65 数据。
- 读取逻辑需要支持：
  - 无表头 CSV
  - `wavelength,value` 两列
  - `380-780 nm` 范围
- 若 optics 保持 `5 nm` 网格，则需要对 `D65.csv` 做确定性的重采样或插值，并在代码中固定规则。
- 保留加载后的校验：
  - 波长是否连续
  - 范围是否覆盖 `380-780`
  - 与 `tristimulus.csv` 第 5 列在 `5 nm` 采样点上偏差是否在可接受范围内

验收标准：

- [ ] `optics.py` 中不再直接定义 pseudo `_ILLUMINANT_D65`。
- [ ] D65 数据来自 `refer_data/D65.csv`。
- [ ] 数据加载错误会抛出可读异常，而不是静默回退到 pseudo 数据。

### R3. 用真实 tristimulus 数据替换 pseudo color matching functions

目标：移除 `_X_BAR`、`_Y_BAR`、`_Z_BAR` 的 Gaussian 近似。

实现要求：

- 从 `refer_data/tristimulus.csv` 读取真实 tristimulus 数据。
- 兼容 UTF-8 BOM。
- 明确列映射：
  - 第 1 列：波长
  - 第 2 列：X
  - 第 3 列：Y
  - 第 4 列：Z
  - 第 5 列：D65
- 构建统一的数据加载函数，并允许 `optics.py` 和测试代码复用。

验收标准：

- [ ] `optics.py` 中不再直接定义 pseudo `_X_BAR`、`_Y_BAR`、`_Z_BAR`。
- [ ] `spectrum_to_xyz` 的积分使用真实 tristimulus 数据。
- [ ] `_REF_WHITE` 与归一化常数由真实数据计算得到。

### R4. 特征缩放策略调整

目标：让输入输出缩放更贴合数据性质，并和后续 checkpoint 管理兼容。

实现要求：

- thickness 特征继续使用 per-dimension normalization。
- Lab 特征改为 per-dimension standardization：
  - 保存 `lab_mean`
  - 保存 `lab_std`
- 替换当前 `lab_min/lab_max` 归一化路径。
- checkpoint 中必须新增缩放元数据，至少包括：
  - `lab_scaling_type`
  - `lab_mean`
  - `lab_std`
  - `design_scaling_type`
  - `design_min`
  - `design_max`
  - `checkpoint_format_version`
- 旧 checkpoint 的兼容策略需要明确：
  - 要么提供兼容读取逻辑
  - 要么在加载时给出明确错误并提示重新训练

验收标准：

- [ ] `Lab` 训练输入不再走 min-max normalization。
- [ ] thickness 仍保持 normalization。
- [ ] 新旧 checkpoint 的行为可解释，不出现 silent mismatch。

### R5. optics 参数值和颜色变换尽量转交 `colour-science`

目标：减少手写颜色学公式，尽量复用成熟库。

实现要求：

- 引入 `colour-science` 依赖。
- 对以下能力优先使用库内实现或库内常用接口封装：
  - `XYZ -> Lab`
  - `XYZ -> sRGB`
  - `Delta E 2000`
- 项目内部保留统一包装函数，避免上层直接依赖第三方库 API。
- 若某一步仍保留自实现，需要在文档或注释里说明原因。

验收标准：

- [ ] `xyz_to_lab`、`xyz_to_srgb`、`delta_e_2000` 的主路径来自 `colour-science` 封装。
- [ ] 回归测试能证明接入后结果稳定且在预期范围内。
- [ ] 环境依赖中包含该库。

### R6. nearest retrieval 增加 `Delta E 2000` 路径

目标：让 retrieval 的“最近”定义不只依赖 Lab 欧氏距离。

实现要求：

- `backend/app/algorithms/inverse_design.py` 增加 retrieval metric 配置：
  - `euclidean_lab`
  - `delta_e_2000`
- `backend/scripts/train_cgan_reproduction.py` 中的 `nearest_retrieval` 也同步支持这两种指标。
- 对当前小规模检索网格，允许直接全量计算 `Delta E 2000`。
- 若后续样本量增大，可采用“两阶段检索”：
  - 先欧氏距离粗筛
  - 再用 `Delta E 2000` 精排

验收标准：

- [ ] runtime inverse design 可切换 retrieval metric。
- [ ] paper reproduction 脚本可切换 retrieval metric。
- [ ] metrics/artifacts 能标记当前 retrieval metric。

## Technical Design

### A. 颜色学数据加载层

建议新增独立数据加载模块，例如：

- `backend/app/algorithms/colorimetry_data.py`

职责：

- 读取 `refer_data/D65.csv`
- 读取 `refer_data/tristimulus.csv`
- 统一做波长校验、BOM 处理、数组缓存
- 对外提供：
  - `wavelengths_nm`
  - `x_bar`
  - `y_bar`
  - `z_bar`
  - `d65`

这样可以避免 `optics.py` 同时承担“物理仿真 + 参考数据管理 + 颜色空间变换”三种职责。

### B. 颜色变换封装层

建议在 `optics.py` 保留对外函数名不变，内部改为调用封装后的库函数：

- `spectrum_to_xyz`
- `xyz_to_lab`
- `xyz_to_srgb_hex`
- `delta_e_2000`

目标是减少上层模块改动范围，只替换内部实现。

### C. cGAN 缩放与 checkpoint 版本化

这是本次实现里最容易埋坑的部分。

需要明确：

- 训练时使用的 Lab 缩放方式
- 推理采样时使用的 Lab 缩放方式
- checkpoint 持久化时记录的缩放统计量
- 加载 checkpoint 时如何恢复这些统计量

若不做版本化，旧模型会在“能加载但结果不对”的情况下悄悄失真。

### D. Retrieval 指标双轨制

本次不建议直接删除欧氏距离路径，而是先做双轨：

- baseline: `euclidean_lab`
- candidate metric: `delta_e_2000`

再用同一批 target 颜色比较：

- retrieval best Delta E
- cGAN beats retrieval count
- top-k 稳定性
- 耗时差异

## Commands

以下命令是本 PRD 对应实现阶段的目标命令。

环境：

```bash
cd /home/fangyikai/code/opt-sim
conda env create -f environment.yml
conda activate opt_sim
```

后端测试：

```bash
cd /home/fangyikai/code/opt-sim
conda activate opt_sim
pytest backend/tests/test_optics_oblique_incidence.py backend/tests/test_inverse_design_runtime.py backend/tests/test_cgan_reproduction.py
```

训练 smoke run：

```bash
cd /home/fangyikai/code/opt-sim
conda activate opt_sim
python3 -u backend/scripts/train_cgan_reproduction.py \
  --dataset-source synthetic \
  --generator-learning-rate 1e-3 \
  --discriminator-learning-rate 2e-4 \
  --steps-per-batch 1
```

后续若保留 paper 复现实验：

```bash
cd /home/fangyikai/code/opt-sim
conda activate opt_sim
python3 -u backend/scripts/train_cgan_reproduction.py \
  --dataset-source paper \
  --generator-learning-rate 1e-3 \
  --discriminator-learning-rate 2e-4 \
  --steps-per-batch 1 \
  --retrieval-metric delta_e_2000
```

## Project Structure Impact

预计会影响这些文件或模块：

- `backend/app/algorithms/cgan.py`
- `backend/app/algorithms/optics.py`
- `backend/app/algorithms/inverse_design.py`
- `backend/scripts/train_cgan_reproduction.py`
- `backend/tests/test_cgan_reproduction.py`
- `backend/tests/test_inverse_design_runtime.py`
- `backend/tests/test_optics_oblique_incidence.py`
- `environment.yml`
- `backend/requirements.txt`
- `refer_data/D65.csv`
- `refer_data/tristimulus.csv`

可能新增：

- `backend/app/algorithms/colorimetry_data.py`
- `backend/tests/test_colorimetry_data.py`

## Success Criteria

- [ ] 真实 `D65.csv` 和 `tristimulus.csv` 已接入主计算路径。
- [ ] pseudo `_ILLUMINANT_D65` / `_X_BAR` / `_Y_BAR` / `_Z_BAR` 已移除。
- [ ] `colour-science` 已接入主颜色计算路径。
- [ ] cGAN 默认训练参数变为 `G 1e-3 / D 2e-4 / 1:1`。
- [ ] thickness normalization 与 Lab standardization 均已生效并写入 checkpoint metadata。
- [ ] retrieval 支持 `euclidean_lab` 与 `delta_e_2000` 两种指标。
- [ ] metrics / logs / artifacts 能追踪：
  - 数据源
  - 缩放策略
  - G/D 超参数
  - retrieval metric
- [ ] 关键单元测试和 smoke training 能通过。

## Implementation Plan

### Phase 1: 颜色参考数据与 optics 主路径对齐

Task 1.1: 建立 D65 / tristimulus 数据加载器

- Acceptance:
  - [ ] 可正确读取两个 CSV
  - [ ] 支持 BOM 和无表头
  - [ ] 带波长连续性校验
- Verify:
  - [ ] 新增单元测试验证首尾波长、数组长度、数值范围

Task 1.2: optics 接入真实数据

- Acceptance:
  - [ ] `spectrum_to_xyz` 使用真实 D65 与 tristimulus
  - [ ] `_REF_WHITE` 由真实数据推导
- Verify:
  - [ ] optics 相关测试通过
  - [ ] 平坦透过谱的白点行为符合预期

Task 1.3: 接入 `colour-science` 封装

- Acceptance:
  - [ ] `xyz_to_lab`、`xyz_to_srgb_hex`、`delta_e_2000` 主路径使用库封装
- Verify:
  - [ ] 新旧结果差异可解释
  - [ ] 关键颜色转换测试通过

### Phase 2: cGAN 缩放和训练参数重构

Task 2.1: 重构 Lab 缩放策略

- Acceptance:
  - [ ] Lab 从 min-max 改为 standardization
  - [ ] 训练、采样、checkpoint 三处一致
- Verify:
  - [ ] `test_cgan_reproduction.py` 补充缩放元数据断言

Task 2.2: 重构 G/D 超参数接口

- Acceptance:
  - [ ] 单独支持 G/D 学习率和步数
  - [ ] 默认参数符合本 PRD
- Verify:
  - [ ] 训练脚本日志与 `metrics.json` 均能反映新参数

Task 2.3: checkpoint 兼容策略

- Acceptance:
  - [ ] 新 checkpoint 含格式版本与缩放元数据
  - [ ] 旧 checkpoint 的处理逻辑明确
- Verify:
  - [ ] 加载兼容测试或错误提示测试通过

### Phase 3: retrieval 指标扩展与对比

Task 3.1: runtime inverse design 支持双 retrieval metric

- Acceptance:
  - [ ] `inverse_design.py` 可配置 `euclidean_lab` / `delta_e_2000`
- Verify:
  - [ ] 运行时测试覆盖两种分支

Task 3.2: reproduction 脚本支持双 retrieval metric

- Acceptance:
  - [ ] `nearest_retrieval` 支持切换指标
  - [ ] artifacts 记录指标名称
- Verify:
  - [ ] `candidate_samples.csv` 和 `metrics.json` 含 retrieval metric 元数据

Task 3.3: A/B 对比实验

- Acceptance:
  - [ ] 同一 target 集合下输出两种 retrieval 指标的对比结果
- Verify:
  - [ ] 至少包含 Delta E、胜率和耗时对比

### Phase 4: 文档和运行说明更新

Task 4.1: 更新后端运行文档

- Acceptance:
  - [ ] `backend/README.md` 说明新的训练参数和 retrieval metric
- Verify:
  - [ ] 命令示例可直接复制执行

Task 4.2: 更新算法操作文档

- Acceptance:
  - [ ] 文档能说明数据源、缩放、颜色学路径和 checkpoint 兼容策略
- Verify:
  - [ ] 文档与代码现状一致

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| 旧 checkpoint 与新缩放策略不兼容 | High | 引入 `checkpoint_format_version`，并显式兼容或拒绝加载 |
| 真实 D65 / tristimulus 接入后颜色输出整体漂移 | High | 增加白点、平坦谱、已知颜色样本回归测试 |
| `colour-science` 的输入输出范围与当前自实现不一致 | High | 统一封装入口，固定 XYZ 归一化约定，补回归测试 |
| `Delta E 2000` 全量检索在样本量增大后变慢 | Medium | 当前先全量算；后续样本增大时改为“粗筛 + 精排” |
| 论文补充材料的超参数与当前实验设定不完全一致 | Medium | 本期先落你的目标默认值；若后续提取到 S2，再做 source-aligned follow-up |
| `tristimulus.csv` 与 `D65.csv` 的 D65 列存在轻微差异 | Low | 用 `D65.csv` 为主，`tristimulus.csv` 仅作采样点一致性校验 |

## Open Questions

- 是否在本期直接把 runtime retrieval 默认值切为 `delta_e_2000`，还是先保留 `euclidean_lab` 为默认并输出 A/B 结果？
- 若 paper supplementary 参数后续提取成功且与本 PRD 默认值冲突，是否以 supplementary 为 source of truth 覆盖当前默认值？
- 是否需要把新的缩放策略同步写入前端调试面板或算法概览页，方便区分旧实验与新实验？

## Boundaries

- Always:
  - 使用真实参考数据替换 pseudo 数据。
  - 为新缩放策略增加测试和 checkpoint 元数据。
  - 在 artifact 中记录训练和检索配置。
- Ask first:
  - 若需要废弃旧 checkpoint 兼容。
  - 若需要把 runtime 默认 retrieval metric 直接改为 `delta_e_2000`。
  - 若需要改变波长主网格为 `1 nm`。
- Never:
  - 静默混用旧 checkpoint 与新缩放策略。
  - 在数据加载失败时悄悄回退到 pseudo D65 / pseudo tristimulus。
  - 在没有记录元数据的情况下更新训练默认参数。
