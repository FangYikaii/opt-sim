# Opt-Sim 项目的 Vibe Coding 分析报告

## 说明

这份报告专门回答一个问题：

**如果不从“算法做了什么”出发，而是从“这个项目是怎么被大模型和智能体协同开发出来的”出发，应该怎样理解它的 `vibe coding` 方式？**

这里的 `vibe coding` 我不把它理解成网络上那种很松散的说法，比如：

- 跟模型聊几句，代码就自动长出来；
- 不太看代码细节，只看跑起来有没有感觉；
- 靠提示词“氛围感”推动开发。

因为从这个项目的代码和文档看，它显然不是这种“随缘型 AI 编程”。

这个项目更接近一种**工程化的 vibe coding**：

- 大模型参与，但不凌驾于物理和工程约束之上；
- 智能体参与，但不是无限自主，而是被 API、Schema、文档、测试约束；
- 技能系统存在，但大多在仓库外的开发环境里，不是硬塞进业务代码里；
- memory 不是一个神秘“长期记忆黑盒”，而是通过文档、artifact、workspace、runtime store 等显式形式落地；
- 代码生成不是目的，**可追踪、可复现、可扩展的协作开发过程**才是目的。

所以如果要一句话概括，我会这样说：

**Opt-Sim 的 vibe coding，不是“让 AI 帮我写点代码”，而是“把 AI 当成一个可配置、可降级、可被约束的开发与决策协作者，再用软件工程把这种协作固定下来”。**

下面我会按几个层次详细讲：

1. 这个项目为什么能算是一种 vibe coding 项目。
2. 它在大模型使用上是什么风格。
3. 它在智能体设计上是什么风格。
4. 它依赖了哪些开源框架，它们在 vibe coding 中分别扮演什么角色。
5. skills 在这里是怎么体现的，哪些在仓库里，哪些在仓库外。
6. memory 在这里是怎么体现的，为什么它不是单一概念。
7. 这个项目的 vibe coding 有什么优点、边界和风险。

---

## 一、先说结论：这个项目的 vibe coding 到底是什么味道

### 1.1 它不是“让 AI 自己写一个 App”，而是“让 AI 参与一条被工程约束的研发流水线”

如果只看页面和接口，这个项目像一个光学设计工作台。

如果从开发方式看，它更像一个被 AI 深度参与构建的工程系统。这里的“AI 参与”不是停留在：

- 生成几段代码；
- 写几段文档；
- 回答几个问题。

而是更系统地进入了下面这些环节：

1. 需求理解  
   例如项目文档里不断把需求拆成结构色路线和 neural holography 路线。
2. 架构设计  
   例如 ADR、technical proposal、requirements breakdown 这套文档化设计。
3. 代码搭建  
   包括后端 API、前端 workspace、算法概览、artifact 体系。
4. 决策辅助  
   后端的 `DecisionSupport` 模块可以用 LLM，也可以回退到启发式规则。
5. 开发环境协作  
   仓库外的 Codex / Claude 技能和权限配置，说明开发过程本身也被 agent 化。

所以这个项目的 vibe coding，不是一个点，而是一条链。

### 1.2 它的核心气质：文档先行、契约先行、状态显式、失败可回退

从代码和配置看，这个项目的 AI 协作开发有四个非常明显的特征。

#### 特征一：文档先行

项目不是先胡乱写代码，再补文档。

相反，文档体系非常重：

- `README.md`
- `docs/architecture/technical-proposal.md`
- `docs/planning/requirements-breakdown.md`
- `docs/planning/neural-holography-citl-prd.md`
- `docs/architecture/colorimetry-and-cgan-alignment-design.md`
- `docs/decisions/ADR-001...`
- `docs/decisions/ADR-002...`

这说明开发过程里，大模型和人不是在“边写边猜”，而是在借助文档持续外化共识。

从 vibe coding 的角度看，这其实非常重要，因为大模型最怕的不是任务难，而是上下文飘。

文档越清楚，AI 越容易持续在正确轨道上协作。

#### 特征二：契约先行

项目里后端 `schemas/`、OpenAPI、前端生成类型这一整套东西说明：

- AI 不是想返回什么就返回什么；
- 页面也不是想怎么取字段就怎么取；
- 所有参与者都被数据契约约束。

这意味着这里的 vibe coding 不是“自然语言全覆盖”，而是“自然语言驱动 + 结构化契约收口”。

#### 特征三：状态显式

项目里有几个很关键的“状态显式化”设计：

- `algorithm_overview.py`
- `runtime_store.py`
- `WorkspaceDetail`
- `ArtifactDetail`
- `DecisionSupport`

它们共同作用的结果是：

**系统自己会描述自己当前处于什么状态。**

这对 vibe coding 特别有价值，因为模型协作最怕“黑箱状态”。

如果系统不知道：

- 当前活跃模型是谁；
- 当前实验进展如何；
- 当前 run 处于哪个阶段；
- 当前 agent 是 live 还是 fallback；

那么人和模型都只能靠猜。

这个项目显然在主动避免这件事。

#### 特征四：失败可回退

这一点是整个项目 vibe coding 味道里最工程化的一部分。

后端对 AI 决策支持不是“成了就成，不成就炸”，而是设计了：

- `live`
- `fallback`
- `disabled`

三种状态。

如果 LLM 没有 API Key，或者调用失败，或者返回格式不对，系统会退回本地启发式判断。

这件事非常关键。

因为真正成熟的 vibe coding，不是迷信模型，而是承认：

- 模型会失败；
- 接口会抖动；
- 输出会不合法；

然后提前为这种失败设计好工程退路。

---

## 二、从大模型角度看：这个项目是怎样“用模型”的

这一部分讲最直接的问题：项目里的 LLM 到底被用在了哪里，没被用在了哪里。

---

## 2.1 它没有让大模型替代物理引擎，而是把大模型放在“语言和决策接口层”

从 `backend/app/agent.py` 和 `backend/app/config.py` 可以看得很清楚：

这个项目里的大模型，主要不是用来替代：

- TMM 正向仿真；
- cGAN 训练；
- CGH 波前传播；
- Delta E 计算；
- 候选评分。

也就是说，项目没有做那种最危险的事情：

“把本来应该精确计算的部分交给 LLM 自由发挥。”

相反，它把大模型放在几个更适合它的位置：

1. 总结需求  
   `_call_agent_summary`
2. 输出决策建议  
   `_call_agent_decision_support`
3. 把技术结果翻译成人能快速消费的语言

这是一种很成熟的 AI 使用方式。

因为在这种物理约束很强的项目里，大模型最擅长的是：

- 归纳；
- 解释；
- 结构化语言输出；
- 辅助人做判断。

而不是精确代替数值算法。

### 2.1.1 用最通俗的话讲，它在做什么

可以这样理解：

- 真正“算结果”的，是物理仿真和逆向设计算法；
- 真正“解释结果、建议下一步”的，是 LLM 或 heuristic agent；
- 真正“把这些东西打包成一次任务”的，是后端 Agent 编排层。

所以大模型在这里更像一个：

**懂上下文、会写结构化意见书的副驾驶。**

而不是主驾驶。

---

## 2.2 模型接口设计：它用的是 OpenAI-compatible 风格，而不是绑死某一家

`backend/app/config.py` 和 `.env` 体现出一件很重要的事：

项目没有把自己写死在某个唯一模型供应商上。

它的 Agent 配置支持：

- `OPT_SIM_AGENT_API_BASE_URL`
- `OPT_SIM_AGENT_API_KEY`
- `OPT_SIM_AGENT_MODEL`
- `OPT_SIM_AGENT_PROVIDER_LABEL`

默认还支持：

- `CODEX_BASE_URL`
- `CODEX_API_KEY`
- `CODEX_MODEL`

这说明它采用的是一种**OpenAI-compatible 接口思路**。

### 2.2.1 这意味着什么

意味着项目把大模型当成一种“可插拔服务”，而不是某个产品 logo。

从 vibe coding 角度看，这特别重要，因为 AI 开发环境通常变化很快：

- 今天用 OpenAI；
- 明天可能用 DashScope；
- 后天可能用别的兼容接口。

如果系统把模型供应商硬编码进业务逻辑，项目会非常脆弱。

现在这种做法明显更成熟：

- provider label 只是展示用；
- 真正调用的是兼容 `/chat/completions` 的接口；
- model 名和 base URL 可配置；
- API key 也是环境变量输入。

### 2.2.2 当前这个仓库实际配置了什么

从 `.env` 可以看到，当前开发环境里配置的是：

- `OPT_SIM_AGENT_PROVIDER_LABEL=Bailian`
- `OPT_SIM_AGENT_API_BASE_URL=https://coding.dashscope.aliyuncs.com/v1`
- `OPT_SIM_AGENT_MODEL=glm-5`

这说明项目作者在实际开发中，确实是在把大模型当成“可替换的底座”，而不是只围绕单一模型写死。

这本身就是一种 vibe coding 经验积累：

**模型会变，但系统的协作接口要稳定。**

---

## 2.3 为什么说这里的 LLM 使用方式很克制，也很聪明

### 2.3.1 它只把模型放在“高语义、低数值风险”的环节

比如：

- 总结需求，用一句工程化描述说清楚当前 run 是什么；
- 看候选和约束，生成 headline、summary、rationale、risks、nextAction；

这些任务的共同特点是：

- 需要语义组织；
- 需要上下文理解；
- 对结果格式有要求；
- 但不要求模型自己做高精度数值推导。

这正是大模型相对稳定、性价比高的使用区间。

### 2.3.2 它不是“让模型自由聊天”，而是要求返回结构化 JSON

`_call_agent_decision_support` 里对 system prompt 的要求非常明确：

- 只返回 JSON；
- 必须包含固定 key；
- `confidence` 必须是 `high/medium/low`；
- `rationale` 和 `risks` 都必须是 1 到 3 条短句。

这很关键。

因为在 vibe coding 里，一个常见误区是：

- 让模型输出“看起来很聪明的一段话”；
- 但系统没法稳定接收。

这个项目显然在避免这种情况。

它要求模型的输出必须被系统“消费”。

这意味着大模型不是聊天装饰，而是被纳入软件接口的一部分。

### 2.3.3 它还专门做了非法输出兜底

`_extract_json_object` 和 `llm -> heuristic fallback` 这套逻辑说明：

- 项目并不相信模型一定会听话；
- 它假设模型可能返回带 markdown fence 的内容；
- 也假设模型可能返回不合法 JSON；
- 更假设整个请求可能直接失败。

于是：

- 能提取就提取；
- 不能提取就 fallback；
- 不会把整个系统拖死。

这就是高质量 vibe coding 和低质量 vibe coding 的一个分水岭：

**不是只会“接上模型”，而是会“设计模型失效后的系统行为”。**

---

## 三、从智能体角度看：这个项目是怎样做 Agent 化的

这里要先澄清一个常见误解。

很多人一听“智能体”，就以为一定是：

- 可以自己拆任务；
- 可以自己找工具；
- 可以自己长时间循环；
- 可以自己改代码、跑实验、做决策。

但项目里的 Agent 不是这种“泛化大智能体”，而是一种**任务编排型 Agent**。

---

## 3.1 这个项目里的 Agent，本质上是“任务总控层”

`backend/app/agent.py` 的角色很像一个总控器。

它干的事情是：

1. 接收用户设计请求
2. 判断设计模式
3. 调用对应算法入口
4. 取当前 active model 信息
5. 取当前 agent configuration 信息
6. 组织 timeline
7. 组织 decision support
8. 返回完整的 `DesignRunResponse`

也就是说，这里的 Agent 不是纯推理体，而是：

**把多种底层能力装配成一个统一任务对象的协调层。**

这点很重要，因为它说明项目对“智能体”的理解是偏工程落地的，不是概念秀。

### 3.1.1 从结构上看，它像什么

可以把这个 Agent 理解成一个小型导演：

- 算法模块是演员；
- LLM 是文案和判断顾问；
- runtime store 是场记；
- workspace/page 是观众看到的舞台；
- Agent 则负责把这些人按场次组织起来。

没有 Agent，系统仍然有很多底层功能，但这些功能不会自然组成一次“有起承转合”的 run。

---

## 3.2 它的 Agent 设计有一个很强的特点：按 design mode 分支，而不是一把梭

项目当前明确支持两个 `designMode`：

- `structural-color`
- `neural-holography`

Agent 层会根据这个模式分支。

这说明它的 agent 化不是“所有任务交给一个万能提示词”，而是：

- 先识别任务类型；
- 再选择对应算法和上下文；
- 再生成对应的候选、约束、timeline、导出估算。

### 3.2.1 为什么这很像成熟 vibe coding 的产物

因为真实项目里，随着功能增长，最容易崩的是“所有场景塞进一个 if else 黑洞”。

而这个项目在比较早的阶段就开始做：

- 模式分层；
- 路线分支；
- 输出对象统一；
- 内部逻辑分流。

这说明 AI 参与开发并没有把系统带向混乱，反而借助文档和 schema，把复杂度显式化了。

---

## 3.3 它的 Agent 输出不是“最终结论”，而是“工作区素材”

这一点非常关键。

`DesignRunResponse` 包含的不只是一个答案，而是：

- `activeRun`
- `draft`
- `targets`
- `timeline`
- `candidates`
- `constraints`
- `exportEstimate`
- `activeModel`
- `agentConfiguration`
- `decisionSupport`

这意味着 Agent 层产出的不是一个单值，而是一整个工作区上下文。

### 3.3.1 这和很多轻量 agent demo 有什么不同

很多 demo 的 Agent 是这样的：

- 输入一句话；
- 输出一段话。

而这里不是。

这里 Agent 的产物是一个结构化状态对象，后面要被：

- runtime store 保存；
- 前端 Workspace 渲染；
- Artifact 面板引用；
- 时间线组件展示；
- Inspector 面板解释；

这说明 Agent 已经不是“接口附属品”，而是系统状态生产者。

从 vibe coding 的角度看，这很高级，因为它代表：

**AI 不只是帮忙写代码，也已经被用来构建软件运行时的“任务语义骨架”。**

---

## 四、开源框架的使用：它们在这套 vibe coding 里分别扮演什么角色

这一部分不只是列清单，而是解释“为什么这些框架组合在一起，很适合 AI 协作开发”。

---

## 4.1 FastAPI：把 AI 参与开发的后端变得更容易结构化

项目后端使用 FastAPI。

从 vibe coding 角度，FastAPI 特别适合这种项目，原因有几条。

### 4.1.1 它天然适合 schema 驱动

这个项目特别依赖：

- `DesignRequest`
- `DesignRunResponse`
- `WorkspaceDetail`
- `AlgorithmOverview`

FastAPI 和 Pydantic 的组合，天然鼓励先把数据结构说清楚。

这对人类工程师有利，对大模型也特别有利。

因为大模型写接口代码时，最怕目标对象不明确。

一旦对象形状稳定，模型更容易：

- 补充字段；
- 调整路由；
- 生成前后端一致代码；
- 不乱改接口语义。

### 4.1.2 它天然适合导出 OpenAPI

这个项目还导出了 `backend/openapi.json`，前端又用生成类型去消费。

这很适合 vibe coding，因为 AI 协作最怕的一个问题是：

- 后端改了；
- 前端不知道；
- 类型没同步；
- bug 到页面上才暴露。

OpenAPI 可以把这个同步链路显式化。

### 4.1.3 它天然适合快速拼装原型，但不至于太松散

FastAPI 开发速度很快，这对 AI 辅助编码很友好。

但它又不是那种完全无约束的脚本式后端。

所以它正好卡在一个很好的点：

- 足够快；
- 足够结构化；
- 足够容易被 AI 理解和扩展。

---

## 4.2 Pydantic：它其实是整个 vibe coding 的“边界护栏”

很多人会把 Pydantic 看成后端小配件，但在这个项目里，它的作用更像护栏。

### 4.2.1 为什么说它是护栏

因为当大模型参与开发时，系统会天然面临更多“看起来合理、其实不严谨”的改动。

例如：

- 把字段名改了；
- 把数值字段变成字符串；
- 漏了某个必须项；
- 枚举值写错；
- 返回结构少一层。

Pydantic 在这里的价值就是：

**你可以有很多开发灵活性，但最终都必须过结构化数据校验。**

### 4.2.2 对 vibe coding 最大的意义

它相当于告诉 AI：

- 你可以帮我生成；
- 但你不能随便发明协议。

这就是把“生成自由度”限制在“可运行系统边界”以内。

---

## 4.3 Vue 3 + TypeScript + Vite：为什么前端这套也很适合 AI 协作开发

### 4.3.1 组件化特别适合拆给模型逐块生成

前端当前拆成很多明确组件：

- `AlgorithmOverviewPanel.vue`
- `AgentTimeline.vue`
- `ArtifactDetailPanel.vue`
- `InspectorPanel.vue`
- `AppShell.vue`
- `OperationsGuidePanel.vue`

这种组件化拆分非常适合 vibe coding。

因为它让任务天然变成：

- “做一个 timeline 组件”
- “补一个 artifact detail panel”
- “把 algorithm overview 接到首页”

相比之下，如果整个页面写成一个巨型文件，模型更容易失控。

### 4.3.2 TypeScript 再次提供边界

和 Pydantic 在后端扮演的角色类似，TypeScript 在前端也扮演了“防止 AI 写飞”的角色。

它的价值不是让代码显得高级，而是：

- 当接口变了，前端会报错；
- 当字段假设不成立，组件更早暴露问题；
- 当 generated types 更新后，使用方要同步改。

这非常符合高质量 vibe coding 的基本思路：

**让错误尽早暴露，而不是让模型把错一路铺到运行时。**

### 4.3.3 Vite 为什么也重要

Vite 让前端开发反馈特别快。

对 AI 辅助开发来说，快反馈非常关键。

因为很多 AI 生成的 UI/前端改动，本质上都需要：

- 立刻运行；
- 立刻看效果；
- 立刻调整。

所以像 Vite 这种低摩擦开发环境，其实很适合和 AI 协作。

---

## 4.4 PyTorch、NumPy、scikit-learn、colour-science：这些库在 vibe coding 里起什么作用

### 4.4.1 PyTorch：让模型实验仍然可控

项目里的 cGAN 和相关训练逻辑基于 PyTorch。

这意味着即使开发过程有 AI 参与，训练部分依然建立在一个成熟、透明、可调、可调试的数值框架上。

这很重要，因为 vibe coding 最怕两种极端：

1. 全手搓，开发太慢；
2. 全黑箱，无法控制。

PyTorch 给出的正是一个中间地带：

- 表达力强；
- 训练流程清晰；
- 容易加入日志、checkpoint、超参数；
- 模型结构能被人和 AI 一起读懂。

### 4.4.2 NumPy：让物理和信号处理逻辑保持直接

不论是 TMM 还是 CGH 传播，NumPy 让代码保持：

- 贴近公式；
- 容易验证；
- 方便测试。

这点对 AI 协作也很重要。

因为如果所有逻辑都藏在过于复杂的抽象里，模型很难准确修改。

NumPy 这种相对直接的数值风格，反而很适合在 agent 协作中做小步精改。

### 4.4.3 scikit-learn：补足“简单基线”和“验证工具”

项目里用到了：

- `NearestNeighbors`
- `DBSCAN`

这说明作者没有陷入“所有东西都要深度学习化”的惯性。

这其实也是一种很好的 vibe coding 气质：

**该简单时就简单，该有基线时就有基线。**

因为 AI 最容易带来的坏习惯之一，就是把事情做复杂。

而这里保留检索基线、保留聚类工具，说明系统仍然尊重简单可靠的方法。

### 4.4.4 colour-science：让“AI 写出来的颜色系统”不只是看着像

颜色学最怕被瞎写。

引入 `colour-science`，相当于把一块容易被模型写成“近似示意代码”的地方，交给成熟库来兜底。

这很符合成熟 vibe coding 的一个原则：

**AI 负责连接、组织、扩展，核心标准实现尽量用可信库托底。**

---

## 五、skills：这个项目的 skills 到底怎么理解

这一部分最容易误解，所以要讲清楚：

**这个仓库本身几乎没有项目内 skills 代码，但它明显运行在一个有强 skills 文化的开发环境里。**

这两句话不矛盾。

---

## 5.1 先说最直接结论：项目内没有完整业务级 skills 仓库，但项目外有明显的技能化开发环境

从当前仓库内容看：

- `.codex/` 里主要是环境备份，不是项目 skill 定义；
- `.claude/` 里主要是权限配置；
- 仓库里没有一整套像“技能目录 + skill markdown + skill runtime”的项目内实现。

所以如果问题是：

“这个项目有没有把 skills 当成业务代码的一部分？”

答案是：

**没有明显这么做。**

但如果问题是：

“这个项目是不是在一个强技能化的 AI 开发环境里被开发的？”

答案则是：

**非常明显是。**

证据包括：

- `.claude/settings.local.json` 里允许 `Skill(update-config)`；
- 项目文档多次提到 Claude Code inspired workspace；
- 后端配置中支持 `CODEX_*` 环境变量；
- 开发流程明显符合 agent skill 化工作方式：文档、计划、实现、验证、回写。

这说明 skills 更多存在于**开发环境层**，而不是业务仓库层。

---

## 5.2 从 vibe coding 视角，skills 在这里更像“开发习惯模块”

我会把这里的 skills 分成两类理解。

### 5.2.1 第一类：平台级 skills

也就是不属于项目代码，但强烈影响项目开发方式的那些技能。

例如：

- 如何先读文档再动手
- 如何按 schema 写接口
- 如何小步提交改动
- 如何先补测试再改逻辑
- 如何生成 OpenAPI 再同步前端类型

这些东西不一定写在这个仓库里，但从项目形态看，它们显然在开发中起作用了。

### 5.2.2 第二类：项目内隐式 skills

虽然仓库里没有叫 `skills/` 的业务目录，但其实项目本身已经把很多“技能模板”固化成代码结构了。

例如：

- `algorithm_overview.py`
  其实是一种“状态解释技能”。
- `runtime_store.py`
  其实是一种“运行时记忆技能”。
- `DecisionSupport`
  其实是一种“结果解释与建议技能”。
- `requirements-breakdown.md`
  其实是一种“任务拆解技能”。
- `ADR` 体系
  其实是一种“技术决策记录技能”。

也就是说，虽然没有以 `skill` 命名，它已经把很多协作经验硬化成了工程结构。

---

## 5.3 如果非要总结一句：这个项目的 skills 设计重心不在业务运行时，而在开发运行时

很多 agent 产品会把 skills 设计成：

- 运行时调用某个 skill；
- 业务里执行某个 skill。

这个项目不是这种重心。

它的重心更偏向：

- 在开发时让 AI 有稳定工作套路；
- 在系统里给 Agent 明确职责边界；
- 在项目里通过文档、契约、组件、脚本沉淀“半技能化模式”。

所以更准确的说法是：

**它不是把 skills 做成业务插件，而是把 skills 做成开发范式。**

---

## 六、memory：这个项目里的 memory 不是一个东西，而是四层

很多人说 AI 项目的 memory，会默认想到：

- 长期向量记忆；
- 用户偏好记忆；
- 自动回忆系统。

但在这个项目里，如果只这么理解，会严重误判。

这个项目的 memory 更适合拆成四层。

---

## 6.1 第一层：模型短上下文 memory

这是最普通的一层，就是：

- 当前请求内容；
- 当前候选；
- 当前约束；
- 当前设计模式；
- 当前活跃模型和配置；

这些会被组织后发给 LLM。

比如 `_call_agent_decision_support` 会把：

- requirementText
- designMode
- targetHex
- thetaDeg
- polarization
- top 3 candidates
- top 5 constraints

一起发给模型。

这就是最基础的“上下文记忆”。

它不是长期记忆，但对单次推理至关重要。

---

## 6.2 第二层：工作区运行时 memory

这层非常关键，也是这个项目最像“工程化记忆”的部分。

`runtime_store.py` 做的事情，本质上就是：

**把一次 run 的工作区状态存起来。**

包括：

- activeRun
- draft
- timeline
- candidates
- constraints
- artifacts
- decisionSupport
- activeModel

这相当于系统对“刚刚发生过什么”有显式记忆。

它不是向量库，但它对实际工作更重要，因为它让系统具备：

- 刷新页面后还能回到上一次 run；
- 查看历史 run；
- 追踪当前 artifact 属于谁；
- 把一次 AI 协同结果沉淀成可回看的对象。

### 6.2.1 为什么这比“聊天历史记忆”更有工程价值

因为真正的软件系统，不是靠聊天上下文持续存在的。

聊天历史很容易丢、很难结构化，也不适合前后端共享。

而 runtime store 把工作区状态对象化了。

这是一种更成熟的 memory 设计：

**把记忆从“对话残留”提升为“系统状态资产”。**

---

## 6.3 第三层：artifact 和 experiment 记忆

项目里大量 artifact 目录和 `metrics.json`、checkpoint、CSV、analysis summary，本质上也是 memory。

只是它们记住的不是“一个对话”，而是：

- 某次训练做了什么；
- 某次实验达到了什么指标；
- 哪个 checkpoint 被选中；
- 当前最佳实验是谁；
- 某次运行导出了什么。

这层 memory 的特点是：

- 更偏长期；
- 更偏可复盘；
- 更偏机器和人都能读。

### 6.3.1 Algorithm Overview 就是“把实验记忆转成人类可读状态”

`algorithm_overview.py` 会从 artifact 里读取历史实验结果，再汇总成：

- currentAssessment
- trainingConclusion
- gpuTrainingSummary
- activeModel
- headlineMetrics

这其实就是把原始 experiment memory，重新翻译成操作 memory。

换句话说：

- artifact 是冷记忆；
- algorithm overview 是热记忆。

---

## 6.4 第四层：文档记忆

这是最容易被忽视，但其实对 vibe coding 最重要的一层。

项目文档体系本身，就是团队和 AI 的长期外置记忆。

例如：

- technical proposal 记住了架构目标；
- ADR 记住了为什么这么选；
- requirements breakdown 记住了任务分阶段逻辑；
- PRD 记住了新需求分支；
- 当前这类分析报告记住了“项目是怎么被理解的”。

### 6.4.1 为什么文档也是 memory，而不是附属材料

因为对大模型来说，最可靠的长期记忆不是模型自己“想起来”，而是有人把事实写在明确位置。

文档越完善，AI 越能稳定接棒。

所以从 vibe coding 视角，文档不是额外负担，而是：

**最可控、最可共享、最可回溯的长期记忆系统。**

---

## 6.5 如果要用一句话总结这里的 memory 设计

我会这样说：

**这个项目没有把 memory 神秘化，而是把 memory 拆成了上下文、工作区状态、实验产物、文档知识四层，并且尽量都显式落盘。**

这是一种很强的工程思维。

---

## 七、这个项目的 vibe coding 不是“放飞”，而是“被多重约束驯化”

这一部分非常重要，因为它解释了为什么这个项目看起来有 AI 味，但又不散。

---

## 7.1 第一重约束：物理约束

不论是结构色路线还是全息路线，底层都不是自由文本游戏，而是：

- TMM
- D65 / tristimulus
- Delta E
- ASM / Fresnel
- PSNR / SSIM

这意味着 AI 不可能随便编一个“看起来像结果”的故事就混过去。

系统最后要落到数值和物理评价上。

---

## 7.2 第二重约束：Schema 约束

AI 写出来的代码和运行时输出，都要过：

- Pydantic
- OpenAPI
- generated types

这意味着 AI 不能随便改协议。

---

## 7.3 第三重约束：测试约束

项目测试覆盖：

- optics
- colorimetry
- inverse design runtime
- cGAN reproduction
- cgh simulation
- workspace api
- algorithm overview api

这意味着 AI 改完代码后，不是“看起来像”就行，而是必须在已有测试语义下站住。

---

## 7.4 第四重约束：文档与计划约束

项目里需求拆解、技术方案、PRD 都非常清楚。

所以 AI 并不是在空白画布上发挥，而是在已定义路线图上推进。

这会大大降低“看起来很聪明，实际上方向偏了”的风险。

---

## 八、从 vibe coding 的成熟度来看，这个项目做对了什么

这一部分更偏评价。

---

## 8.1 它做对了：没有让 AI 替代正确性底座

最值得肯定的一点是：

AI 被放在了解释、编排、辅助、生成代码的位置上；
而真正的物理正确性、颜色正确性、训练评估、数值传播，仍然落在可验证代码和标准库上。

这是非常对的。

---

## 8.2 它做对了：把“AI 参与”也做成了可观测状态

前端甚至会显示：

- 当前决策模式是 `LLM assisted` 还是 `Heuristic fallback`
- 当前模型是谁
- 当前 agent configuration 是什么

这说明 AI 不再是幕后不可见黑箱，而是系统可观察组件。

这非常先进。

---

## 8.3 它做对了：把复杂协作外化成工作区

工作区、timeline、artifact、decision support 这些对象，实际上都在帮助人理解：

- AI 做了什么；
- 物理算了什么；
- 当前结果为什么长这样；
- 下一步该干什么。

也就是说，项目在努力把 AI 协作结果“界面化”。

这是 vibe coding 想走向团队协作时必须经过的一步。

---

## 8.4 它做对了：保留简单基线和人工可读结构

项目没有把所有内容都深度神经网络化。

它保留：

- nearest retrieval
- heuristic fallback
- algorithm overview
- operation steps
- artifact metadata

这些看似“土”，其实非常关键。

因为 AI 项目最容易犯的错之一，就是过早把所有内容都做成黑箱自动化。

这个项目目前还没有陷进去。

---

## 九、它的 vibe coding 还有哪些边界和风险

说优点也要说边界。

---

## 9.1 第一类风险：skills 很强，但主要还在开发环境层，不在项目内显式沉淀

当前项目明显运行在一个带 Codex / Claude 风格 skill 的环境里，但仓库本身没有完整的项目内 skills 体系。

这会带来一个风险：

- 当前作者和当前 AI 环境配合得很好；
- 但换环境、换团队成员后，很多“隐含工作流技巧”可能无法完整迁移。

所以后续如果要进一步工程化，值得考虑把一部分关键开发 skill 显式沉淀为仓库内文档或脚本模板。

---

## 9.2 第二类风险：memory 很多，但还偏文件型，未来可能需要更强查询层

现在的 memory 很强，但主要是：

- 文档文件
- runtime json
- artifact 目录

这已经很好，但随着项目变大，可能会遇到：

- 历史实验太多，overview 不够；
- 某类 artifact 不易检索；
- 决策建议无法跨 run 分析；
- 某些经验只能在文档里全文搜索。

未来可以考虑增强：

- 实验索引；
- 产物索引；
- 更强的 run 检索或 dashboard。

---

## 9.3 第三类风险：LLM 辅助层现在偏“单次调用”，还不是完整多步 Agent

当前系统里的 LLM 决策支持相对成熟，但仍然偏：

- 单次 summarization；
- 单次 JSON decision；

还不是那种：

- 多步工具调用；
- 自动回看 artifact；
- 自动生成后续任务；
- 自动分析实验趋势。

这不是缺点，而是阶段性选择。

但如果以后真要继续往“智能设计助手”推进，Agent 可能还会继续扩展。

---

## 十、如果必须用最通俗的话向外行解释：这个项目是怎么 vibe coding 的

我会这样说：

### 10.1 第一层

它不是让 AI 代替工程师，而是让 AI 成为工程师旁边一个很会读文档、会组织结果、会写结构化接口和说明的人。

### 10.2 第二层

它不是让 AI 自己决定物理规律，而是让 AI 在物理模型已经确定的前提下，加速开发、解释和决策。

### 10.3 第三层

它不是只靠聊天记住上下文，而是把“记忆”落到：

- 文档
- artifact
- runtime workspace
- schema

这些可见、可查、可测试的地方。

### 10.4 第四层

它不是神化大模型，而是给大模型准备了：

- 配置入口；
- 结构化输入；
- 结构化输出；
- 非法输出兜底；
- 无模型时的 fallback。

这才是成熟 vibe coding 的样子。

---

## 十一、最终总结：Opt-Sim 的 vibe coding 方法论是什么

如果把整份报告浓缩成几句话，我会给出下面这个总结。

### 11.1 它的方法论不是“AI 生成代码”，而是“AI 参与系统化研发”

这个项目真正体现出的，不是某个 prompt 写得多花，而是这样一套方法：

1. 用文档把需求和路线写清楚；
2. 用 schema 把接口和对象写清楚；
3. 用标准框架把系统骨架搭清楚；
4. 用 AI 参与代码生成、结果解释和决策辅助；
5. 用 artifact、runtime store、workspace 把过程记下来；
6. 用测试和 fallback 防止 AI 协作失控。

### 11.2 它的核心不是“自由”，而是“被约束的自由”

它给 AI 足够多的参与空间：

- 帮忙设计
- 帮忙实现
- 帮忙解释
- 帮忙产出文档

但同时又用：

- 物理模型
- schema
- 测试
- artifact
- fallback

把这种自由圈在一个可控边界里。

这就是它最值得学习的地方。

### 11.3 如果要一句最通俗的话收尾

**Opt-Sim 的 vibe coding，本质上不是“让大模型替我开发”，而是“把大模型变成一个受文档、契约、测试和物理规律约束的协作开发者”。**

这比“随手让 AI 写点代码”高一个层级。

---

## 附：本报告判断所依据的主要仓库证据

大模型与 Agent 相关：

- `backend/app/config.py`
- `backend/app/agent.py`
- `.env`

运行时状态与 memory 相关：

- `backend/app/runtime_store.py`
- `backend/app/algorithm_overview.py`
- `backend/artifacts/*`

契约与前后端协作相关：

- `backend/app/schemas/*.py`
- `backend/openapi.json`
- `frontend/src/api/generated-types.ts`
- `frontend/src/services/workspace-service.ts`

工作区与 AI 结果展示相关：

- `frontend/src/pages/HomePage.vue`
- `frontend/src/pages/WorkspacePage.vue`
- `frontend/src/components/AgentTimeline.vue`
- `frontend/src/components/ArtifactDetailPanel.vue`
- `frontend/src/components/InspectorPanel.vue`

开发环境与 skills 痕迹相关：

- `.claude/settings.json`
- `.claude/settings.local.json`
- `.codex/`

文档与外置长期记忆相关：

- `README.md`
- `docs/README.md`
- `docs/architecture/technical-proposal.md`
- `docs/planning/requirements-breakdown.md`
- `docs/planning/neural-holography-citl-prd.md`
- `docs/decisions/ADR-001-architecture-and-model-choices.md`
- `docs/decisions/ADR-002-use-real-colorimetry-data-and-versioned-cgan-scaling.md`
