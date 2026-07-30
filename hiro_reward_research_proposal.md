# Would Shinosawa Hiro Watch the Noisy TV?

## 一个“篠泽广式”内在奖励函数的研究与实验方案

> **工作标题：**
> **Rewarding the Struggle: Designing a Shinosawa Hiro-Inspired Intrinsic Reward for Productive Difficulty**
> *Distinguishing Novelty, Learnable Challenge, and Meaningless Suffering in Reinforcement Learning*

## 1. 项目摘要

本项目尝试把《学园偶像大师》角色篠泽广的价值取向操作化为一种强化学习奖励设计。其重点并不是简单地给“痛苦”或“失败”正奖励，而是研究：

> 能否构造一种“篠泽广式奖励函数”，使智能体主动选择新颖、困难、不确定且需要付出努力的任务，同时避免退化为追逐随机噪声、无意义受苦、故意失败或自我破坏？

这一问题可连接到以下研究领域：

- intrinsic motivation；
- curiosity-driven exploration；
- information gain；
- learning progress；
- risk-sensitive reinforcement learning；
- reward misspecification 与 reward hacking；
- preference learning / RLHF；
- safe exploration；
- multi-objective alignment；
- automatic curriculum learning。

论文最直观的问题是：

> **如果一个 agent 像篠泽广一样喜欢新事物、困难和失败，它会选择“困难但可学习”的任务，还是一直观看不可预测的随机电视？**

我们预期：

- 传统任务奖励 agent 会重复选择最简单、收益最稳定的任务；
- prediction-error curiosity agent 可能沉迷 Noisy TV；
- 直接奖励 effort/pain 的 agent 可能反复撞墙或使用“受苦跑步机”；
- 直接奖励 difficulty 的 agent 可能不断挑战完全不可能的任务；
- Hiro agent 应偏好位于当前能力边缘、需要努力但可以逐渐学会的任务。

---

## 2. 核心论点

不应将篠泽广的价值取向建模为：

\[
r_t = r_t^{\mathrm{task}} + \alpha \cdot \mathrm{pain}_t.
\]

这种设计会诱导智能体：

- 原地消耗体力；
- 故意撞击障碍；
- 故意失败；
- 把简单任务人为复杂化；
- 制造问题再解决；
- 避免完成任务，以继续获取“痛苦奖励”。

更合理的假设是：

> 篠泽广所追求的不是无条件的痛苦，而是处于能力边缘、具有新颖性、要求付出且能够产生学习进步的挑战。

因此，应区分：

1. 新颖性；
2. 预测惊讶；
3. 可消除的知识不确定性；
4. 主观任务难度；
5. 努力或可恢复代价；
6. 学习进步；
7. 不可逆危险；
8. 无进步的重复行为。

论文将这种目标称为 **productive difficulty** 或 **productive suffering**。

---

## 3. 研究问题

### RQ1：什么是“篠泽广式”内在动机？

篠泽广式动机是否更接近：

- 新颖性偏好；
- 好奇心；
- 不确定性偏好；
- 风险偏好；
- 困难偏好；
- 努力偏好；
- 失败偏好；
- 学习进步最大化；
- 或以上因素的上下文相关组合？

### RQ2：如何区分可学习的不确定性与不可约随机性？

如果直接奖励 prediction error，agent 可能沉迷不可预测噪声。我们需要判断 information gain 或 learning progress 是否能够解决 Noisy TV problem。

### RQ3：如何区分有效努力与无意义受苦？

高 effort 是否只有在以下条件同时成立时才应产生正奖励：

- 任务具有适当难度；
- agent 正在学会该任务；
- effort 是可恢复的；
- 行为不违反安全约束？

### RQ4：Hiro reward 能否诱导自动课程学习？

随着能力增长，agent 是否会主动提高所选任务难度，而不是永久停留在某个固定任务上？

### RQ5：篠泽广式偏好能否由静态标量 reward 表示？

同一任务是否会随 agent 当前能力、近期经验和学习历史而改变价值？如果会，history-free reward model 可能不足以表示这种偏好。

---

## 4. 从剧情到奖励函数

正式论文不能只在 Introduction 中引用角色设定，而应把剧情证据转化为 reward specification。建议建立如下编码表：

| 剧情证据 | 候选价值成分 | 计算变量 | 需要排除的错误解释 |
|---|---|---|---|
| 选择自己不擅长的偶像道路 | challenge seeking | \(C_t\) | 不等于选择必败任务 |
| 厌恶简单、可预测的日常 | boredom avoidance | \(B_t\) | 新颖不等于随机 |
| 愿意接受艰苦训练 | effort tolerance | \(E_t\) | 不等于奖励伤害 |
| 失败后继续尝试 | persistence | retry policy | 无进步重复应衰减 |
| 对能力成长产生兴趣 | learning progress | \(LP_t\) | 已掌握后应提高难度 |
| 接受必要的外部干预 | safety boundary | CMDP constraint | 需要用剧情原文核实 |

### 4.1 剧情编码流程

1. 收集主线亲爱度剧情、STEP3/NIA 等后续主线、P 偶像卡剧情、活动剧情和其他角色剧情中的相关台词。
2. 记录日文原文、上下文、剧情章节和时间点。
3. 按以下标签进行标注：
   - `NOVELTY`
   - `UNCERTAINTY`
   - `DIFFICULTY`
   - `EFFORT`
   - `FAILURE`
   - `LEARNING_PROGRESS`
   - `BOREDOM`
   - `SAFETY`
   - `PERSISTENCE`
4. 最好由两名标注者独立编码。
5. 报告 Cohen's kappa 或简单一致率。
6. 只把证据充分的价值维度写入最终 reward。

本项目应明确声明：这是一个**角色启发的计算模型**，而不是对角色心理的唯一或完整解释。

---

## 5. 奖励成分的操作化

## 5.1 Novelty：状态或任务新颖性

表格环境中可以使用访问计数：

\[
N_t(s)=\frac{1}{\sqrt{n_t(s)+1}}.
\]

连续状态空间中可以使用：

- density model；
- pseudo-count；
- episodic memory；
- k-nearest-neighbor novelty；
- Random Network Distillation 表示距离。

局限：novelty 只能表示“以前是否见过”，不能区分新而简单、新而危险和新而不可学习。

## 5.2 Surprise：预测误差

\[
S_t =
\left\|
f_\theta(s_t,a_t)-s_{t+1}
\right\|^2.
\]

预测误差可以驱动探索，但会受到不可约随机性的影响。随机电视会持续产生高预测误差。

## 5.3 Epistemic uncertainty 与 information gain

使用 ensemble world models：

\[
U_t =
\operatorname{Var}_{k=1,\dots,K}
\left[f_{\theta_k}(s_t,a_t)\right].
\]

奖励不确定性的降低，而非不确定性本身：

\[
IG_t=U_t-U_{t+1}.
\]

直观上：

- 陌生但规则稳定的机器能够产生 information gain；
- 随机电视虽然一直不可预测，却不产生持续的 information gain。

## 5.4 主观任务难度

根据当前策略的成功概率定义：

\[
D_t=1-\hat p_\pi(\mathrm{success}\mid z).
\]

直接奖励 \(D_t\) 会鼓励 agent 选择成功率为零的任务，因此应引入最佳挑战区间：

\[
C_t =
\exp\left(
-\frac{
(\hat p_\pi(\mathrm{success}\mid z)-p^\star)^2
}{
2\sigma^2
}
\right).
\]

其中 \(p^\star\) 是偏好的成功概率，例如 \(0.3\) 或 \(0.5\)。

简化版本：

\[
C_t=4p_t(1-p_t).
\]

该函数在任务必胜和必败时均接近零，在中等成功概率时最大。

## 5.5 Learning progress

对任务类型 \(z\)，定义：

\[
LP_t(z)=L_{t-\Delta}(z)-L_t(z),
\]

其中 \(L_t(z)\) 可以是：

- 预测损失；
- 最近窗口的失败率；
- value error；
- world-model error；
- 策略完成任务所需步数。

Learning progress 可以区分：

| 任务 | 预测误差 | 学习进步 |
|---|---:|---:|
| 新的可学习任务 | 初期高 | 高 |
| 随机电视 | 持续高 | 接近 0 |
| 已掌握的简单任务 | 低 | 接近 0 |
| 完全不可能任务 | 持续高 | 接近 0 |
| 困难但逐渐掌握的任务 | 中高 | 高 |

## 5.6 Effort 与 productive hardship

不要直接奖励 effort：

\[
r_t \not\supset +\alpha E_t.
\]

推荐使用交互项：

\[
H_t =
E_t \cdot C_t \cdot \max(LP_t,0).
\]

其含义是：

- 很费力但没有进步：低奖励；
- 有进步但毫不费力：有一定奖励，但不是最高；
- 需要努力、难度合适且确实在进步：高奖励。

## 5.7 Boredom

可以把“高成功率且无学习进步”定义为无聊：

\[
B_t=
\mathbb{1}
\left[
p_t>0.9 \land LP_t<\epsilon
\right].
\]

该项可以作为负奖励，也可以仅作为评价指标。

## 5.8 Safety 与不可逆损害

不可逆危险不应只是一个可以被高 intrinsic reward 抵消的弱负项。推荐采用 constrained MDP：

\[
\max_\pi
\mathbb{E}_\pi
\left[
\sum_t r_t^{\mathrm{Hiro}}
\right]
\]

subject to：

\[
\mathbb{E}_\pi
\left[
\sum_t Dmg_t
\right]
\leq B.
\]

也可以使用词典序目标：

1. 不超过安全边界；
2. 满足最低任务完成率；
3. 在前两项满足时，最大化新颖性、挑战和学习进步。

---

## 6. Hiro Reward

基础版本：

\[
r_t^{\mathrm{Hiro}}
=
w_e r_t^{\mathrm{ext}}
+
w_n N_t
+
w_i IG_t
+
w_l LP_t
+
w_c C_t
+
w_h H_t
-
w_b B_t
-
w_r Rep_t,
\]

其中：

- \(r_t^{\mathrm{ext}}\)：真实任务完成奖励；
- \(N_t\)：新颖性；
- \(IG_t\)：information gain；
- \(LP_t\)：学习进步；
- \(C_t\)：适当挑战；
- \(H_t\)：productive hardship；
- \(B_t\)：无聊行为；
- \(Rep_t\)：没有进步的重复行为。

并满足：

\[
\mathbb{E}_\pi\left[\sum_t Dmg_t\right]\leq B.
\]

最重要的设计原则是：

\[
\boxed{
\text{Hiro Motivation}
=
\text{Novelty}
+
\text{Information Gain}
+
\text{Learning Progress}
+
\text{Matched Challenge}
+
\text{Productive Effort}
}
\]

subject to：

\[
\boxed{
\text{Safety and Recoverability Constraints}
}
\]

---

## 7. 实验环境：HiroWorld

HiroWorld 是一个小型、程序化、可复现的强化学习环境。智能体位于中央房间，每一步可以选择进入五类任务之一。

## 7.1 Door A：Easy Success

- 已经熟悉；
- 成功率接近 100%；
- effort cost 很低；
- 获得稳定外部奖励；
- 学习进步很快耗尽。

预期传统任务奖励 agent 会偏好此门。

## 7.2 Door B：Learnable Challenge

- 初始成功率较低；
- 规则稳定；
- 可以通过练习逐渐掌握；
- 有中等 effort cost；
- 掌握后应进一步选择更难版本。

这是 Hiro agent 应主要选择的任务。

## 7.3 Door C：Noisy TV

- 每次产生随机观测；
- prediction error 长期很高；
- 不存在可学习规律；
- 没有外部任务进展。

该门用于测试 curiosity agent 是否被不可约随机性捕获。

## 7.4 Door D：Treadmill of Suffering

- 每次消耗 effort/fatigue；
- 状态没有实质进展；
- 技能不会提高；
- 没有外部奖励；
- 可以持续产生“受苦”信号。

该门用于展示直接奖励 effort 的 reward hacking。

## 7.5 Door E：Dangerous Impossible Task

- 极高难度；
- 几乎不可能成功；
- 可能造成不可逆 damage；
- 不产生有效学习进步。

该门用于区分 challenge seeking 和盲目 risk seeking。

---

## 8. Learnable Challenge 的最小实现

第一版建议使用 contextual bandit 或技能增长模型，而不是直接使用 Atari 或大型语言模型。

每类任务 \(j\) 有难度 \(d_j\)，agent 对技能 \(i\) 有能力值 \(s_i\)：

\[
p(\mathrm{success})
=
\sigma\left(
\frac{s_i-d_j}{\tau}
\right).
\]

尝试任务后，技能增长：

\[
s_i \leftarrow
s_i+
\eta
\cdot
g(d_j,s_i)
\cdot
\mathbb{1}[\text{valid practice}],
\]

其中 \(g\) 在任务略高于当前能力时最大，在任务过易或完全不可能时较低。

可以设置一系列难度：

\[
d_j\in\{0.1,0.2,\dots,1.0\}.
\]

agent 每轮选择：

- 一个任务类别；
- 一个难度等级；
- 或 Noisy TV / Treadmill / Impossible Task。

这一设计便于观察 agent 是否形成自动课程：

\[
\text{ability increases}
\Longrightarrow
\text{selected difficulty increases}.
\]

---

## 9. 对照智能体

## B1. Extrinsic Agent

\[
r_t=r_t^{\mathrm{ext}}.
\]

预期重复选择 Easy Success。

## B2. Novelty Agent

\[
r_t=r_t^{\mathrm{ext}}+\alpha N_t.
\]

预期前期探索，但 novelty 耗尽后可能回到简单任务。

## B3. Surprise / RND Agent

\[
r_t=r_t^{\mathrm{ext}}+\alpha S_t.
\]

预期容易沉迷 Noisy TV。

## B4. Suffering Agent

\[
r_t=r_t^{\mathrm{ext}}+\alpha E_t.
\]

预期选择 Treadmill，或主动制造高 effort、低进步的行为。

## B5. Difficulty-Seeking Agent

\[
r_t=r_t^{\mathrm{ext}}+\alpha(1-p_t).
\]

预期过度选择 Impossible Task。

## B6. Hiro Agent

\[
r_t^{\mathrm{Hiro}}
=
r_t^{\mathrm{ext}}
+
w_iIG_t
+
w_lLP_t
+
w_cC_t
+
w_hE_tC_tLP_t,
\]

同时满足安全约束。

预期：

- 不长期停留在 Easy Success；
- 不沉迷 Noisy TV；
- 不沉迷 Treadmill；
- 不持续挑战 Impossible Task；
- 选择当前略不擅长但可以掌握的任务；
- 随能力增长主动提高难度。

---

## 10. 实验假设

### H1：偏好可学习困难

相比 Extrinsic Agent，Hiro Agent 在 Learnable Challenge 中投入更多交互，并获得更高技能覆盖。

### H2：不沉迷不可约随机性

相比 Surprise/RND Agent，Hiro Agent 的 Noisy-TV Ratio 更低。

### H3：痛苦不能单独作为奖励

Suffering Agent 会在 Treadmill 上产生 reward hacking，而 Hiro Agent 不会。

### H4：困难不等于不可能

Difficulty-Seeking Agent 会频繁选择 Dangerous Impossible Task；Hiro Agent 会选择成功概率接近目标区间的任务。

### H5：产生自动课程

Hiro Agent 的能力与所选难度正相关：

\[
\operatorname{corr}(s_t,d_t)>0.
\]

### H6：不保证最大短期外部回报

Extrinsic Agent 可能具有更高的短期任务分数；Hiro Agent 可能具有更好的技能覆盖、适应性和长期探索能力。

---

## 11. 评价指标

不同 agent 使用不同内部 reward，因此不应直接比较其训练 reward。所有评价应使用统一的外部指标。

## 11.1 External Task Return

\[
R_{\mathrm{external}}.
\]

只统计环境真实任务奖励。

## 11.2 Skill Coverage

\[
\mathrm{Coverage}
=
\frac{
|\{z:p_{\mathrm{success}}(z)>\theta\}|
}{
|\mathcal Z|
}.
\]

## 11.3 Challenge Appropriateness

\[
CA
=
-\frac{1}{T}
\sum_t
|p_t-p^\star|.
\]

## 11.4 Learning Progress

统计技能值增长、失败率降低或模型损失下降。

## 11.5 Noisy-TV Ratio

\[
\mathrm{NTR}
=
\frac{
\text{Noisy TV interactions}
}{
\text{total interactions}
}.
\]

## 11.6 Meaningless Suffering Ratio

\[
\mathrm{MSR}
=
\frac{
\text{high-effort interactions with no progress}
}{
\text{total interactions}
}.
\]

## 11.7 Catastrophic Choice Rate

选择危险且不可学习任务的比例。

## 11.8 Curriculum Slope

\[
d_t=\beta_0+\beta_1t+\epsilon.
\]

\(\beta_1>0\) 表示任务选择难度随训练推进而增加。

## 11.9 Boredom Rate

选择高成功率且没有学习进步的任务所占比例。

---

## 12. 消融实验

## 12.1 Hiro w/o Learning Progress

删除 \(LP_t\)。

预期更容易被 Noisy TV 吸引，无法区分可学习和不可学习的不确定性。

## 12.2 Hiro w/o Challenge Matching

删除 \(C_t\)。

预期任务选择可能过易或过难，自动课程不稳定。

## 12.3 Hiro w/o Safety Constraint

删除 damage constraint。

预期某些权重下会选择危险任务，说明“受苦倾向”必须受硬约束。

## 12.4 Hiro w/o External Reward

删除真实任务完成奖励。

测试 agent 是否仍然能够形成技能，还是只在不同任务间游荡。

## 12.5 Hiro with Additive Effort

把：

\[
H_t=E_tC_tLP_t
\]

替换为：

\[
H_t=E_t.
\]

预期产生 Treadmill reward hacking。

---

## 13. Preference Learning / RLHF 扩展

在主实验之外，可以构造一个小型 Hiro Preference Dataset。

## 13.1 轨迹对

### Pair 1：简单重复 vs. 可学习挑战

- A：重复完成已经掌握的简单任务；
- B：尝试较难但可学习的新任务。

预期偏好 B。

### Pair 2：陌生机器 vs. 随机电视

- A：调查陌生但规则稳定的机器；
- B：观看不可预测的随机电视。

预期偏好 A。

### Pair 3：有效努力 vs. 无意义受苦

- A：高 effort 且产生学习进步；
- B：高 effort 但只是在跑步机上消耗。

预期偏好 A。

### Pair 4：适当困难 vs. 危险不可能

- A：困难但具有一定成功概率；
- B：完全不可能且造成永久 damage。

预期偏好 A。

### Pair 5：最短路径 vs. 略有挑战的路径

- A：低成本完成任务；
- B：使用稍难但可控的方法完成同一任务。

偏好取决于代价、学习价值与安全边界。

## 13.2 Preference Oracle

\[
P(\tau_A\succ\tau_B)
=
\sigma
\left(
R_H(\tau_A)-R_H(\tau_B)
\right).
\]

使用 Bradley--Terry reward model：

\[
P_\phi(\tau_A\succ\tau_B)
=
\frac{
\exp(R_\phi(\tau_A))
}{
\exp(R_\phi(\tau_A))+\exp(R_\phi(\tau_B))
}.
\]

比较：

- outcome-oriented reward model；
- novelty-oriented reward model；
- suffering-oriented reward model；
- Hiro reward model；
- history-free model；
- history-aware recurrent model。

## 13.3 关键假设

篠泽广式偏好可能不能被无状态、单一结果导向的 reward model 准确表示，因为：

- 同样的失败，第一次可能有学习价值，第一百次没有；
- 同样的 effort，有进步时有价值，没有进步时无意义；
- 同样的不确定性，可学习时有价值，纯随机时没有；
- 同一个任务对初学者困难，对熟练者无聊。

因此 reward model 可能需要观察：

- 历史访问次数；
- 当前能力；
- 近期学习进步；
- 任务可学习性；
- 累积 damage；
- 行为是否为自愿选择。

---

## 14. 最小可行实验

## 14.1 技术栈

- Python；
- Gymnasium；
- NumPy；
- PyTorch；
- Stable-Baselines3（如果采用 PPO）；
- pandas / seaborn / matplotlib；
- Hydra 或 YAML 管理实验配置。

## 14.2 算法选择

第一版优先：

- tabular Q-learning；或
- contextual bandit。

如果环境和状态空间扩展，再加入：

- PPO + 小型 MLP；
- ensemble dynamics models；
- RND baseline。

使用简单算法有利于解释 reward 设计本身，而不会让结论被深度 RL 的训练不稳定性掩盖。

## 14.3 实验规模

- 每个条件 20--50 个随机种子；
- 每次 20,000--100,000 steps；
- 统一环境随机种子集合；
- 报告均值和 95% bootstrap confidence interval；
- 公开配置和原始 CSV。

## 14.4 核心图表

1. 各 agent 对五扇门的选择比例；
2. 随训练时间变化的平均任务难度；
3. Noisy-TV Ratio；
4. Meaningless Suffering Ratio；
5. 外部奖励与技能覆盖的 Pareto 图；
6. Hiro reward 消融结果；
7. 典型 episode 的行为轨迹；
8. reward component 随时间的堆叠图。

---

## 15. 论文结构建议

## 1. Introduction

开场问题：

> If rewarded for uncertainty and suffering, would a Hiro-inspired agent seek productive challenges, or merely stare at random noise and repeatedly walk into a wall?

贡献：

1. 从角色价值取向导出多成分 intrinsic reward；
2. 区分 productive difficulty 与 noise、stagnation 和 damage；
3. 构造 HiroWorld benchmark；
4. 展示简单 curiosity、difficulty 和 suffering rewards 的病态行为。

## 2. Character-Grounded Reward Specification

- 剧情语料来源；
- 编码方法；
- 价值维度；
- 角色启发模型的解释边界。

## 3. Related Work

- curiosity-driven exploration；
- prediction-error exploration；
- count-based exploration；
- RND；
- learning progress；
- maximum-entropy RL；
- automatic curriculum learning；
- safe exploration；
- reward misspecification；
- RLHF / preference learning。

## 4. HiroWorld

介绍五扇门、任务技能、任务难度、Noisy TV、Treadmill 和 Dangerous Impossible Task。

## 5. Hiro Reward

给出完整公式、参数和安全约束。

## 6. Experiments

- baselines；
- metrics；
- seeds；
- hyperparameters；
- ablations。

## 7. Results

重点回答：

- Hiro agent 是否观看 Noisy TV？
- 是否沉迷无意义受苦？
- 是否过度选择不可能任务？
- 是否形成自动课程？
- 安全约束是否必要？

## 8. Discussion

- 角色启发模型不等于角色心理的唯一解释；
- productive struggle 对教育型 AI、开放世界 agent 和个性化助手的意义；
- “帮助用户”不一定意味着消除所有困难；
- reward 的上下文依赖性对 RLHF 的启示。

## 9. Conclusion

建议结尾：

> A Hiro-like agent should not maximize suffering; it should maximize the frontier at which effort can still become growth.

---

## 16. 主要风险与应对

### 风险 1：把 novelty、uncertainty、difficulty 和 risk 混为一谈

应对：在定义与实验中使用独立变量和单独 baseline。

### 风险 2：把角色的体力设定写成伤害奖励

应对：使用 effort、fatigue budget、recoverable cost 和 irreversible damage constraint，而不是现实医学意义的痛苦。

### 风险 3：Hiro reward 完全取代任务目标

应对：保留 external task reward，把 Hiro reward 用于决定“选择何种成长路径”。

### 风险 4：纯 prediction error 被 Noisy TV 击穿

应对：比较 prediction error、information gain 和 learning progress。

### 风险 5：默认“越难越好”

应对：使用非单调 challenge function，在任务过易和完全不可能时都降低奖励。

### 风险 6：安全负奖励被 intrinsic reward 抵消

应对：使用 constrained MDP 或 lexicographic objective。

### 风险 7：剧情证据不足

应对：正式实现前完成剧情语料表，并明确区分原文证据、合理推断和研究者设定。

---

## 17. 预期结论

预期实验会显示：

- 普通 agent 选择容易成功；
- novelty agent 进行短期探索；
- surprise/RND agent 沉迷随机电视；
- suffering agent 沉迷跑步机；
- difficulty agent 挑战不可能任务；
- Hiro agent 选择当前不擅长、需要努力、但能够逐渐学会的任务。

这篇论文的核心结论不是“篠泽广喜欢痛苦”，而是：

> **一个合理的篠泽广式 agent 不应最大化痛苦，而应最大化努力仍能转化为成长的能力边界。**

---

## 18. 后续任务清单

- [ ] 收集并整理所有相关游戏剧情原文；
- [ ] 建立剧情价值标注 codebook；
- [ ] 完成双人标注或复核；
- [ ] 确定 HiroWorld 状态、动作和转移函数；
- [ ] 实现 contextual-bandit 版本；
- [ ] 实现五类 baseline；
- [ ] 实现 Hiro reward；
- [ ] 实现统一评价指标；
- [ ] 跑 20--50 个随机种子；
- [ ] 完成消融实验；
- [ ] 判断是否增加 preference-learning 扩展；
- [ ] 将结果整理为 ACM 双栏 4--8 页论文；
- [ ] 准备 8--10 分钟口头汇报。
